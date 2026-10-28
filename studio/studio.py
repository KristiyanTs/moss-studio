"""MOSS Studio: a queue + gallery web UI for MOSS-Image.

Owns the model in-process (via MOSS-Image's ModelRunner), runs queued jobs one at a time,
persists the queue in SQLite, and serves every image saved under --output-dir.
Login is a single password from $MOSS_PASS; the app refuses to start without it.
"""

import argparse
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from starlette.background import BackgroundTask

log = logging.getLogger("studio")
HERE = Path(__file__).resolve().parent
FOLDER_RE = re.compile(r"^[0-9a-f]{32}$")
IMAGE_RE = re.compile(r"^\d{2}\.png$")
FILE_RE = re.compile(r"^\d{2}\.(png|mp4)$")
SESSION_COOKIE = "studio_session"
SESSION_TTL = 7 * 24 * 3600
MAX_PROMPT = 20000
UPLOAD_MAX = 30 * 1024 * 1024


# ---------- generation backends ----------

# ---------- Wan2GP jobs: photo -> video, extend a video, edit a photo ----------
# Run headless (`wgp.py --process settings.json`) as a subprocess, so its memory is freed afterwards.

WAN_VIDEO_MODEL = "ti2v_2_2_fastwan"  # Wan 2.2 5B distilled to 3 steps; 720p, 121 frames (~5 s at 24 fps)
WAN_EDIT_MODEL = "moss_image_edit_hq"  # MOSS-Image SFT + community Edit HQ v4 adapter: localized, detail-preserving edits
WGP_STEP_RE = re.compile(r"(\d+)/(\d+) \[")
REALISM = "Realistic live-action footage, natural lifelike motion, steady camera. "
WGP_KINDS = ("video", "extend", "edit")


def video_resolution(width, height):
    if width >= height * 1.2:
        return "1280x720"
    if height >= width * 1.2:
        return "720x1280"
    return "960x960"


def edit_resolution(width, height):
    """Source shape at ~1 megapixel, sides multiples of 16."""
    scale = (1024 * 1024 / (width * height)) ** 0.5
    return f"{max(256, round(width * scale / 16) * 16)}x{max(256, round(height * scale / 16) * 16)}"


def wgp_settings(p, source):
    if p["kind"] == "video":
        return {"model_type": WAN_VIDEO_MODEL, "prompt": REALISM + p["prompt"], "image_prompt_type": "S", "image_start": str(source),
                "resolution": video_resolution(p["width"], p["height"]), "seed": p["seed"]}
    if p["kind"] == "extend":
        return {"model_type": WAN_VIDEO_MODEL, "prompt": REALISM + p["prompt"], "image_prompt_type": "V", "video_source": str(source),
                "resolution": video_resolution(p["width"], p["height"]), "seed": p["seed"]}
    return {"model_type": WAN_EDIT_MODEL, "prompt": p["prompt"], "image_refs": [str(source)], "image_mode": 1, "video_prompt_type": "KI",
            "resolution": edit_resolution(p["width"], p["height"]), "seed": p["seed"]}


def run_wgp(settings, work, wan_dir, wan_python, progress, outputs):
    """Run one Wan2GP task headless; returns the newest output file matching `outputs` (glob patterns)."""
    import subprocess

    work.mkdir(parents=True)
    (work / "settings.json").write_text(json.dumps(settings))
    cmd = [wan_python, "wgp.py", "--process", str(work / "settings.json"), "--output-dir", str(work)]
    proc = subprocess.Popen(cmd, cwd=wan_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    try:
        progress(0, desc="Starting")
        buf = ""
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            if ch not in "\r\n":
                buf += ch
                continue
            line, buf = buf.strip(), ""
            if not line:
                continue
            tail = (tail + [line])[-30:]
            m = WGP_STEP_RE.search(line)
            if m and int(m.group(2)) > 0:
                progress((int(m.group(1)), int(m.group(2))), desc="Generating")
            elif "ownload" in line:
                progress(0, desc="Downloading model (first time only)")
            elif "oading" in line:
                progress(0, desc="Loading model")
            else:
                progress(0, desc="Working")
        code = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    found = sorted((f for pat in outputs for f in work.rglob(pat)), key=lambda f: f.stat().st_mtime)
    if code != 0 or not found:
        log.error("Wan2GP failed (exit %s):\n%s", code, "\n".join(tail))
        hint = next((l for l in reversed(tail) if "rror" in l), tail[-1] if tail else "no output")
        raise RuntimeError(f"Failed: {hint[:300]}")
    return found[-1]


class WanWorker:
    """Keeps Wan2GP (and its last model) loaded between consecutive video/edit jobs. Stopped before MOSS
    runs so the two never compete for GPU memory. Any worker failure falls back to a one-off subprocess."""

    def __init__(self, wan_dir, wan_python, output_dir):
        self.wan_dir, self.wan_python = wan_dir, wan_python
        self.out_dir = Path(output_dir) / ".wan-out"
        self.proc = None

    def stop(self):
        if self.proc:
            self.proc.kill()
            self.proc.wait()
            self.proc = None

    def _start(self, progress):
        import subprocess

        progress(0, desc="Loading model")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        script = Path(__file__).with_name("wan_worker.py")
        self.proc = subprocess.Popen([self.wan_python, str(script), self.wan_dir, str(self.out_dir)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1, cwd=self.wan_dir)
        if not self._read(progress, (0, "Loading model")).get("ready"):
            raise RuntimeError("worker did not start")

    def _read(self, progress, last):
        """Next protocol message; re-reports `last` each second so Stop is noticed even while Wan2GP is busy."""
        import select

        while not select.select([self.proc.stdout], [], [], 1.0)[0]:
            progress(last[0], desc=last[1])
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("worker exited")
        return json.loads(line)

    def run(self, settings, progress, outputs):
        if not self.proc or self.proc.poll() is not None:
            self._start(progress)
        try:
            self.proc.stdin.write(json.dumps({"settings": settings}) + "\n")
            self.proc.stdin.flush()
            last = (0, "Starting")
            progress(*last[:1], desc=last[1])
            while True:
                msg = self._read(progress, last)
                if "progress" in msg:
                    step, total = msg["progress"]
                    last = ((step, total) if total else 0, "Generating" if total else (msg.get("phase") or "Working").capitalize())
                    progress(last[0], desc=last[1])
                elif "done" in msg:
                    break
        except BaseException:
            self.stop()  # includes Stop from the UI: kill the generation
            raise
        files = [Path(f) for f in msg["files"] if any(Path(f).match(pat) for pat in outputs) and Path(f).is_file()]
        if not msg["done"] or not files:
            raise RuntimeError("Failed: " + ("; ".join(msg["errors"]) or "no output")[:300])
        return files[-1]


def run_wgp_job(p, output_dir, folder, progress, wan_dir, wan_python, worker=None):
    """Produce the gallery folder for a video / extend / edit job."""
    from PIL import Image

    source = Path(output_dir) / p["source"]
    work = folder / ".wan"
    if p["kind"] == "edit":
        result = wgp_run(p, source, work, wan_dir, wan_python, progress, ("*.png", "*.jpg", "*.jpeg", "*.webp"), worker)
        with Image.open(result) as im:
            im.convert("RGB").save(folder / "01.png")
        result.unlink(missing_ok=True)
        images, video = ["01.png"], None
    else:
        result = wgp_run(p, source, work, wan_dir, wan_python, progress, ("*.mp4",), worker)
        os.replace(result, folder / "01.mp4")
        poster = source if p["kind"] == "video" else source.parent / "01.png"
        shutil.copyfile(poster, folder / "01.png")
        images, video = ["01.png"], "01.mp4"
    shutil.rmtree(work, ignore_errors=True)
    return images, video


def wgp_run(p, source, work, wan_dir, wan_python, progress, outputs, worker):
    settings = wgp_settings(p, source)
    if worker:
        try:
            return worker.run(settings, progress, outputs)
        except JobStopped:
            raise
        except Exception:
            log.exception("Wan2GP worker failed; retrying as a one-off process")
            worker.stop()
    return run_wgp(settings, work, wan_dir, wan_python, progress, outputs)


def save_wgp_record(folder, p, start, images, video):
    record = {"schema_version": 2, "kind": p["kind"], "model": WAN_EDIT_MODEL if p["kind"] == "edit" else WAN_VIDEO_MODEL,
              "parameters": p, "elapsed_seconds": round(time.monotonic() - start, 3), "images": images}
    if video:
        record["video"] = video
    (folder / "parameters.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


# Every image is a photograph: this style is added to the prompt the model sees (the gallery keeps the
# user's own prompt). Best mode also steers away from drawn styles; Fast mode has no negative prompt.
PHOTO_STYLE = ("A real photograph, photorealistic, shot on a professional camera with natural lighting, "
               "realistic skin and material textures, true-to-life colors. ")
PHOTO_NEGATIVE = "anime, manga, cartoon, illustration, drawing, painting, sketch, comic, 3d render, cgi, digital art"


def model_prompt(p):
    return PHOTO_STYLE + p["prompt"]


class StageTimer:
    """Wraps a progress callback and records seconds spent per stage (desc prefix), for parameters.json."""

    def __init__(self, progress):
        self.progress, self.stages, self.current, self.t = progress, {}, None, time.monotonic()

    def __call__(self, value, desc="", **kw):
        stage = desc.split(" · ")[0] if desc else self.current
        if stage != self.current:
            self.mark()
            self.current = stage
        return self.progress(value, desc=desc, **kw)

    def mark(self):
        now = time.monotonic()
        if self.current:
            self.stages[self.current] = round(self.stages.get(self.current, 0) + now - self.t, 2)
        self.t = now
        return self.stages


class RapidAdapter:
    """rapid-moss 4-step adapter (darask0/MOSS-Image_4-step_adapter_lora), switchable on the shared pipeline.

    Mirrors the adapter's inference.py: an unmerged fp32 LoRA on every transformer projection, fp32
    replacement time-path weights, and a CFG-free data-space sampler. Switching back restores the
    original bf16 time-path weights and silences the LoRA, so Best mode runs the untouched model.
    """
    TIME_MODULES = ("time_embedding", "shared_block_modulation", "video_head_modulation")

    def __init__(self, path):
        from safetensors.torch import load_file

        self.path = Path(path)
        self.time_path = load_file(self.path / "time_embedding.safetensors")
        dataspace = load_file(self.path / "dataspace.safetensors")
        self.Q, self.codes = dataspace["latent_mix.Q"], dataspace["step_codes"]
        self.lora = None
        self.embeds = {}  # prompt -> (embeddings, mask), oldest first
        self.pipe = None
        self.original = None
        self.active = False

    def _attach(self, pipe):
        import torch
        from safetensors.torch import load_file

        tr = pipe.transformer
        if self.lora is None:
            self.lora = load_file(self.path / "lora.safetensors", device="cuda")
        state = tr.state_dict()
        self.original = {k: state[k].detach().clone().cpu() for k in self.time_path}
        for i, layer in enumerate(tr.layers):
            for name, lin in layer.proj.items():
                key = f"diffusion_model.layers.{i}.proj.{name}"
                down, up = self.lora[key + ".lora_down.weight"], self.lora[key + ".lora_up.weight"]
                scale = float(self.lora[key + ".alpha"]) / down.shape[0]

                def hook(module, inputs, output, down=down, up=up, scale=scale):
                    if not self.active:
                        return output
                    x = inputs[0].to(down.device, torch.float32)
                    return output + (scale * ((x @ down.t()) @ up.t())).to(output.device, output.dtype)
                lin.register_forward_hook(hook)
        self.pipe, self.active = pipe, False

    def set_active(self, pipe, active):
        import torch

        if self.pipe is not pipe:  # first use, or the runner reloaded the model
            self._attach(pipe)
            self.embeds.clear()
        if active == self.active:
            return
        tr = pipe.transformer
        for name in self.TIME_MODULES:
            getattr(tr, name).to(torch.float32 if active else torch.bfloat16)
        result = tr.load_state_dict(self.time_path if active else self.original, strict=False)
        assert not result.unexpected_keys, result.unexpected_keys
        self.active = active

    def sample(self, pipe, prompt, width, height, seed, progress):
        import torch

        steps = len(self.codes)
        device = pipe._execution_device
        hit = self.embeds.get(prompt)
        if hit is None:  # repeats (×N, "More like this") reuse the encoding: no 17 GB text-encoder trip to the GPU
            progress(0, desc="Encoding prompt")
            hit = pipe.encode_prompt(prompt)
            self.embeds[prompt] = hit
            while len(self.embeds) > 16:
                self.embeds.pop(next(iter(self.embeds)))
        emb, mask = hit
        Q, codes = self.Q.to(device), self.codes.to(device)
        z = torch.randn(1, 128, height // 16, width // 16, generator=torch.Generator("cpu").manual_seed(seed)).to(device)
        x0 = torch.zeros_like(z)
        for r in range(steps):
            x = z + torch.einsum("oc,bchw->bohw", Q, x0)
            x0 = z - pipe.transformer(x, codes[r], emb, mask).sample.float()
            progress((r + 1, steps), desc=f"Sampling · step {r + 1}/{steps}")
        progress(0, desc="Decoding image")
        vae = pipe.vae
        lat = x0.to(vae.dtype)
        bn, eps = vae.bn, vae.config.batch_norm_eps
        lat = lat * (bn.running_var.to(device) + eps).sqrt().view(1, -1, 1, 1) + bn.running_mean.to(device).view(1, -1, 1, 1)
        b, c, h, w = lat.shape
        lat = lat.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
        image = vae.decode(lat, return_dict=False)[0].clamp(-1, 1)
        return pipe.image_processor.postprocess(image, output_type="pil")[0]


class MossBackend:
    def __init__(self, model, output_dir, adapter_dir=None, wan_dir=None, wan_python=None, compile_model=False):
        from gradio_app import GenerationConfig, ModelRunner, load_pipeline, profile_defaults

        self.Config = GenerationConfig
        self.load_pipeline = load_pipeline
        self.model = model
        self.output_dir = Path(output_dir)
        self.runner = ModelRunner(output_dir, local_files_only=True)
        self.offload = profile_defaults("auto")[0]
        self.wan_dir, self.wan_python = wan_dir, wan_python
        self.compile = compile_model  # torch.compile the DiT blocks: same output, faster steps after a one-time warm-up
        self.wan = WanWorker(wan_dir, wan_python, output_dir) if wan_dir else None
        self.adapter = RapidAdapter(adapter_dir) if adapter_dir and Path(adapter_dir, "lora.safetensors").is_file() else None
        if adapter_dir and not self.adapter:
            log.warning("Fast mode unavailable: no adapter in %s", adapter_dir)

    def config(self, p):
        if p.get("kind") in WGP_KINDS:
            return None
        config = self.Config(
            prompt=model_prompt(p), negative_prompt=", ".join(x for x in (PHOTO_NEGATIVE, p["negative_prompt"]) if x),
            width=p["width"], height=p["height"], steps=p["steps"], cfg=p["cfg"],
            seed=p["seed"], num_images=p["num_images"], offload=self.offload, compile_transformer=self.compile,
        )
        config.validate()
        return config

    def generate(self, p, progress):
        if p.get("kind") in WGP_KINDS:
            return self.generate_wgp(p, progress)
        if self.wan:
            self.wan.stop()  # MOSS needs the GPU memory Wan2GP is holding
        try:
            return self._generate_moss(p, progress)
        except Exception as exc:
            if not (self.compile and type(exc).__module__.startswith(("torch._dynamo", "torch._inductor"))):
                raise
            log.exception("Compiled model failed; continuing without compilation")
            self.compile = False
            with self.runner.lock:
                self.runner.unload()
            return self._generate_moss(p, progress)

    def _generate_moss(self, p, progress):
        if p.get("mode") == "fast":
            return self.generate_fast(p, progress)
        timer = StageTimer(progress)
        with self.runner.lock:
            if self.adapter and self.runner.pipe is not None:
                self.adapter.set_active(self.runner.pipe, False)
            paths, meta, _ = self.runner.generate(self.model, self.config(p), progress=timer)
        # Show the user's prompt in the gallery; keep what the model actually saw alongside it.
        record = json.loads(Path(meta).read_text())
        params = record["parameters"]
        record["model_prompt"], record["model_negative_prompt"] = params["prompt"], params["negative_prompt"]
        params.update(prompt=p["prompt"], negative_prompt=p["negative_prompt"], mode="best")
        record["timings"] = timer.mark()
        Path(meta).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        return Path(paths[0]).parent.name

    def generate_wgp(self, p, progress):
        import torch

        if not (self.wan_dir and Path(self.wan_dir, "wgp.py").is_file()):
            raise RuntimeError("The video/edit tool (Wan2GP) is not installed on this server.")
        with self.runner.lock:
            start = time.monotonic()
            if self.runner.pipe is not None:  # give the GPU to Wan2GP; MOSS weights stay in system memory
                self.runner.pipe.maybe_free_model_hooks()
            torch.cuda.empty_cache()
            folder = self.output_dir / uuid.uuid4().hex
            folder.mkdir(parents=True)
            try:
                images, video = run_wgp_job(p, self.output_dir, folder, progress, self.wan_dir, self.wan_python, self.wan)
            except BaseException:
                shutil.rmtree(folder, ignore_errors=True)
                raise
            save_wgp_record(folder, p, start, images, video)
            return folder.name

    def generate_fast(self, p, progress):
        import torch

        if not self.adapter:
            raise RuntimeError("Fast mode is not installed on this server.")
        runner = self.runner
        progress = timer = StageTimer(progress)
        with runner.lock, torch.no_grad():
            start = time.monotonic()
            key = (str(self.model), self.offload, self.compile)
            if runner.key != key:  # same cache key as the runner, so Best mode reuses this load
                progress(0, desc="Loading model")
                runner.unload()
                runner.pipe = self.load_pipeline(self.model, self.offload, self.compile, True)
                runner.key = key
            self.adapter.set_active(runner.pipe, True)
            seed = p["seed"] if p["seed"] != -1 else secrets.randbelow(2**32)
            try:
                images = [self.adapter.sample(runner.pipe, model_prompt(p), p["width"], p["height"], seed + i, progress)
                          for i in range(p["num_images"])]
            except torch.cuda.OutOfMemoryError as exc:
                runner.unload()
                raise ValueError("GPU memory exhausted. Try a smaller size.") from exc
            folder = self.output_dir / uuid.uuid4().hex
            folder.mkdir(parents=True)
            names = []
            for i, image in enumerate(images):
                names.append(f"{i + 1:02d}.png")
                image.save(folder / names[-1])
            record = {
                "schema_version": 2, "model": Path(self.model).name, "adapter": "darask0/MOSS-Image_4-step_adapter_lora",
                "parameters": {**p, "seed": seed, "steps": len(self.adapter.codes), "cfg": 0},
                "sampler": "rapid-moss data-space", "timings": timer.mark(), "model_prompt": model_prompt(p), "elapsed_seconds": round(time.monotonic() - start, 3), "images": names,
            }
            (folder / "parameters.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
            progress(1, desc="Complete")
            return folder.name


class FakeBackend:
    """Local UI development without a GPU: writes solid-colour PNGs after a short delay."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def generate_wgp(self, p, progress):
        import subprocess
        from PIL import Image, ImageOps

        start = time.monotonic()
        for step in range(30):
            time.sleep(0.1)
            progress((step + 1, 30), desc="Generating")
        folder = self.output_dir / uuid.uuid4().hex
        folder.mkdir(parents=True)
        source = self.output_dir / p["source"]
        video = None
        if p["kind"] == "edit":
            with Image.open(source) as im:
                ImageOps.invert(im.convert("RGB")).save(folder / "01.png")
        else:
            poster = source if p["kind"] == "video" else source.parent / "01.png"
            shutil.copyfile(poster, folder / "01.png")
            seconds = "2" if p["kind"] == "video" else "4"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-loop", "1", "-i", str(poster), "-t", seconds,
                            "-vf", "scale=256:-2,zoompan=z='1+0.002*on':d=96", "-pix_fmt", "yuv420p", str(folder / "01.mp4")], check=True)
            video = "01.mp4"
        save_wgp_record(folder, p, start, ["01.png"], video)
        return folder.name

    def config(self, p):
        if p.get("kind") not in WGP_KINDS and not p["prompt"].strip():
            raise ValueError("Enter a prompt.")

    def generate(self, p, progress):
        import struct
        import zlib

        if p.get("kind") in WGP_KINDS:
            return self.generate_wgp(p, progress)
        if p.get("mode") == "fast":
            p = {**p, "steps": 4, "cfg": 0}
        for step in range(p["steps"]):
            time.sleep(0.05)
            progress((step + 1, p["steps"]), desc=f"Sampling · step {step + 1}/{p['steps']}")
        folder = self.output_dir / uuid.uuid4().hex
        folder.mkdir(parents=True)
        w, h = p["width"] // 8, p["height"] // 8
        colour = bytes(secrets.token_bytes(3))
        raw = b"".join(b"\x00" + colour * w for _ in range(h))

        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
        names = []
        for i in range(p["num_images"]):
            (folder / f"{i + 1:02d}.png").write_bytes(png)
            names.append(f"{i + 1:02d}.png")
        seed = p["seed"] if p["seed"] != -1 else secrets.randbelow(2**32)
        record = {"parameters": {**p, "seed": seed}, "elapsed_seconds": p["steps"] * 0.05, "images": names}
        (folder / "parameters.json").write_text(json.dumps(record))
        return folder.name


# ---------- queue ----------

class JobStopped(Exception):
    pass


class Queue:
    def __init__(self, db_path, backend):
        self.backend = backend
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.progress = {}  # job id -> {"fraction": float|None, "desc": str}
        self.stop_id = None
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self.lock, self.db:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY, pos REAL, status TEXT, params TEXT,"
                " folder TEXT, error TEXT, created REAL, started REAL, finished REAL)"
            )
            # A job that was running when the process died goes back to the front of the queue.
            self.db.execute("UPDATE jobs SET status='queued', started=NULL WHERE status='running'")

    def add(self, params, repeat):
        """New work goes to the front of the waiting line (right after whatever is generating), batch order kept."""
        with self.lock, self.db:
            front = self.db.execute("SELECT MIN(pos) FROM jobs WHERE status='queued'").fetchone()[0]
            front = 0 if front is None else front
            for i in range(repeat):
                self.db.execute(
                    "INSERT INTO jobs (pos, status, params, created) VALUES (?, 'queued', ?, ?)",
                    (front - repeat + i, json.dumps(params), time.time()),
                )
        self.wake.set()

    def move(self, job_id, index):
        """Put a waiting job at position `index` among waiting jobs (0 = next to run)."""
        with self.lock, self.db:
            ids = [r["id"] for r in self.db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY pos")]
            if job_id not in ids:
                raise HTTPException(409, "Only waiting jobs can be moved.")
            ids.remove(job_id)
            ids.insert(max(0, min(index, len(ids))), job_id)
            base = self.db.execute("SELECT COALESCE(MIN(pos), 0) FROM jobs WHERE status='queued'").fetchone()[0]
            self.db.executemany("UPDATE jobs SET pos=? WHERE id=?", [(base + i, jid) for i, jid in enumerate(ids)])

    def update_params(self, job_id, changes):
        with self.lock, self.db:
            row = self.db.execute("SELECT params FROM jobs WHERE id=? AND status='queued'", (job_id,)).fetchone()
            if not row:
                raise HTTPException(409, "Only waiting jobs can be changed.")
            old = json.loads(row["params"])
            if old.get("kind") in WGP_KINDS:
                params = clean_source_params({**old, "prompt": changes.get("prompt", old["prompt"])}, self.output_dir)
            else:
                params = clean_params({**old, **changes})
            self.backend.config(params)
            self.db.execute("UPDATE jobs SET params=? WHERE id=?", (json.dumps(params), job_id))

    def cancel(self, job_id):
        with self.lock, self.db:
            cur = self.db.execute(
                "UPDATE jobs SET status='cancelled', finished=? WHERE id=? AND status='queued'", (time.time(), job_id)
            )
            if cur.rowcount:
                return
            row = self.db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row or row["status"] != "running":
                raise HTTPException(409, "This image already finished.")
            self.stop_id = job_id  # the worker's progress callback stops it at the next step

    def retry(self, job_id):
        with self.lock:
            row = self.db.execute("SELECT params FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Job not found.")
        self.add(json.loads(row["params"]), 1)

    def dismiss(self, job_id):
        with self.lock, self.db:
            self.db.execute("DELETE FROM jobs WHERE id=? AND status IN ('done', 'failed', 'cancelled')", (job_id,))

    def clear(self):
        with self.lock, self.db:
            self.db.execute("DELETE FROM jobs WHERE status IN ('done', 'failed', 'cancelled')")

    def snapshot(self):
        with self.lock:
            active = self.db.execute(
                "SELECT * FROM jobs WHERE status IN ('queued', 'running') ORDER BY status='queued', pos"
            ).fetchall()
            recent = self.db.execute(
                "SELECT * FROM jobs WHERE status IN ('done', 'failed', 'cancelled') ORDER BY finished DESC LIMIT 30"
            ).fetchall()
        jobs = []
        for r in [*active, *recent]:
            job = {k: r[k] for k in ("id", "status", "folder", "error", "created", "started", "finished")}
            job["params"] = json.loads(r["params"])
            job["progress"] = self.progress.get(r["id"]) if r["status"] == "running" else None
            jobs.append(job)
        return jobs

    def _claim(self):
        with self.lock, self.db:
            row = self.db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY pos LIMIT 1").fetchone()
            if row:
                self.db.execute("UPDATE jobs SET status='running', started=? WHERE id=?", (time.time(), row["id"]))
            return row

    def _finish(self, job_id, status, folder=None, error=None):
        with self.lock, self.db:
            self.db.execute(
                "UPDATE jobs SET status=?, folder=?, error=?, finished=? WHERE id=?",
                (status, folder, error, time.time(), job_id),
            )
        self.progress.pop(job_id, None)
        if self.stop_id == job_id:
            self.stop_id = None

    def run_forever(self):
        while True:
            try:
                self._step()
            except Exception:  # e.g. disk full while recording the result; never let the worker die
                log.exception("Worker error")
                time.sleep(5)

    def _step(self):
        row = self._claim()
        if row is None:
            self.wake.wait(5)
            self.wake.clear()
            return
        job_id = row["id"]
        self.progress[job_id] = {"fraction": None, "desc": "Starting"}

        def report(value, desc="", **_):
            if self.stop_id == job_id:
                raise JobStopped()
            fraction = value[0] / value[1] if isinstance(value, tuple) and value[1] else None
            self.progress[job_id] = {"fraction": fraction, "desc": desc}

        try:
            folder = self.backend.generate(json.loads(row["params"]), report)
            self._finish(job_id, "done", folder=folder)
        except JobStopped:
            self._finish(job_id, "cancelled")
        except Exception as exc:  # the job fails; the worker keeps going
            log.exception("Job %s failed", job_id)
            self._finish(job_id, "failed", error=str(exc) or type(exc).__name__)


# ---------- gallery ----------

class Gallery:
    def __init__(self, output_dir):
        self.root = Path(output_dir)
        self.cache = {}  # folder -> (mtime, item)

    def items(self):
        items = []
        for meta in self.root.glob("*/parameters.json"):
            folder = meta.parent.name
            if not FOLDER_RE.match(folder):
                continue
            mtime = meta.stat().st_mtime
            cached = self.cache.get(folder)
            if not cached or cached[0] != mtime:
                try:
                    record = json.loads(meta.read_text())
                except (OSError, ValueError):
                    continue
                p = record.get("parameters") if isinstance(record, dict) else None
                if not isinstance(p, dict) or not isinstance(record.get("images"), list):
                    continue
                images = [n for n in record.get("images", []) if IMAGE_RE.match(n)]
                video = record.get("video") if FILE_RE.match(str(record.get("video", ""))) else None
                cached = (mtime, {
                    "id": folder, "created": mtime, "images": images, "video": video,
                    "elapsed": record.get("elapsed_seconds"),
                    "params": {k: p.get(k) for k in
                               ("prompt", "negative_prompt", "width", "height", "steps", "cfg", "seed", "num_images", "mode", "kind", "source", "file_name")},
                })
                self.cache[folder] = cached
            items.append({**cached[1], "loved": (meta.parent / ".loved").exists()})
        items.sort(key=lambda item: item["created"], reverse=True)
        return items

    TRASH_TTL = 24 * 3600

    def _check(self, folder):
        if not FOLDER_RE.match(folder):
            raise HTTPException(404)

    def delete(self, folder):
        # Moved to .trash so it can be undone; purged after TRASH_TTL.
        self._check(folder)
        src, trash = self.root / folder, self.root / ".trash"
        if not src.is_dir():
            raise HTTPException(404)
        trash.mkdir(exist_ok=True)
        dst = trash / folder
        if dst.exists():
            shutil.rmtree(dst)
        os.rename(src, dst)
        os.utime(dst)
        self.cache.pop(folder, None)
        for old in trash.iterdir():
            if old.is_dir() and time.time() - old.stat().st_mtime > self.TRASH_TTL:
                shutil.rmtree(old, ignore_errors=True)

    def love(self, folder, loved):
        self._check(folder)
        if not (self.root / folder / "parameters.json").is_file():
            raise HTTPException(404)
        mark = self.root / folder / ".loved"
        if loved:
            mark.touch()
        else:
            mark.unlink(missing_ok=True)

    def restore(self, folder):
        self._check(folder)
        src, dst = self.root / ".trash" / folder, self.root / folder
        if not src.is_dir() or dst.exists():
            raise HTTPException(404)
        os.rename(src, dst)

    THUMB_WIDTH = 640

    def thumb(self, folder, name):
        """640px-wide WebP preview, cached under .thumbs; falls back to the full PNG without Pillow."""
        if not IMAGE_RE.match(name):
            raise HTTPException(404)
        src = self.file(folder, name)
        dst = self.root / ".thumbs" / folder / (name[:-4] + ".webp")
        if dst.is_file():
            return dst, "image/webp"
        try:
            from PIL import Image
        except ImportError:
            return src, "image/png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB")
            if im.width > self.THUMB_WIDTH:
                im = im.resize((self.THUMB_WIDTH, round(im.height * self.THUMB_WIDTH / im.width)), Image.LANCZOS)
            tmp = dst.with_suffix(".tmp")
            im.save(tmp, "WEBP", quality=82, method=4)
        os.replace(tmp, dst)
        return dst, "image/webp"

    def file(self, folder, name):
        if not FOLDER_RE.match(folder) or not FILE_RE.match(name):
            raise HTTPException(404)
        path = self.root / folder / name
        if not path.is_file():
            raise HTTPException(404)
        return path


# ---------- params ----------

def clean_source_params(body, output_dir):
    """Jobs that start from a gallery item: video (from a photo), extend (a video), edit (a photo)."""
    kind = body.get("kind")
    if kind not in WGP_KINDS:
        raise HTTPException(422, "Unknown job type.")
    source = body.get("source")
    folder, _, name = source.partition("/") if isinstance(source, str) else ("", "", "")
    pattern = r"^\d{2}\.mp4$" if kind == "extend" else IMAGE_RE.pattern
    if not (FOLDER_RE.match(folder) and re.match(pattern, name) and (Path(output_dir) / folder / name).is_file()):
        raise HTTPException(422, "Pick a video to extend." if kind == "extend" else "Pick a photo from the gallery.")
    prompt = body.get("prompt") or ""
    if not isinstance(prompt, str) or len(prompt) > MAX_PROMPT:
        raise HTTPException(422, "Prompt is too long.")
    if kind == "edit" and not prompt.strip():
        raise HTTPException(422, "Describe the change you want.")
    width, height = body.get("width"), body.get("height")
    if not all(isinstance(v, int) and not isinstance(v, bool) and 64 <= v <= 8192 for v in (width, height)):
        raise HTTPException(422, "Invalid size.")
    seed = body.get("seed", -1)
    if not isinstance(seed, int) or isinstance(seed, bool) or not -1 <= seed < 2**31:
        seed = -1
    return {"kind": kind, "mode": "edit" if kind == "edit" else "video", "source": f"{folder}/{name}", "prompt": prompt,
            "width": width, "height": height, "seed": seed if seed != -1 else secrets.randbelow(2**31), "negative_prompt": "", "num_images": 1}


def clean_params(body):
    def num(name, kind, lo, hi, default):
        value = body.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise HTTPException(422, f"{name} must be a number.")
        if kind is int and value != int(value):
            raise HTTPException(422, f"{name} must be a whole number.")
        value = kind(value)
        if not lo <= value <= hi:
            raise HTTPException(422, f"{name} must be between {lo} and {hi}.")
        return value

    prompt, negative = body.get("prompt"), body.get("negative_prompt", "")
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(422, "Enter a prompt.")
    if not isinstance(negative, str) or len(prompt) > MAX_PROMPT or len(negative) > MAX_PROMPT:
        raise HTTPException(422, "Prompt is too long.")
    params = {
        "prompt": prompt, "negative_prompt": negative,
        "width": num("width", int, 256, 2048, 1024), "height": num("height", int, 256, 2048, 1024),
        "steps": num("steps", int, 1, 100, 50), "cfg": num("cfg", float, 0, 20, 3.0),
        "seed": num("seed", int, -1, 2**53 - 1, -1), "num_images": num("num_images", int, 1, 4, 1),
        "mode": body.get("mode", "best"),
    }
    if params["mode"] not in ("best", "fast"):
        raise HTTPException(422, "Mode must be best or fast.")
    origin = body.get("origin")
    if isinstance(origin, str) and re.match(r"^[0-9a-f]{32}/\d{2}\.png$", origin):
        params["origin"] = origin
    if params["width"] % 16 or params["height"] % 16:
        raise HTTPException(422, "Width and height must be multiples of 16.")
    return params


# ---------- app ----------

async def read_json(request):
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(400, "Invalid request.")
    if not isinstance(body, dict):
        raise HTTPException(400, "Invalid request.")
    return body


def create_app(queue, gallery, password):
    fast_available = getattr(queue.backend, "adapter", True) is not None
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    # Stateless signed sessions ("expiry.signature"): they survive restarts, and changing the
    # password invalidates every one of them.
    key = hashlib.sha256(b"moss-studio-session:" + password.encode()).digest()

    def sign(expiry):
        return hmac.new(key, str(expiry).encode(), hashlib.sha256).hexdigest()
    failures = {"count": 0, "last": 0.0}
    login_lock = threading.Lock()
    page = (HERE / "index.html").read_text()
    login_page = (HERE / "login.html").read_text()
    public = {"/login", "/api/login"}

    def authed(request):
        expiry, _, sig = request.cookies.get(SESSION_COOKIE, "").partition(".")
        return expiry.isdigit() and int(expiry) > time.time() and hmac.compare_digest(sig, sign(int(expiry)))

    @app.middleware("http")
    async def guard(request: Request, call_next):
        path = request.url.path
        if path not in public:
            if not authed(request):
                if path.startswith(("/api/", "/files/", "/thumbs/")):
                    return JSONResponse({"detail": "Sign in required."}, status_code=401)
                return RedirectResponse("/login", status_code=303)
            # Custom header + SameSite=Strict cookie: cross-site pages cannot forge state changes.
            if request.method not in ("GET", "HEAD") and request.headers.get("x-studio") != "1":
                return JSONResponse({"detail": "Missing request header."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        )
        return response

    @app.get("/login")
    def login_form():
        return HTMLResponse(login_page)

    @app.post("/api/login")
    async def login(request: Request):
        if request.headers.get("x-studio") != "1":
            raise HTTPException(403, "Missing request header.")
        attempt = (await read_json(request)).get("password")
        with login_lock:
            wait = min(300, 30 * max(0, failures["count"] - 4)) - (time.time() - failures["last"])
            if wait > 0:
                raise HTTPException(429, f"Too many attempts. Try again in {int(wait) + 1}s.")
            if not isinstance(attempt, str) or not hmac.compare_digest(attempt.encode(), password.encode()):
                failures["count"] += 1
                failures["last"] = time.time()
                raise HTTPException(401, "Wrong password.")
            failures["count"] = 0
        expiry = int(time.time()) + SESSION_TTL
        token = f"{expiry}.{sign(expiry)}"
        response = JSONResponse({"ok": True})
        response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_TTL, httponly=True,
                            secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https",
                            samesite="strict", path="/")
        return response

    @app.post("/api/logout")
    def logout():
        response = JSONResponse({"ok": True})
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    @app.get("/")
    def index():
        return HTMLResponse(page, headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state():
        return {"jobs": queue.snapshot(), "fast": fast_available}

    @app.get("/api/gallery")
    def gallery_list():
        return {"items": gallery.items()}

    @app.post("/api/jobs")
    async def add_job(request: Request):
        body = await read_json(request)
        if not isinstance(body.get("params"), dict):
            raise HTTPException(400, "Invalid request.")
        if body["params"].get("kind"):
            params = clean_source_params(body["params"], gallery.root)
            if not params["prompt"].strip():  # video/extend default: the source item's own prompt
                src = next((i for i in gallery.items() if i["id"] == params["source"].split("/")[0]), None)
                params["prompt"] = (src or {}).get("params", {}).get("prompt") or ""
        else:
            params = clean_params(body["params"])
        if params["mode"] == "fast" and not fast_available:
            raise HTTPException(422, "Fast mode is not installed on this server.")
        repeat = body.get("repeat", 1)
        if not isinstance(repeat, int) or isinstance(repeat, bool) or not 1 <= repeat <= 20:
            raise HTTPException(422, "Repeat must be between 1 and 20.")
        try:
            queue.backend.config(params)  # same validation the model will apply, before queueing
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        queue.add(params, repeat)
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/move")
    async def move_job(job_id: int, request: Request):
        index = (await read_json(request)).get("index")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < 10000:
            raise HTTPException(422, "Index must be a whole number.")
        queue.move(job_id, index)
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/update")
    async def update_job(job_id: int, request: Request):
        body = await read_json(request)
        changes = {k: body[k] for k in ("prompt", "mode", "width", "height") if k in body}
        if changes.get("mode") == "fast" and not fast_available:
            raise HTTPException(422, "Fast mode is not installed on this server.")
        try:
            queue.update_params(job_id, changes)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: int):
        queue.cancel(job_id)
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/dismiss")
    def dismiss_job(job_id: int):
        queue.dismiss(job_id)
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: int):
        queue.retry(job_id)
        return {"ok": True}

    @app.post("/api/jobs/clear")
    def clear_jobs():
        queue.clear()
        return {"ok": True}

    @app.post("/api/gallery/{folder}/delete")
    def delete_item(folder: str):
        gallery.delete(folder)
        return {"ok": True}

    @app.post("/api/upload")
    async def upload(request: Request):
        """Add your own photo to the gallery (raw image bytes in the body), to edit or animate."""
        from io import BytesIO

        from PIL import Image, ImageOps, UnidentifiedImageError

        data = await request.body()
        if not data:
            raise HTTPException(422, "Choose an image.")
        if len(data) > UPLOAD_MAX:
            raise HTTPException(413, "That image is too large (max 30 MB).")
        try:
            with Image.open(BytesIO(data)) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise HTTPException(422, "That file isn't an image this server can read.")
        if max(im.size) > 4096:
            im.thumbnail((4096, 4096), Image.LANCZOS)
        folder = gallery.root / uuid.uuid4().hex
        folder.mkdir(parents=True)
        im.save(folder / "01.png")
        name = request.headers.get("x-filename", "")[:200]
        record = {"schema_version": 2, "kind": "upload", "images": ["01.png"], "elapsed_seconds": None,
                  "parameters": {"kind": "upload", "mode": "upload", "prompt": "", "file_name": name, "width": im.width, "height": im.height}}
        (folder / "parameters.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
        return {"ok": True, "id": folder.name}

    @app.post("/api/gallery/{folder}/love")
    async def love_item(folder: str, request: Request):
        loved = (await read_json(request)).get("loved")
        if not isinstance(loved, bool):
            raise HTTPException(422, "loved must be true or false.")
        gallery.love(folder, loved)
        return {"ok": True}

    @app.post("/api/gallery/{folder}/restore")
    def restore_item(folder: str):
        gallery.restore(folder)
        return {"ok": True}

    @app.get("/thumbs/{folder}/{name}")
    def thumbnail(folder: str, name: str):
        path, media = gallery.thumb(folder, name)
        return FileResponse(path, media_type=media, headers={"Cache-Control": "private, max-age=31536000, immutable"})

    @app.get("/files/{folder}/{name}")
    def image(folder: str, name: str):
        return FileResponse(gallery.file(folder, name), media_type="video/mp4" if name.endswith(".mp4") else "image/png",
                            headers={"Cache-Control": "private, max-age=31536000, immutable"})

    @app.get("/api/download-all")
    def download_all():
        # Built on the volume (not the small container disk); removed after sending or on error.
        tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False, dir=gallery.root)
        try:
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
                for item in gallery.items():
                    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(item["created"]))
                    for name in item["images"] + ([item["video"]] if item.get("video") else []):
                        path = gallery.root / item["id"] / name
                        if path.is_file():
                            zf.write(path, f"moss-{stamp}-{item['id'][:6]}-{name}")
            tmp.close()
        except Exception:
            tmp.close()
            os.unlink(tmp.name)
            raise
        return FileResponse(tmp.name, media_type="application/zip", filename="moss-images.zip",
                            background=BackgroundTask(os.unlink, tmp.name))

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Local MOSS-Image bundle directory")
    parser.add_argument("--moss-dir", default="/workspace/MOSS-Image", help="MOSS-Image checkout (for imports)")
    parser.add_argument("--output-dir", default="/workspace/outputs")
    parser.add_argument("--db", default="/workspace/studio.db")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--compile", action="store_true", help="torch.compile the image model (faster after warm-up)")
    parser.add_argument("--wan-dir", help="Wan2GP checkout (enables photo -> video)")
    parser.add_argument("--wan-python", default="python3", help="Python of the Wan2GP environment")
    parser.add_argument("--adapter", help="rapid-moss 4-step adapter directory (enables Fast mode)")
    parser.add_argument("--fake", action="store_true", help="No GPU: generate placeholder images (UI development)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    password = os.environ.get("MOSS_PASS", "")
    if len(password) < 8:
        sys.exit("MOSS_PASS must be set to a password of at least 8 characters.")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.fake:
        backend = FakeBackend(args.output_dir)
    else:
        if not args.model:
            sys.exit("--model is required unless --fake is set.")
        sys.path.insert(0, args.moss_dir)
        backend = MossBackend(args.model, args.output_dir, args.adapter, args.wan_dir, args.wan_python, args.compile)
    queue = Queue(args.db, backend)
    queue.output_dir = args.output_dir
    threading.Thread(target=queue.run_forever, daemon=True, name="worker").start()
    uvicorn.run(create_app(queue, Gallery(args.output_dir), password), host=args.host, port=args.port,
                proxy_headers=True, forwarded_allow_ips="*", log_level="warning")


if __name__ == "__main__":
    main()

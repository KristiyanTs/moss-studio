# Instructions for AI coding assistants

You are helping a user install and run MOSS Studio. It rents a GPU on RunPod, installs MOSS-Image and WanGP on it, and serves a web UI. Nothing runs on the user's own machine except `moss-pod.sh`, which only calls the RunPod API.

## Before you start

1. **Confirm a RunPod account with credit.** `up` starts billing at about $0.27/hour (RTX 3090 Ti, community cloud). Tell the user this before running it.
2. **Get the API key into `~/.runpod-key`** (`chmod 600`), or into `RUNPOD_API_KEY`. Ask the user to paste it into the file themselves; do not echo it into logs or commit it.
3. **Local requirements:** `bash`, `curl`, `python3`. macOS or Linux. No GPU, Docker or Python packages needed locally.
4. **Do not try to run MOSS-Image on the user's laptop.** The text encoder alone needs about 17 GB of memory, and the full download is 27 GB. A 16 GB Mac cannot run it.

## Install

```bash
./moss-pod.sh up       # prints link and password; billing starts
./moss-pod.sh status   # repeat every few minutes until "7860 ready"
```

- First boot takes 10–20 minutes, mostly the 27 GB model download. HTTP 502 from the link during this time is normal.
- Port 7861 (the raw WanGP UI) comes up later than 7860. The studio on 7860 is what the user needs.
- Give the user the link and password from the output. Do not sign in on their behalf.
- Remind the user that `stop` pauses GPU billing and `down` ends all billing. A forgotten pod costs about $195/month at $0.27/hour.

## Watching the boot

The REST API does not return container logs. Two places have them:
- **RunPod console** → the pod → **Logs** → **Container** tab (not System).
- **`/workspace/boot.log`** on the pod, via the console's web terminal.

Lines starting with `[wan]` come from the WanGP install, which runs in parallel with MOSS-Image.

Harmless lines, do not act on them:
- `pip's dependency resolver ... conflict` for an unrelated package
- `waiting to acquire lock` during the model download

## Known problems and their fixes

All of these are already fixed in `moss-pod.sh`. If you change the boot script, keep these fixes in place.

| Symptom | Cause | Fix in the script |
|---|---|---|
| Install stops in the first minute with `externally-managed-environment` | The PyTorch base image blocks system-wide `pip install` | `export PIP_BREAK_SYSTEM_PACKAGES=1` |
| Pod idle, link never comes up, no errors visible | Install output was silenced | Boot output goes to `/workspace/boot.log` and the Container log via `tee` |
| `up` fails, or the pod never gets a machine | The requested GPU is out of stock on community cloud | Retry with `GPUS="NVIDIA RTX A6000,NVIDIA A40" ./moss-pod.sh up`, or add `CLOUD=SECURE` |
| WanGP install aborts on a version it "cannot parse" | Base image has system packages with non-standard version strings | WanGP gets its own venv (`/workspace/wan-env`) with its own PyTorch (cu128) |
| WanGP install fails building `insightface` | No C compiler in the base image | `apt-get install build-essential python3-dev` |
| WanGP UI shows "Cross-origin request rejected" | RunPod's proxy changes the host the app sees | `--public-url https://${RUNPOD_POD_ID}-7861.proxy.runpod.net` |
| Out of GPU memory when making a video | MOSS and Wan share one 24 GB card | The studio queue runs one job at a time and releases Wan before the next image. Don't run the raw WanGP UI on 7861 while the studio is busy. |
| Video or edit job fails with "no space" | Wan 2.2 and the edit adapter download tens of GB on first use | Grow the volume in the RunPod console (pod → Edit Pod), which keeps the data; raise `volumeInGb` in `body()` for new pods. 120 GB is the default. |
| HTTP 502 right after `update` | The pod restarts to apply changes | Wait 2–5 minutes. Queued jobs that were running must be re-submitted. |

## Changing the code

- `studio/` is shipped to the pod as a base64 tarball in an environment variable at `up` and `update`. Run `./moss-pod.sh update` after any edit; it restarts the pod and keeps `/workspace` (models, images, database).
- **Test UI changes locally without a GPU:** `MOSS_PASS=localtest123 python3 studio/studio.py --fake --output-dir ./outputs --db ./studio.db` serves placeholder images at http://localhost:7860. First run `python3 -m venv .venv && .venv/bin/pip install fastapi uvicorn pillow` and use `.venv/bin/python`. The app refuses to start without a `MOSS_PASS` of at least 8 characters.
- Everything persistent lives on `/workspace`. The container disk is wiped on restart, so `apt-get` runs on every boot; that is expected.
- `.moss-pod` holds the pod ID and password. It is git-ignored; never commit it.

## Licensing

WanGP has its own licence, which forbids embedding it in a paid product or paid hosted service without the author's permission. If the user wants to sell access to this, point them to the Credits section of the README.

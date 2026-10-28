# MOSS Studio

**A fast, modern web UI for [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image): text-to-image, photo editing and image-to-video in one place. Runs on a rented GPU for about $0.27/hour, with one command.**

Type a prompt, press Enter, and get photorealistic images in a queue you can reorder, plus a gallery you can browse, love, filter and turn into videos. No ComfyUI graphs and no local GPU needed.

## Features

- **Text to image with MOSS-Image**, the open 5B diffusion model from OpenMOSS
  - **⚡ Fast mode**: 4 steps via the [rapid-moss adapter](https://huggingface.co/darask0/MOSS-Image_4-step_adapter_lora)
  - **✦ Best mode**: full 50-step quality, at HD or 2K, in 1:1, 4:3, 3:4, 16:9 and 9:16
  - Photo-realistic by default
- **🎬 Image to video**: one click turns any photo into a ~5-second 720p clip with Wan 2.2. Extend a video with **⏩ Extend**.
- **✏️ Edit a photo with words**: for example "make it night" or "red dress". Uses the MOSS-Image Edit HQ adapter.
- **⤒ Upload your own photos** to edit or animate them. You can also drag and drop.
- **A real queue**: new jobs go first; drag to reorder; switch a waiting job between Fast and Best; edit or stop jobs; see time-left estimates learned from your past runs.
- **A gallery that stays organised**: each photo's videos live inside its tile and play on hover. ♥ Love items, filter by Loved / Fast / Best / Edits / Videos / Uploads, sort, delete with undo, and use ✨ More like this.
- **Private by default**: password-protected, with signed session cookies and CSRF protection. Your images stay on your own GPU server.

## Install with an AI assistant

Open this folder in Claude Code, Cursor, Codex or a similar coding assistant and say:

> Set up MOSS Studio for me by following AGENTS.md.

[AGENTS.md](AGENTS.md) walks the assistant through the install and lists every deploy problem we hit, with its fix, so it doesn't have to rediscover them.

## Quick start

You need a [RunPod](https://www.runpod.io) account with a few dollars of credit, plus an API key. The script runs on macOS or Linux and only needs `bash`, `curl` and `python3`.

```bash
git clone https://github.com/KristiyanTs/moss-studio.git
cd moss-studio
echo "YOUR_RUNPOD_API_KEY" > ~/.runpod-key && chmod 600 ~/.runpod-key
./moss-pod.sh up        # rents the GPU and installs everything; prints your link and password
./moss-pod.sh status    # wait until port 7860 says "ready"
```

1. Open the printed `https://<pod>-7860.proxy.runpod.net` link and sign in with the printed password.
2. The first boot downloads about 27 GB of model weights and takes 10–20 minutes. Videos and edits download their own models the first time you use them.

| Command | What it does |
|---|---|
| `./moss-pod.sh up` | Create the server; billing starts |
| `./moss-pod.sh status` | Show whether it is ready, plus the link and password |
| `./moss-pod.sh update` | Push local changes to the running server; it restarts and keeps your images |
| `./moss-pod.sh stop` / `start` | Pause or resume GPU billing; storage is still billed |
| `./moss-pod.sh down` | Delete the server and its storage; all billing ends |

Your pod ID and password are kept in `.moss-pod` next to the script. That file is git-ignored; keep it private.

## Troubleshooting

- **Logs:** RunPod console → your pod → **Logs** → **Container** (not System). The same output is saved in `/workspace/boot.log` on the pod.
- **HTTP 502 on the link:** normal during the first 10–20 minutes, and for a few minutes after `update`.
- **Pod never starts:** the GPU is out of stock. Try `GPUS="NVIDIA RTX A6000,NVIDIA A40" ./moss-pod.sh up`, or add `CLOUD=SECURE`.
- **Video or edit fails with "no space":** raise the volume size in the RunPod console (pod → Edit Pod). Your images stay. Raise `volumeInGb` in `moss-pod.sh` too, so new pods start bigger.
- **Can I run it on my Mac?** No. The model needs about 17 GB of memory at once plus a 27 GB download. MOSS-Image's Mac support is experimental and was only tested on 32 GB machines.

More fixes are listed in [AGENTS.md](AGENTS.md#known-problems-and-their-fixes).

## Cost and speed

- **Default GPU**: RTX 3090 Ti (24 GB), community cloud, **$0.27/hour** (RunPod's listed price on 2026-10-28). Choose another GPU with `GPUS="NVIDIA RTX A6000" CLOUD=SECURE ./moss-pod.sh up`.
- **Best image**: about **1.5 minutes** for a 1024×1024 image on that GPU (measured at 1.66 s per step × 50 steps).
- **Fast image**: Fast mode does 4 steps instead of 50. The adapter's author measured 0.81 s vs 16.2 s of model time on an H100.
- **Timings**: every image stores a per-stage timing breakdown in its `parameters.json`.

## How it works

`moss-pod.sh` creates a RunPod pod and installs the following on its persistent `/workspace` volume:
- **MOSS-Image** and its SFT weights
- **The Fast-mode adapter**
- **[WanGP](https://github.com/deepbeepmeep/Wan2GP)**, which runs video and editing models in its own environment

`studio/studio.py` is a single FastAPI app that owns the GPU:
- **One worker** runs the queue in order.
- **Video and edit jobs** go to a long-lived WanGP worker, which is released before the next MOSS image so the two never compete for VRAM.

## Credits and licences

- **MOSS Studio** (this repo): MIT, see [LICENSE](LICENSE).
- **[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)** by OpenMOSS: Apache-2.0.
- **[rapid-moss 4-step adapter](https://huggingface.co/darask0/MOSS-Image_4-step_adapter_lora)** by darask0: Apache-2.0.
- **Videos and edits are powered by [WanGP](https://github.com/deepbeepmeep/Wan2GP)** by deepbeepmeep.
  - WanGP is installed from its own repository and is **not** covered by this repo's MIT licence. It has its own [WanGP Community License](https://github.com/deepbeepmeep/Wan2GP/blob/main/LICENSE.txt).
  - That licence lets you use it free and sell your outputs.
  - It does **not** allow embedding WanGP in a paid product or a paid hosted service without a commercial licence from its author.
- **Models**: Wan 2.2 and the MOSS-Image Edit adapter are downloaded by WanGP under their own licences.

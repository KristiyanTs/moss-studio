#!/usr/bin/env bash
# Rent a RunPod GPU, install MOSS-Image with the MOSS Studio UI (images, port 7860) and Wan2GP (videos, port 7861).
#   ./moss-pod.sh up      create the pod (billing starts)
#   ./moss-pod.sh update  apply edits to this script to the running pod (restarts it, keeps /workspace)
#   ./moss-pod.sh status  show state and whether the UIs are ready
#   ./moss-pod.sh stop    pause GPU billing (model stays on disk, ~$0.20/GB/mo)
#   ./moss-pod.sh start   resume a stopped pod
#   ./moss-pod.sh down    delete the pod and its disk (all billing ends)
# Override hardware with GPUS="NVIDIA A40,NVIDIA RTX A6000" CLOUD=SECURE.
# Needs RUNPOD_API_KEY in the environment or in ~/.runpod-key.
set -euo pipefail

API=https://rest.runpod.io/v1
STATE="$(dirname "$0")/.moss-pod"
KEY="${RUNPOD_API_KEY:-$(cat ~/.runpod-key 2>/dev/null || true)}"
[ -n "$KEY" ] || { echo "Set RUNPOD_API_KEY or put the key in ~/.runpod-key"; exit 1; }

api() { curl -sS --fail-with-body -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" "$@"; }
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }
load() { [ -f "$STATE" ] || { echo "No pod recorded. Run: $0 up"; exit 1; }; source "$STATE"; }

# Runs inside the pod. Everything lands on /workspace so a stop/start skips the 27 GB download.
BOOT='
set -e
export PIP_BREAK_SYSTEM_PACKAGES=1
apt-get update -qq && apt-get install -y -qq git python3-venv python3-dev build-essential ffmpeg libgl1 libglib2.0-0 >/dev/null
cd /workspace
# Wan2GP pins older transformers/gradio than MOSS-Image, so it gets its own venv.
(
  set -e
  cd /workspace
  [ -d Wan2GP ] || git clone --depth 1 https://github.com/deepbeepmeep/Wan2GP.git
  # Isolated venv: the base image has system packages with versions pip cannot parse.
  [ -d wan-env ] || python3 -m venv wan-env
  cd Wan2GP
  if [ ! -f ../wan-env/.installed ]; then
    ../wan-env/bin/pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
    ../wan-env/bin/pip install -r requirements.txt
    touch ../wan-env/.installed
  fi
  WANGP_AUTH_PASSWORD="$MOSS_PASS" ../wan-env/bin/python wgp.py --listen --server-port 7861 --auth --public-url "https://${RUNPOD_POD_ID}-7861.proxy.runpod.net"
) 2>&1 | sed -u "s/^/[wan] /" &
[ -d MOSS-Image ] || git clone --depth 1 https://github.com/OpenMOSS/MOSS-Image.git
cd MOSS-Image
pip install -e ".[ui]"
hf download OpenMOSS-Team/MOSS-Image-SFT --local-dir /workspace/models/MOSS-Image-SFT
hf download darask0/MOSS-Image_4-step_adapter_lora lora.safetensors time_embedding.safetensors dataspace.safetensors --local-dir /workspace/models/rapid-moss || echo "Fast mode adapter download failed"
# MOSS Studio (queue + gallery UI, shipped from ./studio via $STUDIO_TGZ) owns the model on port 7860.
mkdir -p /workspace/studio && echo "$STUDIO_TGZ" | base64 -d | tar xz -C /workspace/studio
python3 /workspace/studio/studio.py --model /workspace/models/MOSS-Image-SFT --moss-dir /workspace/MOSS-Image \
  --adapter /workspace/models/rapid-moss --compile \
  --wan-dir /workspace/Wan2GP --wan-python /workspace/wan-env/bin/python \
  --output-dir /workspace/outputs --db /workspace/studio.db --port 7860
'

# Pod settings. $1=create adds the fields only accepted at creation.
body() { STUDIO_TGZ=$(cd "$(dirname "$0")/studio" && COPYFILE_DISABLE=1 tar czf - studio.py wan_worker.py index.html login.html | base64 | tr -d '\n') \
  BOOT="$BOOT" PASS="$PASS" MODE="$1" python3 -c '
import json, os
b = {
  "imageName": "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime",
  "containerDiskInGb": 30,
  "volumeInGb": 120,  # MOSS ~27 GB + Wan 2.2 models and Wan2GP venv
  "volumeMountPath": "/workspace",
  "ports": ["7860/http", "7861/http"],
  "env": {"MOSS_PASS": os.environ["PASS"], "STUDIO_TGZ": os.environ["STUDIO_TGZ"]},
  "dockerStartCmd": ["bash", "-c",
    "(" + os.environ["BOOT"] + ") 2>&1 | tee -a /workspace/boot.log; sleep infinity"],
}
if os.environ["MODE"] == "create":
  b.update(name="moss-image", gpuTypePriority="custom",
           gpuTypeIds=os.environ.get("GPUS", "NVIDIA GeForce RTX 3090 Ti").split(","),
           cloudType=os.environ.get("CLOUD", "COMMUNITY"))
print(json.dumps(b))'; }

case "${1:-}" in
  up)
    [ ! -f "$STATE" ] || { echo "Pod already exists ($STATE). Run '$0 down' first."; exit 1; }
    PASS=$(python3 -c "import secrets; print(secrets.token_urlsafe(12))")
    ID=$(api -X POST "$API/pods" -d "$(body create)" | json 'd["id"]')
    printf 'ID=%s\nPASS=%s\n' "$ID" "$PASS" > "$STATE"
    echo "Pod $ID created. Billing has started."
    echo "First boot downloads ~27 GB; expect 10-20 min. Check with: $0 status"
    echo "Images:   https://$ID-7860.proxy.runpod.net"
    echo "Videos:   https://$ID-7861.proxy.runpod.net"
    echo "Password: $PASS"
    ;;
  update)
    load; api -X PATCH "$API/pods/$ID" -d "$(body update)" >/dev/null
    echo "Pod $ID updated and restarting. Run '$0 status' in a few minutes." ;;
  status)
    load
    api "$API/pods/$ID" | json '"state: %s  gpu: %s  $/hr: %s" % (d.get("desiredStatus"), (d.get("machine") or {}).get("gpuTypeId"), d.get("costPerHr"))'
    for port in 7860 7861; do
      url="https://$ID-$port.proxy.runpod.net"
      code=$(curl -s -o /dev/null -w '%{http_code}' "$url/" || true)
      if [ "$code" = 200 ]; then echo "$port ready: $url"; else echo "$port not up yet (HTTP $code)"; fi
    done
    echo "Password: $PASS"
    ;;
  stop)  load; api -X POST "$API/pods/$ID/stop" >/dev/null; echo "Stopped. GPU billing paused; disk still billed." ;;
  start) load; api -X POST "$API/pods/$ID/start" >/dev/null; echo "Starting. Run '$0 status' in a few minutes." ;;
  down)  load; api -X DELETE "$API/pods/$ID" >/dev/null; rm -f "$STATE"; echo "Deleted. Billing has ended." ;;
  *) sed -n 2,8p "$0"; exit 1 ;;
esac

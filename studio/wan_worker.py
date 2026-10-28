"""Long-lived Wan2GP worker: keeps the last video/edit model loaded between jobs.

Runs inside the Wan2GP virtualenv. Protocol: one JSON request per stdin line ({"settings": {...}}),
JSON events per stdout line: {"ready"}, {"progress": [step, total], "phase": str}, {"done": bool, "files": [...], "errors": [...]}.
Everything else Wan2GP prints goes to stderr.
"""
import json
import os
import sys
from pathlib import Path

wan_dir, output_dir = sys.argv[1], sys.argv[2]
out = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)  # keep stdout clean for the protocol


def send(**msg):
    out.write(json.dumps(msg) + "\n")


os.chdir(wan_dir)
sys.path.insert(0, wan_dir)
from shared.api import init  # noqa: E402

session = init(root=Path(wan_dir), output_dir=Path(output_dir), console_output=True, console_isatty=False)
send(ready=True)
for line in sys.stdin:
    try:
        job = session.submit_task(json.loads(line)["settings"])
        for event in job.events.iter(timeout=0.5):
            if event.kind == "progress":
                d = event.data
                send(progress=[getattr(d, "current_step", 0) or 0, getattr(d, "total_steps", 0) or 0], phase=str(getattr(d, "phase", "") or ""))
        result = job.result()
        send(done=bool(result.success), files=list(result.generated_files or []), errors=[str(e.message) for e in (result.errors or [])])
    except Exception as exc:  # report and keep serving
        send(done=False, files=[], errors=[f"{type(exc).__name__}: {exc}"])

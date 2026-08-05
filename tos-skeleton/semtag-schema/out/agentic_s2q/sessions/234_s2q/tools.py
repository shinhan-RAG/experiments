
import json, os, subprocess, sys
_here = os.path.dirname(os.path.abspath(__file__))
_cfg = json.load(open(os.path.join(_here, "cfg.json")))
VIEW = _cfg["view"]; SDIR = _here
CAP = 10
calls = os.path.join(SDIR, "calls.txt")
n = sum(1 for _ in open(calls)) if os.path.exists(calls) else 0
if n >= CAP:
    print(json.dumps({"error": "call cap reached — 지금까지 정보로 답하라"})); sys.exit(0)
open(calls, "a").write("\t".join(sys.argv[1:3]) + "\n")
cmd = sys.argv[1]
if cmd == "grep":
    pat = sys.argv[2]
    r = subprocess.run(["rg", "-i", "-e", pat, "--", VIEW],
                       capture_output=True, text=True, timeout=30)
    lines = r.stdout.splitlines()
    if len(lines) > 20:
        step = len(lines) / 20
        lines = [lines[int(i * step)] for i in range(20)]
        note = f"(전체 {len(r.stdout.splitlines())}건 중 등간격 20건)"
    else:
        note = f"({len(lines)}건)"
    print(note)
    for l in lines:
        print(l[:300])
elif cmd == "read":
    eid = sys.argv[2]
    for l in open(VIEW):
        if l.startswith(eid + " | "):
            print(l[:4000]); break
    else:
        print(json.dumps({"error": "eid not found"}))

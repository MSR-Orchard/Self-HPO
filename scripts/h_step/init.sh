#!/bin/bash
# init.sh <work_dir> <parent_harness_dir> <search_set_dir>
# Create an H-step work dir.
#   parent_harness_dir : the harness this H-step starts from (harnesses/seed for the first H-step,
#                        the previous H-step's output for later ones). Needs agent.py + config.yaml.
#   search_set_dir     : output of build_search_set.sh (search_set.json, gold_patches.jsonl, tasks_*.jsonl)
set -euo pipefail
source "$(dirname "$0")/../env.sh"
W=${1:?usage: init.sh <work_dir> <parent_harness_dir> <search_set_dir>}
PARENT=${2:?parent harness dir}; SS=${3:?search set dir}
[ -f "$PARENT/agent.py" ] && [ -f "$PARENT/config.yaml" ] || { echo "parent harness needs agent.py + config.yaml"; exit 1; }
[ -f "$SS/search_set.json" ] || { echo "no search_set.json in $SS (run build_search_set.sh)"; exit 1; }
[ -e "$W" ] && { echo "$W exists — move it away first"; exit 1; }
H=$SELFHPO_REPO/scripts/h_step
mkdir -p "$W/agents" "$W/experience" "$W/runs" "$W/data"
W=$(cd "$W" && pwd)
cp "$H"/search_eval.py "$H"/fire_gate.py "$H"/parent_check.py "$H"/record_result.py "$H"/inspect_trajs.py \
   "$H"/build_experience.py "$H"/render_prompt.py "$H"/propose_prompt.tmpl "$H"/seed_bar.sh "$W/"
cp "$SS"/search_set.json "$W/"
cp "$SS"/gold_patches.jsonl "$W/data/"
cp "$SS"/tasks_*.jsonl "$W/data/" 2>/dev/null || true
python3 - "$W" <<'PY'
import json, os, sys
w = sys.argv[1]; p = f"{w}/search_set.json"; s = json.load(open(p))
for g in s["groups"]:
    if g["subset"].endswith(".jsonl"):
        g["subset"] = f"{w}/data/{os.path.basename(g['subset'])}"
json.dump(s, open(p, "w"), indent=1)
PY
cp -r "$PARENT" "$W/agents/parent"
rm -f "$W"/agents/parent/{.hook_fired,MARKER.txt,ACT_MARKER.txt,parent,purpose}
echo parent > "$W/experience/current_parent.txt"
echo "initialized $W (parent=$PARENT, $(python3 -c "import json;print(json.load(open('$W/search_set.json'))['n_tasks'])") search tasks)"

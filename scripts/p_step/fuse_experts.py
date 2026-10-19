# fuse_experts.py <exported HF dir> <out dir>   (env BASE_HF: base model dir)
import json, os, re, gc, shutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import sys
SRC, OUT = sys.argv[1], sys.argv[2]
BASE = os.environ.get("BASE_HF") or os.path.join(os.environ.get("SELFHPO_WORK", "work"), "models", "Qwen3.5-35B-A3B")
os.makedirs(OUT, exist_ok=True)

si = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
pat = re.compile(r"^(model\.language_model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
groups = {}; split_keys = set()
for k in si:
    m = pat.match(k)
    if m:
        groups.setdefault(m.group(1), {}).setdefault(int(m.group(2)), {})[m.group(3)] = k
        split_keys.add(k)
def is_stale_fused(k):
    return k.endswith(".mlp.experts.gate_up_proj") or k.endswith(".mlp.experts.down_proj")
print(f"fusing {len(groups)} expert groups", flush=True)

def load(k):
    with safe_open(os.path.join(SRC, si[k]), "pt") as f: return f.get_tensor(k)

SHARD = 4 * 1024**3
weight_map = {}; shard_i = 0; cur = {}; cur_b = 0; total_b = 0
def flush():
    global cur, cur_b, shard_i
    if not cur: return
    fn = f"model-{shard_i:05d}.safetensors"
    save_file(cur, os.path.join(OUT, fn), metadata={"format": "pt"})
    for n in cur: weight_map[n] = fn
    print(f"  wrote {fn}: {len(cur)} tensors {cur_b/1e9:.1f}GB", flush=True)
    cur = {}; cur_b = 0; shard_i += 1
def add(name, t):
    global cur_b, total_b
    t = t.float().contiguous() if (name.endswith("linear_attn.A_log") or name.endswith("linear_attn.norm.weight")) else t.contiguous(); cur[name] = t
    b = t.numel()*t.element_size(); cur_b += b; total_b += b
    if cur_b >= SHARD: flush()

NE = 256
for prefix in sorted(groups):
    g = groups[prefix]; assert len(g) == NE, f"{prefix}:{len(g)}"
    gu, dn = [], []
    for e in range(NE):
        p = g[e]
        gu.append(torch.cat([load(p["gate_proj"]), load(p["up_proj"])], dim=0))
        dn.append(load(p["down_proj"]))
    GU = torch.stack(gu, dim=0); DN = torch.stack(dn, dim=0)
    assert list(GU.shape) == [NE, 1024, 2048] and list(DN.shape) == [NE, 2048, 512], (GU.shape, DN.shape)
    add(f"{prefix}.gate_up_proj", GU); add(f"{prefix}.down_proj", DN)
    del gu, dn, GU, DN; gc.collect()
ncopy = 0
for k in si:
    if k in split_keys or is_stale_fused(k): continue
    add(k, load(k)); ncopy += 1
flush()
print(f"copied {ncopy} non-expert tensors", flush=True)
json.dump({"metadata": {"total_size": total_b}, "weight_map": weight_map},
          open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=2)
for f in os.listdir(BASE):
    if (f.endswith((".json", ".txt")) or "token" in f.lower()) and f != "model.safetensors.index.json":
        shutil.copy(os.path.join(BASE, f), os.path.join(OUT, f))
print(f"FUSE_DONE total={total_b/1e9:.1f}GB shards={shard_i} tensors={len(weight_map)}", flush=True)

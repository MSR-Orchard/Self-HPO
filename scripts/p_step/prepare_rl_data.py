"""Prepare the P-step data under $SELFHPO_WORK/rl/data.

  train.jsonl                  2,484 RL training tasks from the HF dataset self-hpo/Self-HPO-RL-Data
  swebench_verified_500.jsonl  SWE-bench Verified test split in the same format (in-training eval)
"""
import json, os, shutil
from datasets import load_dataset
from huggingface_hub import hf_hub_download

out = os.path.join(os.environ["SELFHPO_WORK"], "rl", "data")
os.makedirs(out, exist_ok=True)

src = hf_hub_download("self-hpo/Self-HPO-RL-Data", "train.jsonl", repo_type="dataset")
shutil.copyfile(src, f"{out}/train.jsonl")

keys = ["instance_id", "repo", "base_commit", "version", "test_patch", "hints_text", "created_at",
        "environment_setup_commit", "FAIL_TO_PASS", "PASS_TO_PASS"]
n = 0
with open(f"{out}/swebench_verified_500.jsonl", "w") as f:
    for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test"):
        meta = {"dataset": "princeton-nlp/SWE-bench_Verified", **{k: r[k] for k in keys}}
        f.write(json.dumps({"problem_statement": r["problem_statement"], "patch": r["patch"],
                            "metadata": meta}) + "\n")
        n += 1
print(f"wrote {out}/train.jsonl and swebench_verified_500.jsonl ({n} tasks)")

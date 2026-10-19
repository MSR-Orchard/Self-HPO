#!/usr/bin/env python3
"""Tally gen_traj.sh / score_traj.sh runs under $SELFHPO_WORK/eval/<Method>[/<Step>]/<Benchmark>/.
Usage: python scripts/eval/results.py -m <method> [-b sbv|multi|sbp] [-s STEP] [--nocache]
"""
import argparse,os,json,glob,statistics,subprocess,collections,re
ROOT=os.path.join(os.environ.get("SELFHPO_WORK") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"work"),"eval")
def resolve_method(x):
    if not x or not os.path.isdir(ROOT): return None
    for d in os.listdir(ROOT):
        if d.lower()==x.lower() and os.path.isdir(os.path.join(ROOT,d)): return d
    return None
BENCH={"sbv":"SWE_Bench_Verified","swe-v":"SWE_Bench_Verified","verified":"SWE_Bench_Verified","swe-multi":"SWE_Bench_Multilingual","multi":"SWE_Bench_Multilingual","ml":"SWE_Bench_Multilingual","sbm":"SWE_Bench_Multilingual","swe-multilingual":"SWE_Bench_Multilingual","swe-pro":"SWE_Bench_Pro","pro":"SWE_Bench_Pro","sbp":"SWE_Bench_Pro"}
DENOM={"SWE_Bench_Verified":500,"SWE_Bench_Multilingual":300,"SWE_Bench_Pro":642}
TIMEOUT_LIM={"SWE_Bench_Verified":900,"SWE_Bench_Multilingual":900,"SWE_Bench_Pro":1800}
def read_steplimit(real):
    try:
        f=glob.glob(real+"/*/*.traj.json")[0]
        return json.load(open(f)).get("info",{}).get("config",{}).get("agent",{}).get("step_limit",250)
    except: return 250
def step_match(step_dir, spec):
    return spec.lower().strip()==step_dir.lower()
def resolve_bench(x):
    x=x.lower().replace("_","-")
    if x in BENCH: return BENCH[x]
    cands={v for k,v in BENCH.items() if k.startswith(x) or x.startswith(k)}
    if len(cands)==1: return cands.pop()
    if "mult" in x or x=="ml": return "SWE_Bench_Multilingual"
    if "pro" in x or "sbp" in x: return "SWE_Bench_Pro"
    if "ver" in x or x in ("sbv","v"): return "SWE_Bench_Verified"
    return None
def find_verify(rundir):
    real=os.path.realpath(rundir); inside=os.path.join(real,"verify_results.json")
    if os.path.isfile(inside): return inside,real
    return (None,real)
def gen_stats(real,nocache=False):
    trajs=glob.glob(real+"/*/*.traj.json"); nt=len(trajs); cache=os.path.join(real,".gen_stats.json")
    mt=0.0
    for f in trajs:
        try: mt=max(mt,os.path.getmtime(f))
        except OSError: pass
    if not nocache and os.path.isfile(cache):
        try:
            s=json.load(open(cache))
            if s.get("_nt")==nt and s.get("_mtime")==mt: return s
        except: pass
    c=collections.Counter()
    try:
        out=subprocess.run("grep -m1 -oh '\"exit_status\": *\"[^\"]*\"' "+real+"/*/*.traj.json 2>/dev/null",shell=True,capture_output=True,text=True,timeout=300).stdout
        for line in out.splitlines():
            p=line.split('"')
            if len(p)>=4: c[p[3]]+=1
    except Exception: pass
    wi=wr=0
    if c.get("WallClockTimeout",0) or c.get("TimeoutError",0):
        try:
            wf=subprocess.run("grep -lm1 -E '\"exit_status\": *\"(WallClockTimeout|TimeoutError)\"' "+real+"/*/*.traj.json 2>/dev/null",shell=True,capture_output=True,text=True,timeout=180).stdout.split()
            for f in wf:
                try:
                    t=json.load(open(f)); msgs=t.get("messages") or []
                    asst=sum(1 for m in msgs if isinstance(m,dict) and m.get("role")=="assistant")
                    if asst==0: wi+=1
                    else: wr+=1
                except: wi+=1
        except Exception: pass
    s=dict(c); s["_wct_infra"]=wi; s["_wct_real"]=wr; s["_nt"]=nt; s["_mtime"]=mt
    try: json.dump(s,open(cache,"w"))
    except: pass
    return s
def score_stats(vf):
    v=json.load(open(vf)); res=0; ge=0
    for x in v.values():
        if not isinstance(x,dict): continue
        if x.get("resolved") or any(sm.get("resolved") for sm in x.get("samples",[])): res+=1
        if x.get("error") or any(sm.get("error") or sm.get("exit_code",0) in (-1,None) for sm in x.get("samples",[])): ge+=1
    den=DENOM.get(os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(vf)))),len(v))
    return res,max(den,len(v)),ge
def _clean_runs(leafpath):
    return sorted(d for d in os.listdir(leafpath) if not d.startswith(".") and not d.startswith("_") and (os.path.isdir(os.path.join(leafpath,d)) or os.path.islink(os.path.join(leafpath,d))))
def show_leaf(leafpath,label,nocache):
    if not os.path.isdir(leafpath): return False
    runs=_clean_runs(leafpath)
    if not runs: return False
    tlim=TIMEOUT_LIM.get(os.path.basename(leafpath),900); slim=None
    rows=[]; rates=[]
    for d in runs:
        vf,real=find_verify(os.path.join(leafpath,d)); g=gen_stats(real,nocache)
        if slim is None: slim=read_steplimit(real)
        sub=g.get("Submitted",0); ise=g.get("InternalServerError",0); le=g.get("LimitsExceeded",0)
        wi=g.get("_wct_infra",0); wr=g.get("_wct_real",0)
        if vf is None:
            rows.append(dict(d=d,pend=True,le=le,wr=wr,wi=wi,ise=ise)); continue
        res,n,ge=score_stats(vf); rate=100.0*res/n; failed=max(0,sub-res)
        rows.append(dict(d=d,pend=False,res=res,n=n,rate=rate,failed=failed,le=le,wr=wr,wi=wi,ise=ise,ge=ge))
        rates.append(rate)
    print("=== %s ==="%label)
    hfmt="  %-46s %9s %6s %7s %13s %12s %9s %6s %9s %6s"
    print(hfmt%("run","resolved","rate","failed","LimitExc(%s)"%slim,"timeout(%ss)"%tlim,"no-start","srvErr","gradeErr","Total"))
    for r in rows:
        d=r["d"][:46]
        if r["pend"]:
            print(hfmt%(d,"pending","-","-","-","-",str(r["wi"]),str(r["ise"]),"-","-")); continue
        print(("  %-46s %9s %5.1f%% %7d %13d %12d %9d %6d %9d %6d")%(d,"%d/%d"%(r["res"],r["n"]),r["rate"],r["failed"],r["le"],r["wr"],r["wi"],r["ise"],r["ge"],r["res"]+r["failed"]+r["le"]+r["wr"]+r["wi"]+r["ise"]))
    if rates:
        m=statistics.mean(rates); sd=statistics.pstdev(rates) if len(rates)>1 else 0.0
        print("  --> avg@%d = %.1f ± %.1f\n"%(len(rates),m,sd))
    else: print("  (no scored runs)\n")
    return True
BENCH_ORDER=["SWE_Bench_Verified","SWE_Bench_Multilingual","SWE_Bench_Pro"]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("-m",required=False,default=None,dest="method"); ap.add_argument("-b",dest="benchmark",default=None); ap.add_argument("-s",dest="step",default=None); ap.add_argument("--nocache",action="store_true")
    a=ap.parse_args()
    if not a.method: raise SystemExit("-m <method dir under %s> required"%ROOT)
    method=resolve_method(a.method)
    if not method: raise SystemExit("no method dir %s under %s"%(a.method,ROOT))
    if a.benchmark:
        bench=resolve_bench(a.benchmark)
        if not bench: raise SystemExit("benchmark: sbv multi sbp")
        benches=[bench]
    else:
        benches=BENCH_ORDER
    mroot=os.path.join(ROOT,method)
    if not os.path.isdir(mroot): raise SystemExit("no method dir "+mroot)
    subs=[x for x in os.listdir(mroot) if os.path.isdir(os.path.join(mroot,x))]
    is_direct=any(x.startswith("SWE_Bench_") for x in subs)
    ok=False
    if is_direct:
        for bench in benches:
            ok=show_leaf(os.path.join(mroot,bench),"%s / %s"%(method,bench),a.nocache) or ok
    else:
        steps=[x for x in subs if not x.startswith("SWE_Bench_")]
        if a.step:
            specs=[x for x in re.split(r"[/,]",a.step) if x.strip()]
            steps=[x for x in steps if any(step_match(x,sp) for sp in specs)]
        steps=sorted(steps)
        for st in steps:
            for bench in benches:
                ok=show_leaf(os.path.join(mroot,st,bench),"%s / %s / %s"%(method,st,bench),a.nocache) or ok
    if not ok: print("(no populated leaves for %s)"%method)
if __name__=="__main__": main()

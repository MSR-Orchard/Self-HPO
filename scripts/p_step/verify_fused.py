# verify_fused.py <fused HF> <exported HF dir>: check a fuse_experts.py output against the export.
# env BASE_HF: base model dir (default $SELFHPO_WORK/models/Qwen3.5-35B-A3B)
import json,os,sys,torch
from safetensors import safe_open
F,D=sys.argv[1],sys.argv[2]; V=os.environ.get('BASE_HF', os.path.join(os.environ.get('SELFHPO_WORK','work'),'models','Qwen3.5-35B-A3B'))
def wm(H): return json.load(open(H+'/model.safetensors.index.json'))['weight_map']
def get(H,k,m):
    with safe_open(H+'/'+m[k],'pt') as s: return s.get_tensor(k)
mf,mv,md=wm(F),wm(V),wm(D)
ok=True
if set(mf)!=set(mv): ok=False; print('key set differs from base:', len(set(mf)-set(mv)), len(set(mv)-set(mf)), list(set(mf)^set(mv))[:5])
bad=0
for k in mf:
    with safe_open(F+'/'+mf[k],'pt') as s: sl=s.get_slice(k); sh,dt=sl.get_shape(),sl.get_dtype()
    with safe_open(V+'/'+mv[k],'pt') as s: sv=s.get_slice(k); shv,dtv=sv.get_shape(),sv.get_dtype()
    if sh!=shv or dt!=dtv: bad+=1
print('shape/dtype differences vs base:',bad); ok&=bad==0
for L in (0,13,39):
    k=f'model.language_model.layers.{L}.mlp.experts.down_proj'
    f=get(F,k,mf); v=get(V,k,mv); e5=get(D,f'model.language_model.layers.{L}.mlp.experts.5.down_proj.weight',md)
    eq=torch.equal(f[5],e5); dv=not torch.equal(f,v)
    gu=get(F,f'model.language_model.layers.{L}.mlp.experts.gate_up_proj',mf)
    g5=torch.cat([get(D,f'model.language_model.layers.{L}.mlp.experts.5.gate_proj.weight',md),get(D,f'model.language_model.layers.{L}.mlp.experts.5.up_proj.weight',md)],0)
    print('layer',L,'down_proj matches export:',eq,'gate_up_proj matches export:',torch.equal(gu[5],g5),'differs from base:',dv); ok&=eq and dv and torch.equal(gu[5],g5)
k='model.language_model.layers.3.self_attn.o_proj.weight' if 'model.language_model.layers.3.self_attn.o_proj.weight' in mf else [x for x in mf if 'shared_expert.down_proj' in x][0]
ne=torch.equal(get(F,k,mf),get(D,k,md))
print('non-expert',k,'matches export:',ne); ok&=ne
print('size GB:',round(sum(os.path.getsize(F+'/'+f) for f in set(mf.values()))/1e9,1))
print('VERIFY_OK' if ok else 'VERIFY_FAIL'); sys.exit(0 if ok else 1)

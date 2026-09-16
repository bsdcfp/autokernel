"""Follow-up root-cause probes. No candidate or baseline is overwritten."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from preflight import ROOT, resolve_candidate
from wanbench.core import digest, dump_new, load_json, validate_device_selection
from wanbench.tasks import build_reference, make_inputs, rmsnorm, rope3d, flatten_outputs


def nearest_bf16(value):
    """Round finite in-range binary64 directly, without a float32 midpoint hop."""
    if not math.isfinite(value) or abs(value) > float.fromhex('0x1.fep127'):
        raise ValueError('diagnostic oracle expects finite BF16-range values')
    if value == 0:
        return value
    quantum = math.ldexp(1.0, max(math.frexp(abs(value))[1] - 8, -133))
    return round(value / quantum) * quantum


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--output', required=True, type=Path)
    args=ap.parse_args()
    validate_device_selection(load_json(ROOT/'configs/device.json'),os.environ.get('CUDA_VISIBLE_DEVICES'),0)
    if args.output.exists(): raise FileExistsError(args.output)
    import torch
    from wanbench.gpu import check_output
    torch.cuda.set_device(0)
    if 'B300' not in torch.cuda.get_device_name(0): raise RuntimeError('requires B300')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    base=build_reference(torch,args.task)
    report={'kind':'precision-followup','task':args.task,'case':args.case,'results':[],
            'diagnostic_sha256':digest(Path(__file__)), 'counterexamples':[]}
    def record(name,fn,values,target,seed):
        r=check_output(torch,fn,values,target,.01,.01)
        actual=fn(*values)
        aa,bb=flatten_outputs(torch,actual),flatten_outputs(torch,target)
        r.update(label=name,seed=seed,unequal_elements=sum(int((a!=b).sum().item()) for a,b in zip(aa,bb)))
        report['results'].append(r)
        print(name,seed,'PASS' if r['passed'] else 'FAIL','bad=',r.get('failed_elements'),'unequal=',r['unequal_elements'],flush=True)
        return actual
    compiled=torch.compile(base)
    if args.task=='wan-linear-gelu':
        no_patterns=torch.compile(base,options={'pattern_matcher':False})
        # Preserve ATen addmm by making its BF16 result observable. This is a
        # diagnostic extra output, not a silently changed performance baseline.
        def observed(x,w,b):
            z=torch.nn.functional.linear(x,w,b)
            return torch.nn.functional.gelu(z,approximate='tanh'),z
        observed_compile=torch.compile(observed)
        keep_linear=lambda x,w,b:observed_compile(x,w,b)[0]
    else:
        preserve=torch.compile(base,options={'emulate_precision_casts':True})
        def compiled_rope(qn,kn,grid,freqs):
            return rope3d(torch,qn.reshape(qn.shape[0],qn.shape[1],12,128),grid,freqs),rope3d(torch,kn.reshape(kn.shape[0],kn.shape[1],12,128),grid,freqs)
        rope_only=torch.compile(compiled_rope)
        def eager_norm_boundary(q,k,qw,kw,grid,freqs):
            return rope_only(rmsnorm(torch,q,qw),rmsnorm(torch,k,kw),grid,freqs)
        path=resolve_candidate(ROOT,load_json(ROOT/'configs/candidates-r02.json'),args.task)
        spec=importlib.util.spec_from_file_location('qk_probe_original',path)
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        candidate=mod.run
        report['candidate_sha256']=digest(path)
    with torch.inference_mode():
        for seed,stress in [(17,False),(18,False),(19,True)]:
            values=make_inputs(torch,args.task,args.case,seed,torch.bfloat16,torch.device('cuda:0'))
            if stress:values[0].add_(2)
            target=base(*values)
            if args.task=='wan-linear-gelu':
                record('compile-no-patterns',no_patterns,values,target,seed)
                record('compile-observable-linear',keep_linear,values,target,seed)
                x,w,b=values
                mm=torch.nn.functional.linear(x,w,None)
                simulate=torch.nn.functional.gelu(mm.float()+b.float(),approximate='tanh').bfloat16()
                whole=compiled(*values)
                record('simulate-mm-bf16-bias-gelu-vs-compile',lambda:simulate,(),whole,seed)
                ratios=(whole.double()-target.double()).abs()/(.01+.01*target.double().abs())
                idx=int(ratios.flatten().argmax().item()); row,col=divmod(idx,w.shape[0])
                xr=x.reshape(-1,x.shape[-1])[row]
                dot=(xr.double()*w[col].double()).sum()
                report['counterexamples'].append({'seed':seed,'row':row,'col':col,
                    'dot_fp64_without_bias':dot.item(),'bias_bf16':b[col].item(),
                    'mm_bf16':mm.reshape(-1,w.shape[0])[row,col].item(),
                    'linear_bf16':torch.nn.functional.linear(x,w,b).reshape(-1,w.shape[0])[row,col].item(),
                    'dot_then_bias_rounded_bf16':(dot+b[col].double()).bfloat16().item(),
                    'eager_gelu':target.flatten()[idx].item(),'compiled_gelu':whole.flatten()[idx].item(),
                    'simulated_gelu':simulate.flatten()[idx].item(),'error_ratio':ratios.flatten()[idx].item()})
            else:
                record('eager-norm-compiled-rope-boundary',eager_norm_boundary,values,target,seed)
                if seed!=18:continue
                q,k,qw,kw,grid,freqs=values
                norm_shape=q.shape
                identity=torch.ones_like(freqs)
                normalized=candidate(q,k,torch.ones_like(qw),torch.ones_like(kw),grid,identity)
                weighted=candidate(q,k,qw,kw,grid,identity)
                actual=preserve(*values)
                for side,x,weight,a,t,cn,cw in zip(['q','k'],[q,k],[qw,kw],actual,target,normalized,weighted):
                    ratio=(a.double()-t.double()).abs()/(.01+.01*t.double().abs())
                    bad=(ratio>1).flatten().nonzero().flatten()[:8]
                    for index in bad.tolist():
                        row,col=divmod(index,1536); even=col-col%2
                        xr=x.reshape(-1,1536)[row]
                        mean32=xr.float().square().mean(); inv32=torch.rsqrt(mean32+1e-6)
                        mean64=xr.double().square().mean(); inv64=torch.rsqrt(mean64+1e-6)
                        eager_norm=(xr.float()*inv32).bfloat16()
                        norm64=xr.double()*inv64
                        # A framework double->BF16 cast may itself pass through
                        # FP32. At a BF16 midpoint that would invalidate the oracle.
                        oracle_norm=torch.tensor([nearest_bf16(v) for v in norm64.tolist()],device=x.device,dtype=torch.bfloat16)
                        full_norm=rmsnorm(torch,x,torch.ones_like(weight)).reshape(-1,1536)[row]
                        full_weighted=rmsnorm(torch,x,weight).reshape(-1,1536)[row]
                        pair=lambda z:[float(v) for v in z[even:even+2].tolist()]
                        # Re-evaluate only the affected two components with the
                        # high-precision reduction and the same BF16 boundaries.
                        oracle_weighted=(oracle_norm*weight).reshape(12,128)
                        isolated=x.new_zeros(x.shape)
                        isolated.reshape(-1,1536)[row]=oracle_weighted.flatten()
                        oracle_rot=rope3d(torch,isolated.reshape(x.shape[0],x.shape[1],12,128),grid,freqs)
                        report['counterexamples'].append({'side':side,'seed':seed,'row':row,'col':col,
                            'ratio':ratio.flatten()[index].item(),'compiled_preserve':a.flatten()[index].item(),
                            'reference':t.flatten()[index].item(),'oracle_output':oracle_rot.flatten()[index].item(),
                            'mean32':mean32.item(),'mean64':mean64.item(),'inv32':inv32.item(),'inv64':inv64.item(),
                            'input_pair':pair(xr),'weight_pair':pair(weight),
                            'normalized_fp64_pair':pair(xr.double()*inv64),
                            'framework_fp64_cast_pair':pair(norm64.bfloat16()),
                            'eager_row_norm_pair':pair(eager_norm),'eager_full_norm_pair':pair(full_norm),
                            'oracle_bf16_norm_pair':pair(oracle_norm),'candidate_norm_pair':pair(cn.reshape(-1,1536)[row]),
                            'eager_weighted_pair':pair(full_weighted),'candidate_weighted_pair':pair(cw.reshape(-1,1536)[row])})
            del values,target
    dump_new(args.output,report)


if __name__=='__main__':main()

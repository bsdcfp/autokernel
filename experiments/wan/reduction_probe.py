"""Source-derived ATen reduction ablation, never a scored Agent candidate."""
import argparse
import os
from pathlib import Path
from preflight import ROOT
from wanbench.core import dump_new, digest, load_json, validate_device_selection


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--output', required=True, type=Path)
    args=ap.parse_args()
    validate_device_selection(load_json(ROOT/'configs/device.json'), os.environ.get('CUDA_VISIBLE_DEVICES'), 0)
    if args.output.exists(): raise FileExistsError(args.output)
    import torch
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
    from wanbench.tasks import make_inputs
    if 'B300' not in torch.cuda.get_device_name(0): raise RuntimeError('requires B300')

    @triton.jit
    def probe(X, OUT, ROWS:tl.constexpr):
        row=tl.program_id(0)
        lane=tl.arange(0,32)
        a0=tl.full((32,),0,tl.float32)
        a1=tl.full((32,),0,tl.float32)
        a2=tl.full((32,),0,tl.float32)
        a3=tl.full((32,),0,tl.float32)
        for step in tl.static_range(12):
            off=row*1536+lane*4+step*128
            x0=tl.load(X+off).to(tl.float32)
            x1=tl.load(X+off+1).to(tl.float32)
            x2=tl.load(X+off+2).to(tl.float32)
            x3=tl.load(X+off+3).to(tl.float32)
            a0=a0+x0*x0
            a1=a1+x1*x1
            a2=a2+x2*x2
            a3=a3+x3*x3
        combined=((a0+a1)+a2)+a3
        mean=tl.sum(combined,0)*(1.0/1536.0)
        eps_mean=mean+1.0e-6
        inv_approx=tl.rsqrt(eps_mean)
        inv_lib=libdevice.rsqrt(eps_mean)
        tl.store(OUT+row,mean)
        tl.store(OUT+ROWS+row,inv_approx)
        tl.store(OUT+2*ROWS+row,inv_lib)

    report={'kind':'source-derived-reduction-probe','case':args.case,
            'diagnostic_sha256':digest(Path(__file__)), 'results':[], 'counterexamples':[]}
    def stats(a,b):
        return {'unequal':int((a!=b).sum().item()),'max_abs':float((a-b).abs().max().item())}
    with torch.inference_mode():
        for seed,shift in [(17,False),(18,False),(19,True),(31,False),(32,False),(33,True)]:
            values=make_inputs(torch,'t2-qknorm-rope',args.case,seed,torch.bfloat16,torch.device('cuda'))
            if shift: values[0].add_(2)
            for side,x in zip(['q','k'],values[:2]):
                rows=x.numel()//1536
                out=torch.empty((3,rows),device=x.device,dtype=torch.float32)
                probe[(rows,)](x,out,rows,num_warps=4,enable_fp_fusion=False)
                f=x.float().reshape(rows,1536)
                mean=f.square().mean(-1)
                inv=torch.rsqrt(mean+1e-6)
                norm=(f*inv[:,None]).bfloat16()
                s={'seed':seed,'shifted':shift,'side':side,'mean':stats(out[0],mean),
                   'inv_approx':stats(out[1],inv),'inv_lib':stats(out[2],inv),
                   'norm_approx':stats((f*out[1,:,None]).bfloat16(),norm),
                   'norm_lib':stats((f*out[2,:,None]).bfloat16(),norm)}
                report['results'].append(s)
                print(seed,side,{k:v['unequal'] for k,v in s.items() if isinstance(v,dict)},flush=True)
                if args.case=='long' and seed==18 and side=='k':
                    report['counterexamples'].append({'row':28564,'native_mean':mean[28564].item(),
                        'native_inv':inv[28564].item(),'probe':out[:,28564].tolist()})
            del values
    dump_new(args.output,report)


if __name__=='__main__':main()

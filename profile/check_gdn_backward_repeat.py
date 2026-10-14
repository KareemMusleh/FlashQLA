"""Probe main-backward repeatability while varying gradient-buffer allocations."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flash_qla.ops.gated_delta_rule.chunk.hopper.fused_bwd import fused_gdr_bwd


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlen", type=int, default=131072)
    parser.add_argument("--hk", type=int, default=16)
    parser.add_argument("--hv", type=int, default=48)
    parser.add_argument("--parts", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--real", action="store_true", help="Use standard forward inverse and reconstructed states")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.seqlen % (64 * args.parts) == 0
    torch.set_num_threads(4)
    kernel = fused_gdr_bwd
    if args.source:
        name = "flash_qla.ops.gated_delta_rule.chunk.hopper._repeat_candidate"
        spec = importlib.util.spec_from_file_location(name, args.source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        kernel = module.fused_gdr_bwd
    torch.manual_seed(404)
    t, hk, hv = args.seqlen, args.hk, args.hv
    q, k = [F.normalize(torch.randn(1, t, hk, 128, device="cuda"), dim=-1).bfloat16() for _ in range(2)]
    v = torch.randn(1, t, hv, 128, device="cuda", dtype=torch.bfloat16)
    do = torch.randn_like(v)
    a = torch.eye(64, device="cuda", dtype=torch.bfloat16)[None, :, None, :].expand(t // 64, 64, hv, 64).reshape(1, t, hv, 64).contiguous()
    g = torch.full((1, t, hv), -0.001, device="cuda").reshape(1, t // 64, 64, hv).cumsum(2).reshape(1, t, hv)
    beta = torch.full_like(g, 0.5)
    # Arbitrary but fixed incoming chunk states isolate this kernel's memory
    # handoffs, independently of forward reconstruction or inverse precision.
    h = torch.randn(1, t // 64, hv, 128, 128, device="cuda", dtype=torch.bfloat16) * 0.01
    dht = torch.zeros(args.parts, hv, 128, 128, device="cuda")
    cu = torch.arange(args.parts + 1, device="cuda", dtype=torch.int32) * (t // args.parts)
    if args.real:
        import flash_qla.ops.gated_delta_rule.chunk as chunk
        from flash_qla.ops.gated_delta_rule.chunk.hopper.fast_fwd import _training_cp_context
        raw_g = F.logsigmoid(torch.randn_like(g)) * 0.001
        beta = torch.randn_like(g).sigmoid()
        g, a, unused_output, h, unused_final, cache = chunk.chunk_gated_delta_rule_fwd(
            q, k, v, raw_g, beta, output_h=True, output_final_state=False,
            cp_context=_training_cp_context(q.device, t, args.parts), enable_fwd_cp_cache=True,
        )
        del raw_g, unused_output, unused_final, cache
    fixture = dict(q=q, k=k, v=v, a=a, g=g, b=beta, do=do, h=h, dht=dht, cu_seqlens=cu)
    expected = kernel(**fixture)
    torch.cuda.synchronize()
    results = dict(seqlen=t, hk=hk, hv=hv, parts=args.parts, source=str(args.source),
                   fixture="Standard forward preparation" if args.real else "Identity inverse and arbitrary fixed chunk states; a determinism probe, not a recurrence accuracy test", repeats=[])
    for i in range(args.repeats):
        # Different sizes/pointers and simultaneous live output buffers exercise
        # both first-use and reused allocations. Sentinel allocations remain live.
        sentinel = torch.full((1, 64 * (i + 1), hv, 128), i + 1, device="cuda", dtype=torch.bfloat16)
        actual = kernel(**fixture)
        torch.cuda.synchronize()
        entry = {}
        for name, x, y in zip(("dq", "dk", "dv", "dg", "db", "dh0"), actual, expected):
            different = x != y
            count = different.sum().item()
            entry[name] = dict(changed_elements=count, finite=bool(x.isfinite().all()))
            if count:
                idx = different.nonzero()[:8]
                entry[name]["first_indices"] = idx.cpu().tolist()
                entry[name]["relative_rms"] = ((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-30)).item()
        results["repeats"].append(entry)
        del actual, sentinel
        if i % 3 == 0:
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print({name: max(e[name]["changed_elements"] for e in results["repeats"])
           for name in ("dq", "dk", "dv", "dg", "db", "dh0")}, flush=True)


if __name__ == "__main__":
    main()

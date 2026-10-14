"""GatedDeltaNet forward benchmark matching YDT gdn.py and delta_rule.py.

Synthetic weights, one unpadded sequence, no state cache or context parallelism.
Includes all six input projections, causal convolution/SiLU, FP32 decay/beta,
Q/K L2 normalization, GDN, RMSNorm/SiLU output gating and output projection.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_prepare_mn import benchmark, error
from fla.modules.conv.causal_conv1d import causal_conv1d
from flash_qla import chunk_gated_delta_rule
from flash_qla.utils.math import l2norm_fwd
from flash_qla.ops.gated_delta_rule.chunk.hopper.norm_gate import norm_gate_fwd, norm_gate_fwd_compiled
from flash_qla.ops.gated_delta_rule.chunk.hopper.prepare_qkv import prepare_qkv_fwd


class Layer:
    def __init__(self, args):
        self.args = args
        self.hk, self.hv, self.d = args.key_heads, args.value_heads, 128
        kd, vd = self.hk * self.d, self.hv * self.d
        self.x = torch.randn(args.seqlen, args.hidden_size, device='cuda', dtype=torch.bfloat16)
        self.weights = [torch.randn(n, args.hidden_size, device='cuda', dtype=torch.bfloat16)
                        / args.hidden_size**.5 for n in (kd, kd, vd, vd, self.hv, self.hv)]
        # Pack once, outside timed forwards; runtime always includes output-layout costs.
        self.projection_sizes = (kd, kd, vd, vd, self.hv, self.hv)
        self.packed_weight = torch.cat(self.weights)
        self.qkv_weight = torch.cat(self.weights[:3])
        self.zab_weight = torch.cat(self.weights[3:])
        self.conv_weights = [torch.randn(n, args.conv_size, device='cuda', dtype=torch.bfloat16)
                             / args.conv_size**.5 for n in (kd, kd, vd)]
        self.wo = torch.randn(args.hidden_size, vd, device='cuda', dtype=torch.bfloat16) / vd**.5
        self.norm_weight = torch.ones(self.d, device='cuda', dtype=torch.bfloat16)
        self.a_log = torch.rand(self.hv, device='cuda').mul_(16).clamp_min_(1e-6).log_()
        dt = torch.rand(self.hv, device='cuda').mul_(math.log(args.dt_max / .001))
        dt = dt.add_(math.log(.001)).exp_()
        self.dt_bias = dt + torch.log(-torch.expm1(-dt))
        self.cu_cpu = torch.tensor([0, args.seqlen], dtype=torch.int64)
        self.cu = self.cu_cpu.cuda()

    def project(self, mode=None):
        mode = mode or self.args.projection_mode
        if mode == 'packed':
            return F.linear(self.x, self.packed_weight).split(self.projection_sizes, dim=-1)
        if mode == 'packed_contiguous':
            return tuple(x.contiguous() for x in F.linear(self.x, self.packed_weight).split(self.projection_sizes, dim=-1))
        if mode == 'qkv_zab':
            qkv = F.linear(self.x, self.qkv_weight).split(self.projection_sizes[:3], dim=-1)
            zab = F.linear(self.x, self.zab_weight).split(self.projection_sizes[3:], dim=-1)
            return qkv + zab
        # Same order as YDT: q, k, v, output gate, decay, beta.
        return tuple(F.linear(self.x, w) for w in self.weights)

    def conv(self, projections, backend=None):
        backend = backend or self.args.conv_backend
        if backend == 'dao':
            from causal_conv1d import causal_conv1d_fn
            # One sequence with zero history; transposes are views of [T,C].
            return tuple(causal_conv1d_fn(x[None].transpose(1, 2), w,
                                         activation='silu').transpose(1, 2)
                         for x, w in zip(projections[:3], self.conv_weights))
        return tuple(causal_conv1d(x=x[None], weight=w, activation='silu',
                                  backend=backend,
                                  cu_seqlens=self.cu, cu_seqlens_cpu=self.cu_cpu)[0]
                     for x, w in zip(projections[:3], self.conv_weights))

    def gates(self, projections):
        alpha, beta = projections[4:]
        return (-self.a_log.float().exp() * F.softplus(alpha.float() + self.dt_bias.float()))[None], beta.float().sigmoid()[None]

    def normalize(self, convolved):
        q, k, v = convolved
        q = q.view(1, -1, self.hk, self.d)
        k = k.view_as(q)
        return l2norm_fwd(q)[0], l2norm_fwd(k)[0], v.view(1, -1, self.hv, self.d)

    def prepare_qkv(self, projections, mode=None):
        mode = mode or getattr(self.args, 'preparation_mode', 'separate')
        if mode == 'fused':
            return prepare_qkv_fwd(projections, self.conv_weights, self.hk, self.hv)
        if mode == 'fused_gates':
            return prepare_qkv_fwd(projections, self.conv_weights, self.hk, self.hv,
                                   gate_biases=(self.a_log, self.dt_bias))
        return self.normalize(self.conv(projections))

    def gdn(self, normalized, gates, backend='optimized'):
        return chunk_gated_delta_rule(*normalized, *gates, auto_cp=False,
                                      fwd_backend=backend, fwd_partitions=self.args.partitions,
                                      fwd_decay_threshold=None, output_final_state=False)[0]

    def norm_gate(self, output, projections, mode=None):
        mode = mode or self.args.norm_gate_mode
        if mode == 'fused':
            return norm_gate_fwd(output, projections[3], self.norm_weight, self.args.norm_eps)
        if mode == 'compiled':
            return norm_gate_fwd_compiled(output, projections[3], self.norm_weight, self.args.norm_eps)
        z = projections[3].view_as(output)
        return F.rms_norm(output, (self.d,), self.norm_weight, self.args.norm_eps).type_as(output) * F.silu(z)

    def output_project(self, gated):
        return F.linear(gated.view(self.args.seqlen, -1), self.wo)

    def forward(self, backend=None, labels=False, conv_backend=None,
                projection_mode=None, norm_gate_mode=None, preparation_mode=None):
        backend = backend or self.args.gdn_backend
        preparation_mode = preparation_mode or getattr(self.args, 'preparation_mode', 'separate')
        def region(name):
            return torch.profiler.record_function(name) if labels else nullcontext()
        with region('layer/input_projections_q_k_v_z_a_b'):
            p = self.project(projection_mode)
        if preparation_mode == 'fused_gates':
            with region('layer/fused_convolution_silu_qk_normalization_gates'):
                n, g = self.prepare_qkv(p, 'fused_gates')
        elif preparation_mode == 'fused':
            with region('layer/fused_convolution_silu_qk_normalization'):
                n = self.prepare_qkv(p, 'fused')
        else:
            with region('layer/causal_convolution_silu'):
                c = self.conv(p, conv_backend)
        if preparation_mode != 'fused_gates':
            with region('layer/decay_and_beta'):
                g = self.gates(p)
        if preparation_mode == 'separate':
            with region('layer/qk_l2_normalization'):
                n = self.normalize(c)
        with region('layer/gdn'):
            o = self.gdn(n, g, backend)
        with region('layer/output_rmsnorm_silu_gate'):
            gated = self.norm_gate(o, p, norm_gate_mode)
        with region('layer/output_projection'):
            return self.output_project(gated)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seqlen', type=int, default=16384)
    parser.add_argument('--hidden-size', type=int, default=2560,
                        help='Qwen3.8-Flash-Next projection hidden size')
    parser.add_argument('--key-heads', type=int, default=16)
    parser.add_argument('--value-heads', type=int, default=48)
    parser.add_argument('--conv-size', type=int, default=4)
    parser.add_argument('--conv-backend', choices=('cuda', 'triton', 'dao'), default='cuda',
                        help='Causal convolution implementation; CUDA requires causal-conv1d')
    parser.add_argument('--compare-conv-backends', action='store_true',
                        help='Compare FLA Triton, FLA CUDA wrapper, and direct Dao CUDA, including full layer')
    parser.add_argument('--norm-eps', type=float, default=1e-6)
    parser.add_argument('--norm-gate-mode', choices=('eager', 'fused', 'compiled'), default='fused')
    parser.add_argument('--projection-mode', choices=('separate', 'packed', 'packed_contiguous', 'qkv_zab'), default='packed')
    parser.add_argument('--preparation-mode', choices=('separate', 'fused', 'fused_gates'), default='fused_gates')
    parser.add_argument('--dt-max', type=float, default=.1,
                        help='Synthetic dt initialization upper bound (YDT uses .1)')
    parser.add_argument('--partitions', type=int, default=8)
    parser.add_argument('--gdn-backend', choices=('default', 'optimized'), default='optimized')
    parser.add_argument('--output', type=Path, default=Path('traces/qwen38_flash_next_layer_16k'))
    args = parser.parse_args()
    if args.seqlen % (args.partitions * 64) or args.value_heads % args.key_heads:
        parser.error('sequence length must align to partitions*64; value heads must divide by key heads')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    layer = Layer(args)
    p = layer.project()
    c = layer.conv(p)
    g = layer.gates(p)
    n = layer.normalize(c)
    if args.preparation_mode == 'fused_gates':
        selected_n, selected_g = layer.prepare_qkv(p)
    elif args.preparation_mode == 'fused':
        selected_n, selected_g = layer.prepare_qkv(p), g
    else:
        selected_n, selected_g = n, g
    print('Validate optimized GDN and complete layer', flush=True)
    ref = layer.gdn(n, g, 'default')
    opt = layer.gdn(selected_n, selected_g, args.gdn_backend)
    reference = layer.output_project(layer.norm_gate(ref, p, 'eager'))
    output = layer.forward()
    checks = {'gdn_relative_rms': error(opt, ref) if opt.isfinite().all() else None,
              'layer_relative_rms': error(output, reference) if output.isfinite().all() else None,
              'finite': bool(output.isfinite().all()), 'reference_finite': bool(ref.isfinite().all()),
              'minimum_chunk_log_decay': g[0].reshape(1, -1, 64, args.value_heads).sum(2).min().item()}
    checks['preparation_qkv_relative_rms'] = [error(a,b) for a,b in zip(selected_n,n)]
    checks['preparation_gates_relative_rms'] = [error(a,b) for a,b in zip(selected_g,g)]
    assert max(checks['preparation_qkv_relative_rms']) < .004 and max(checks['preparation_gates_relative_rms']) < 1e-6, checks
    conv_errors = []
    for projected, weight, actual in zip(p[:3], layer.conv_weights, c):
        length = min(args.seqlen, 128)
        expected = F.conv1d(projected[:length].float().T[None], weight.float()[:, None],
                            padding=args.conv_size-1, groups=weight.shape[0])[..., :length]
        expected = F.silu(expected).transpose(1, 2).bfloat16()
        conv_errors.append(error(actual[:, :length], expected))
    checks['convolution_relative_rms_vs_torch'] = conv_errors
    (args.output / 'validation.json').write_text(json.dumps(checks, indent=2) + '\n')
    assert max(conv_errors) < .003, checks
    assert checks['finite'] and max(checks['gdn_relative_rms'], checks['layer_relative_rms']) < .012, checks
    gated = layer.norm_gate(opt, p)
    functions = {
        'input_projections': layer.project,
        'convolution_silu': lambda: layer.conv(p),
        'decay_beta': lambda: layer.gates(p),
        'qk_normalization': lambda: layer.normalize(c),
        'gdn_' + args.gdn_backend: lambda: layer.gdn(selected_n, selected_g, args.gdn_backend),
        'output_norm_gate': lambda: layer.norm_gate(opt, p),
        'output_projection': lambda: layer.output_project(gated),
        'layer_' + args.gdn_backend: layer.forward,
    }
    if args.preparation_mode != 'separate':
        functions['preparation_' + args.preparation_mode] = lambda: layer.prepare_qkv(p)
    if args.compare_conv_backends:
        checks['convolution_backends_relative_rms'] = {}
        checks['layer_conv_backends_relative_rms'] = {}
        for backend in ('triton', 'cuda', 'dao'):
            candidate = layer.conv(p, backend)
            differences = [error(x, y) for x, y in zip(candidate, c)]
            checks['convolution_backends_relative_rms'][backend] = differences
            assert max(differences) < .003, (backend, differences)
            layer_error = error(layer.forward(conv_backend=backend, preparation_mode='separate'), reference)
            checks['layer_conv_backends_relative_rms'][backend] = layer_error
            assert math.isfinite(layer_error) and layer_error < .012, (backend, layer_error)
            functions['convolution_' + backend] = lambda backend=backend: layer.conv(p, backend)
            functions['layer_conv_' + backend] = lambda backend=backend: layer.forward(conv_backend=backend, preparation_mode='separate')
        (args.output / 'validation.json').write_text(json.dumps(checks, indent=2) + '\n')
    print('Benchmark stages and complete forward', flush=True)
    results = {'gpu': torch.cuda.get_device_name(), 'config': vars(args) | {'output': str(args.output)},
               'shapes': {'input': [args.seqlen, args.hidden_size],
                          'input_projection_weights_q_k_v_z_a_b': [list(w.shape) for w in layer.weights],
                          'conv_weights_q_k_v': [list(w.shape) for w in layer.conv_weights],
                          'q_k': [1, args.seqlen, args.key_heads, 128],
                          'v_output': [1, args.seqlen, args.value_heads, 128],
                          'g_beta': [1, args.seqlen, args.value_heads],
                          'output_projection_weight': list(layer.wo.shape)},
               'scope': 'complete GatedDeltaNet attention module forward; synthetic weights; one sequence; no cache/CP; full history; no residual or MLP',
               'initialization': 'unit-variance scaled linear and conv weights; YDT decay-bias formula with configurable dt upper bound; BF16 activations/weights, FP32 decay/beta',
               'timing': '9 rotating-order CUDA graph samples; 30 replays per sample; stages use precomputed inputs; end-to-end recomputes all intermediates',
               'correctness': checks, 'references': {}}
    results['convolution_implementations'] = {
        'triton': 'FLA Triton forward',
        'cuda': 'FLA wrapper around Dao causal-conv1d; includes sequence metadata preparation',
        'dao': 'Direct Dao causal-conv1d; one sequence, transposes are views, no sequence-id tensor needed',
    }
    results['packages'] = {'torch': torch.__version__}
    if args.conv_backend in ('cuda', 'dao') or args.compare_conv_backends:
        results['packages']['causal-conv1d'] = importlib.metadata.version('causal-conv1d')
    for name in ('gdn.py', 'delta_rule.py'):
        path = Path('/home/kareemm/arcadia/ml/torch/ydt/core/ydt_core/nn/modules/attention') / name
        results['references'][name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None}
    results['timings'] = benchmark(functions, inner_repeats=1)
    (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2), flush=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
        with torch.profiler.record_function('gdn_layer_' + args.gdn_backend):
            layer.forward(labels=True)
        if args.compare_conv_backends:
            for backend in ('triton', 'cuda', 'dao'):
                with torch.profiler.record_function('gdn_layer_conv_' + backend):
                    layer.forward(labels=True, conv_backend=backend, preparation_mode='separate')
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(args.output / 'layer_trace.json'))


if __name__ == '__main__':
    main()

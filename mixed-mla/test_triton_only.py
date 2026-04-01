"""Quick test: pure Triton FP8 kernel on all configs."""
import math, torch, triton
from reference import generate_input, ref_kernel
from triton_fp8_decode import triton_mla_decode_fp8

BENCHMARKS = [
    {"batchsize": 4, "qseqlen": 1, "kvseqlen": 1024, "seed": 4217},
    {"batchsize": 4, "qseqlen": 1, "kvseqlen": 8192, "seed": 4220},
    {"batchsize": 32, "qseqlen": 1, "kvseqlen": 1024, "seed": 5412},
    {"batchsize": 32, "qseqlen": 1, "kvseqlen": 8192, "seed": 5415},
    {"batchsize": 64, "qseqlen": 1, "kvseqlen": 1024, "seed": 1357},
    {"batchsize": 64, "qseqlen": 1, "kvseqlen": 8192, "seed": 1360},
    {"batchsize": 256, "qseqlen": 1, "kvseqlen": 1024, "seed": 9823},
    {"batchsize": 256, "qseqlen": 1, "kvseqlen": 8192, "seed": 9826},
]

print(f"{'bs':>5} {'kv':>6} {'Triton(ms)':>10} {'Ref(ms)':>10} {'Speedup':>8} {'max_diff':>10}")
print("-" * 55)
times_tri, times_ref = [], []
for cfg in BENCHMARKS:
    data = generate_input(**cfg)
    q, kv_data, qo_indptr, kv_indptr, config = data
    kv_fp8, kv_scale = kv_data["fp8"]

    ref_out = ref_kernel(data)
    tri_out = triton_mla_decode_fp8(q, kv_fp8, kv_scale, kv_indptr, config)
    max_diff = (ref_out - tri_out).abs().max().item()

    t_tri = triton.testing.do_bench(
        lambda: triton_mla_decode_fp8(q, kv_fp8, kv_scale, kv_indptr, config),
        warmup=25, rep=100,
    )
    t_ref = triton.testing.do_bench(lambda: ref_kernel(data), warmup=25, rep=100)
    times_tri.append(t_tri)
    times_ref.append(t_ref)
    speedup = t_ref / t_tri
    print(f"{cfg['batchsize']:>5} {cfg['kvseqlen']:>6} {t_tri:>10.3f} {t_ref:>10.3f} {speedup:>7.2f}x {max_diff:>10.4f}")

geo_tri = math.exp(sum(math.log(t) for t in times_tri) / len(times_tri))
geo_ref = math.exp(sum(math.log(t) for t in times_ref) / len(times_ref))
print(f"\nTriton geomean: {geo_tri:.3f} ms")
print(f"Ref    geomean: {geo_ref:.3f} ms")
print(f"Speedup:        {geo_ref/geo_tri:.2f}x")

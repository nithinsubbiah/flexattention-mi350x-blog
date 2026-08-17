# FlexAttention on MI350X — blog draft

Draft of a technical blog post on tuning PyTorch FlexAttention for AMD Instinct MI350X, intended for
submission to [ROCm Blogs](https://rocm.blogs.amd.com/). Shared here for review comments.

**Read the draft: [blog/flexattention-mi350x/README.md](blog/flexattention-mi350x/README.md)**

## Leaving comments

The draft is also open as a pull request, which is the easiest place to comment on specific lines:
open the PR, go to **Files changed**, and click any line. General feedback is welcome as an issue.

## What the post covers

Ten merged upstream changes — five in PyTorch Inductor, five in Triton — that improve FlexAttention
performance on AMD GPUs. The headline results, measured on one MI350X:

| Change | Effect |
|---|---|
| Software pipelining, `num_stages` 1 to 2 | 1.11x to 1.18x on bf16 |
| fp16 wavefront count, `num_warps` 8 to 4 | 1.86x to 2.61x |
| Both together | 2.1x to 3.0x on fp16 forward |

## Reproducing the numbers

Everything quoted in the post comes from these scripts, on one MI350X with ROCm 7.2.1,
PyTorch 2.14.0a0+gitb5460ee, and Triton 3.8.0:

```bash
# Pin to an idle GPU; see the methodology section of the post for why this matters.
HIP_VISIBLE_DEVICES=0 python bench/bench_landed.py --reps 3 > run1.log
HIP_VISIBLE_DEVICES=0 python bench/bench_landed.py --reps 3 > run2.log
HIP_VISIBLE_DEVICES=0 python bench/bench_landed.py --reps 3 > run3.log

python bench/aggregate_landed.py 'run*.log'                # medians + run-to-run spread
python bench/plot_landed.py 'run*.log' blog/flexattention-mi350x/images
```

`aggregate_landed.py` prints the tables used in the post along with the observed spread;
`plot_landed.py` renders the figures from the same logs, so the charts and tables cannot disagree.

## Review status

The draft has been through an independent verification pass on gfx950 hardware. All 18 published
data points re-measured within 0.02x of the reported speedups and ~1% on absolute TFLOP/s, and all
ten referenced PRs were confirmed merged upstream. Three claims were corrected as a result:

- the mechanism behind the `num_warps` result (it is redundant QK^T work plus an LDS layout
  round-trip, not a smaller MFMA instruction),
- the attribution of absolute throughput to the Triton buffer-op work (worth ~1.33x on causal fp16 at
  head dim 128, so "the throughput does not rest on them" was wrong),
- the `TORCHINDUCTOR_EMIT_POINTER_RANGE_32=0` escape hatch, which does not reach template kernels.

The last one is an upstream gap rather than a documentation error: both that flag and the atomics
suppression from #176675 are applied in `TritonKernel.codegen_kernel()`, which template kernels do not
go through. Worth fixing in `TritonTemplateKernel.jit_lines()`.

## Open items before submission

- No author byline yet.
- ROCm Blogs submissions need a YAML front matter block (title, date, author, tags, category, and the
  `myst.html_meta` fields) plus a cover image; neither is in the draft, which is kept as plain prose
  for review.
- The Flash Attention parity comparison is cited from the upstream PR review and was not reproduced
  here, since this build was compiled without the fused attention backends.

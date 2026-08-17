# FlexAttention on MI350X — blog draft

Draft of a technical blog post on tuning PyTorch FlexAttention for AMD Instinct MI350X, intended for
submission to [ROCm Blogs](https://rocm.blogs.amd.com/). Shared here for review comments.

**Read the draft: [blog/flexattention-mi350x/README.md](blog/flexattention-mi350x/README.md)**

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

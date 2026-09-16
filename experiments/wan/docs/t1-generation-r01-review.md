# T1 generation r01 — source review

Status: rejected before GPU execution; not a measured correctness failure.
Candidate SHA256: `6e8410c20b3975b1ccc591fd999958db2743dbe9900689f52bcfbc42e19a08de`
Generation used AutoKernel's program as context, a fixed Wan reference and a
bounded coding-model call. Reported usage: 22,771 input tokens, 1,645 output
tokens (including 1,034 reasoning output tokens). These fields follow the CLI's
reported accounting and should not be double-counted.

## Finding

The kernel launches 2048 lanes for 1536 hidden values. Loads outside the row are
masked to zero, then the mean is subtracted from every lane. Its variance sums
`centered * centered` over all 2048 lanes, including the 512 padded lanes.
Each padded lane incorrectly contributes `mean ** 2` to the variance.

Concrete counterexample: a row with 768 ones and 768 threes has mean 2 and true
variance 1. The candidate computes `(1536 + 512*4)/1536 = 7/3`. Valid normalized
values become approximately ±0.655 rather than ±1 before BF16 rounding.

Next generation should make one focused correction to the masked variance
reduction. Preserve the mandatory intermediate cast and all other interface
semantics. Do not claim a speedup, correctness pass, or successful compilation:
the candidate has not run on GPU. An independent nonzero-mean correctness case
must cover this error after the GPU becomes available.

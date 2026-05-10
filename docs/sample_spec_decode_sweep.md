# Speculative Decoding Sweep

| rank | scenario | draft | accept p | expected accepted | latency ms | tok/s | joules |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | hetero_sram_draft_gpu_verify | 4 | 0.96 | 3.62 | 1457.52 | 686.10 | 853.99 |
| 2 | hetero_sram_draft_gpu_verify | 8 | 0.96 | 6.69 | 1531.08 | 653.13 | 910.78 |
| 3 | hetero_sram_draft_gpu_verify | 4 | 0.93 | 3.35 | 1556.41 | 642.50 | 909.88 |
| 4 | hetero_sram_draft_gpu_verify | 4 | 0.90 | 3.10 | 1664.99 | 600.61 | 971.25 |
| 5 | hetero_sram_draft_gpu_verify | 8 | 0.93 | 5.85 | 1717.69 | 582.18 | 1018.42 |
| 6 | hetero_sram_draft_gpu_verify | 16 | 0.96 | 11.51 | 1727.00 | 579.04 | 1032.81 |
| 7 | hetero_sram_draft_gpu_verify | 8 | 0.90 | 5.13 | 1929.10 | 518.38 | 1140.37 |
| 8 | hetero_sram_draft_gpu_verify | 16 | 0.93 | 9.13 | 2119.78 | 471.75 | 1261.74 |
| 9 | hetero_sram_draft_gpu_verify | 32 | 0.96 | 17.50 | 2190.82 | 456.45 | 1309.14 |
| 10 | hetero_sram_draft_gpu_verify | 16 | 0.90 | 7.33 | 2583.43 | 387.08 | 1531.98 |
| 11 | gpu_spec_decode | 4 | 0.96 | 3.62 | 2989.73 | 334.48 | 2092.81 |
| 12 | hetero_sram_draft_gpu_verify | 32 | 0.93 | 11.98 | 3096.40 | 322.96 | 1839.71 |
| 13 | gpu_spec_decode | 4 | 0.93 | 3.35 | 3211.47 | 311.38 | 2248.03 |
| 14 | gpu_spec_decode | 8 | 0.96 | 6.69 | 3215.03 | 311.04 | 2250.52 |
| 15 | gpu_spec_decode | 4 | 0.90 | 3.10 | 3454.91 | 289.44 | 2418.44 |

## Interpretation

- fastest scenario: hetero_sram_draft_gpu_verify
- why: draft on SRAM-like accelerator, prefill/verify on GPU; wins when transfer is small
- failure mode to watch: longer draft sequences only help when acceptance stays high enough to amortize rejected tokens

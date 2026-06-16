# Speculative-decoding benchmark — Qwen2.5-1.5B-Instruct (vLLM, A100-80GB)

Online `vllm bench serve` driver; median of up to 3 runs/cell. Acceptance from the bench-serve JSON (`spec_decode_acceptance_rate/_length`). Cells grouped by server-config, one server lifetime each (plan §0.5).

> **48/60 cells — complete deliverable.** The 2 EAGLE-3 group(s) (eagle3_st4_auto, eagle3_st4_fp8) are excluded: no valid Qwen2.5-1.5B EAGLE-3 head exists (Decision 1); n-gram is the primary method.


## sharegpt — load (max-concurrency) 1

| group | method | kv | spec_tok | acc % | acc_len | TTFT ms | TPOT ms | out tok/s | GPU MB |
|---|---|---|---|---|---|---|---|---|---|
| baseline_auto | baseline | auto | — | — | — | 13.4 | 4.08 | 241.5 | 73872 |
| baseline_fp8 | baseline | fp8 | — | — | — | 15.7 | 4.42 | 221.5 | 74326 |
| ngram_st2_auto | ngram | auto | 2 | 30.5 | 1.61 | 11.7 | 4.92 | 221.8 | 73998 |
| ngram_st2_fp8 | ngram | fp8 | 2 | 40.9 | 1.82 | 13.1 | 5.64 | 203.4 | 74390 |
| ngram_st4_auto | ngram | auto | 4 | 21.5 | 1.86 | 12.8 | 4.68 | 239.3 | 74080 |
| ngram_st4_fp8 | ngram | fp8 | 4 | 28.8 | 2.15 | 13.4 | 5.36 | 227.4 | 74452 |
| ngram_st8_auto | ngram | auto | 8 | 13.8 | 2.09 | 11.9 | 4.65 | 260.0 | 74174 |
| ngram_st8_fp8 | ngram | fp8 | 8 | 18.3 | 2.45 | 13.4 | 5.14 | 246.2 | 74530 |

## sharegpt — load (max-concurrency) 8

| group | method | kv | spec_tok | acc % | acc_len | TTFT ms | TPOT ms | out tok/s | GPU MB |
|---|---|---|---|---|---|---|---|---|---|
| baseline_auto | baseline | auto | — | — | — | 17.8 | 4.22 | 1778.4 | 73872 |
| baseline_fp8 | baseline | fp8 | — | — | — | 20.7 | 4.51 | 1657.0 | 74326 |
| ngram_st2_auto | ngram | auto | 2 | 32.6 | 1.65 | 13.8 | 5.40 | 1600.7 | 73998 |
| ngram_st2_fp8 | ngram | fp8 | 2 | 41.0 | 1.82 | 20.6 | 7.61 | 1193.3 | 74390 |
| ngram_st4_auto | ngram | auto | 4 | 20.8 | 1.83 | 13.8 | 5.29 | 1691.4 | 74080 |
| ngram_st4_fp8 | ngram | fp8 | 4 | 28.6 | 2.14 | 20.9 | 7.35 | 1275.1 | 74452 |
| ngram_st8_auto | ngram | auto | 8 | 13.1 | 2.03 | 14.3 | 5.44 | 1766.1 | 74388 |
| ngram_st8_fp8 | ngram | fp8 | 8 | 16.9 | 2.33 | 21.2 | 7.38 | 1323.6 | 74750 |

## sharegpt — load (max-concurrency) 32

| group | method | kv | spec_tok | acc % | acc_len | TTFT ms | TPOT ms | out tok/s | GPU MB |
|---|---|---|---|---|---|---|---|---|---|
| baseline_auto | baseline | auto | — | — | — | 22.8 | 5.07 | 4790.5 | 73872 |
| baseline_fp8 | baseline | fp8 | — | — | — | 27.7 | 5.41 | 4521.1 | 74326 |
| ngram_st2_auto | ngram | auto | 2 | 32.4 | 1.65 | 17.9 | 7.03 | 4039.8 | 74098 |
| ngram_st2_fp8 | ngram | fp8 | 2 | 41.0 | 1.82 | 22.1 | 8.47 | 3598.5 | 74756 |
| ngram_st4_auto | ngram | auto | 4 | 21.4 | 1.85 | 19.0 | 7.02 | 4280.6 | 75266 |
| ngram_st4_fp8 | ngram | fp8 | 4 | 28.6 | 2.14 | 23.4 | 8.70 | 3514.3 | 75494 |
| ngram_st8_auto | ngram | auto | 8 | 12.3 | 1.97 | 22.1 | 8.42 | 3691.2 | 77738 |
| ngram_st8_fp8 | ngram | fp8 | 8 | 17.1 | 2.35 | 25.5 | 8.95 | 3816.4 | 79350 |

## humaneval — load (max-concurrency) 1

| group | method | kv | spec_tok | acc % | acc_len | TTFT ms | TPOT ms | out tok/s | GPU MB |
|---|---|---|---|---|---|---|---|---|---|
| baseline_auto | baseline | auto | — | — | — | 14.0 | 4.08 | 242.4 | 73872 |
| baseline_fp8 | baseline | fp8 | — | — | — | 16.4 | 4.43 | 222.3 | 74326 |
| ngram_st2_auto | ngram | auto | 2 | 28.8 | 1.58 | 13.0 | 4.45 | 226.0 | 74098 |
| ngram_st2_fp8 | ngram | fp8 | 2 | 35.2 | 1.70 | 13.7 | 5.14 | 194.7 | 74756 |
| ngram_st4_auto | ngram | auto | 4 | 18.6 | 1.74 | 12.0 | 4.23 | 240.5 | 75266 |
| ngram_st4_fp8 | ngram | fp8 | 4 | 23.3 | 1.93 | 13.7 | 4.90 | 205.6 | 75494 |
| ngram_st8_auto | ngram | auto | 8 | 11.1 | 1.88 | 12.1 | 3.96 | 254.3 | 77738 |
| ngram_st8_fp8 | ngram | fp8 | 8 | 14.6 | 2.15 | 13.7 | 4.46 | 226.0 | 79350 |

## humaneval — load (max-concurrency) 8

| group | method | kv | spec_tok | acc % | acc_len | TTFT ms | TPOT ms | out tok/s | GPU MB |
|---|---|---|---|---|---|---|---|---|---|
| baseline_auto | baseline | auto | — | — | — | 17.5 | 4.18 | 1865.2 | 73872 |
| baseline_fp8 | baseline | fp8 | — | — | — | 20.7 | 4.47 | 1726.0 | 74326 |
| ngram_st2_auto | ngram | auto | 2 | 28.6 | 1.57 | 13.5 | 4.83 | 1644.2 | 74098 |
| ngram_st2_fp8 | ngram | fp8 | 2 | 34.8 | 1.69 | 20.4 | 6.76 | 1166.9 | 74756 |
| ngram_st4_auto | ngram | auto | 4 | 18.3 | 1.73 | 13.2 | 4.53 | 1765.5 | 75266 |
| ngram_st4_fp8 | ngram | fp8 | 4 | 23.6 | 1.94 | 20.9 | 6.41 | 1237.6 | 75494 |
| ngram_st8_auto | ngram | auto | 8 | 10.8 | 1.86 | 13.5 | 4.52 | 1766.1 | 77738 |
| ngram_st8_fp8 | ngram | fp8 | 8 | 14.7 | 2.16 | 21.1 | 6.05 | 1289.3 | 79350 |

## humaneval — load (max-concurrency) 32

| group | method | kv | spec_tok | acc % | acc_len | TTFT ms | TPOT ms | out tok/s | GPU MB |
|---|---|---|---|---|---|---|---|---|---|
| baseline_auto | baseline | auto | — | — | — | 53.9 | 4.67 | 6503.2 | 73872 |
| baseline_fp8 | baseline | fp8 | — | — | — | 54.2 | 5.17 | 5787.7 | 74326 |
| ngram_st2_auto | ngram | auto | 2 | 28.8 | 1.58 | 17.0 | 6.08 | 5016.4 | 74132 |
| ngram_st2_fp8 | ngram | fp8 | 2 | 35.0 | 1.70 | 22.2 | 7.57 | 3985.0 | 74756 |
| ngram_st4_auto | ngram | auto | 4 | 18.5 | 1.74 | 18.2 | 6.12 | 4998.9 | 75266 |
| ngram_st4_fp8 | ngram | fp8 | 4 | 23.9 | 1.95 | 23.7 | 7.46 | 4044.9 | 75896 |
| ngram_st8_auto | ngram | auto | 8 | 10.8 | 1.86 | 21.5 | 7.31 | 4244.4 | 78128 |
| ngram_st8_fp8 | ngram | fp8 | 8 | 14.0 | 2.10 | 25.4 | 7.72 | 3893.7 | 79740 |

## Caveats

- n-gram is the primary spec method; EAGLE-3 is best-effort with a community 1.5B head (Decision 1) — absent rows mean its server failed to start.
- 8 GB host: online server + bench client coexist (go/no-go smoke passed); runs are strictly sequential, one server at a time.
- GPU MB is the server's steady allocation sampled per cell (nvidia-smi), not a true per-request peak.

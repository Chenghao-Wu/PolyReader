# Marker vs MinerU — parser benchmark (5 polymer papers)

## Corpus

| case | documents |
|---|---|
| P1_natcommun_lactone | s41467-024-52090-2.pdf (main), s41467-024-52090-2_SI.pdf (si) |
| P2_polym_star_raft | polymers-15-02002.pdf (main), polymers-2246784-supplementary.pdf (si) |
| P3_polym_tmc_rop | polymers-15-00720.pdf (main) |
| P4_polym_carbonate_scanned | polymers-16-03364_scanned.pdf (main, scanned) |
| P5_pccp_flory_huggins | d4cp03690e.pdf (main), d4cp03690e1_suppl.pdf (si) |

## Setup

| config | backend | version | tier | host | GPU(s) used (index, name, driver, mem total, used + util at start incl. other jobs) | torch | marker-pdf | surya-ocr | device | run start |
|---|---|---|---|---|---|---|---|---|---|---|
| marker | marker | 1.10.2 |  | 6133 | 1, NVIDIA GeForce RTX 3090, 535.288.01, 24576 MiB, 2389 MiB, 47 % | 2.7.1 | 1.10.2 | 0.17.1 | CUDA_VISIBLE_DEVICES=1 | 2026-11-06 15:54:19 |
| marker_4090 | marker | 1.10.2 |  | SA264-30 | 0, NVIDIA GeForce RTX 4090 D, 535.154.05, 24564 MiB, 79 MiB, 0 % | 2.7.1 | 1.10.2 | 0.17.1 | CUDA_VISIBLE_DEVICES=0 | 2026-11-06 15:29:53 |
| mineru_advanced | mineru | 4.1.4 | advanced | 6133 | 0, NVIDIA GeForce RTX 3090, 535.288.01, 24576 MiB, 6488 MiB, 70 %; 1, NVIDIA GeForce RTX 3090, 535.288.01, 24576 MiB, 2389 MiB, 48 % | 2.7.1 | 1.10.2 | 0.17.1 | llama.cpp Vulkan device 2, ONNX gpu; watched GPUs 0,1 | 2026-11-06 15:26:38 |
| mineru_basic | mineru | 4.1.4 | basic | 6133 | 0, NVIDIA GeForce RTX 3090, 535.288.01, 24576 MiB, 6488 MiB, 71 %; 1, NVIDIA GeForce RTX 3090, 535.288.01, 24576 MiB, 2389 MiB, 49 % | 2.7.1 | 1.10.2 | 0.17.1 | llama.cpp Vulkan device 2, ONNX gpu; watched GPUs 0,1 | 2026-11-06 15:11:31 |
| mineru_standard | mineru | 4.1.4 | standard | 6133 | 0, NVIDIA GeForce RTX 3090, 535.288.01, 24576 MiB, 6488 MiB, 70 %; 1, NVIDIA GeForce RTX 3090, 535.288.01, 24576 MiB, 2388 MiB, 75 % | 2.7.1 | 1.10.2 | 0.17.1 | llama.cpp Vulkan device 2, ONNX gpu; watched GPUs 0,1 | 2026-11-06 14:56:13 |

## Speed and memory

| config | host | pages | total s | s/page | pages/s | model init s | cold start s (first doc) | warm s/page | concurrent pages/s (speedup) | peak RSS GB | peak VRAM GB (per-PID) | GPU mem delta GB | failed docs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| marker | 6133 | 106 | 783.8 | 7.39 | 0.135 | 8.2 | 230.8 | 18.26 | 0.088 (1.38×) | 10.5 | 21.3 | 21.2 | 0 |
| marker_4090 | SA264-30 | 106 | 258.2 | 2.44 | 0.411 | 2.3 | 82.3 | 6.60 | 0.195 (0.98×) | 9.5 | 22.7 | 22.8 | 0 |
| mineru_advanced | 6133 | 106 | 1183.4 | 11.16 | 0.090 | – | 199.3 | 16.45 | 0.108 (1.21×) | 5.6 | 6.4 | 6.5 | 0 |
| mineru_basic | 6133 | 106 | 571.0 | 5.39 | 0.186 | – | 166.2 | 13.70 | 0.154 (1.26×) | 5.2 | 5.9 | 6.0 | 0 |
| mineru_standard | 6133 | 106 | 624.5 | 5.89 | 0.170 | – | 144.8 | 11.84 | 0.176 (1.22×) | 4.6 | 6.3 | 6.4 | 0 |

- *total s / s/page / pages/s*: sequential processing of all documents, excluding model init.
- *cold start*: imports + model init + the first document's first parse (first-call CUDA/kernel costs land here). MinerU loads its models inside every per-document subprocess, so every MinerU document time includes model loading.
- *concurrent*: N documents parsed at once by N worker processes (each loads its own models, like N independent Polyextract runs); wall time includes the workers' model loading. Speedup = sum of the same documents' sequential times / concurrent wall time.
- *warm*: the same first document in the measured pass (after the cold run); the per-document and total columns are all measured after the cold run.
- *peak VRAM (per-PID)*: NVML per-process memory of the benchmark process tree; '–' when the driver reported no per-PID usage (e.g. Vulkan). *GPU mem delta*: device-level used-memory rise on the watched GPUs (shared machine: may include other users' jobs).

### Per document

| config | document | role | scanned | PDF pages | parsed pages | s | s/page | peak RSS GB | peak VRAM GB | watched-GPU util % | images | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| marker | P1_natcommun_lactone/s41467-024-52090-2 | main | no | 12 | 12 | 219.1 | 18.26 | 3.3 | 9.4 | 80 | 6 |  |
| marker | P1_natcommun_lactone/s41467-024-52090-2_SI | si | no | 34 | 34 | 28.5 | 0.84 | 10.5 | 6.5 | 68 | 37 |  |
| marker | P2_polym_star_raft/polymers-15-02002 | main | no | 14 | 14 | 189.6 | 13.54 | 3.8 | 9.4 | 84 | 12 |  |
| marker | P2_polym_star_raft/polymers-2246784-supplementary | si | no | 6 | 6 | 21.5 | 3.58 | 3.7 | 7.3 | 78 | 6 |  |
| marker | P3_polym_tmc_rop/polymers-15-00720 | main | no | 11 | 11 | 122.6 | 11.14 | 3.7 | 9.4 | 92 | 11 |  |
| marker | P4_polym_carbonate_scanned/polymers-16-03364_scanned | main | yes | 11 | 11 | 140.3 | 12.76 | 3.7 | 8.8 | 89 | 6 |  |
| marker | P5_pccp_flory_huggins/d4cp03690e | main | no | 12 | 12 | 53.0 | 4.42 | 3.7 | 6.5 | 84 | 7 |  |
| marker | P5_pccp_flory_huggins/d4cp03690e1_suppl | si | no | 6 | 6 | 9.3 | 1.54 | 3.6 | 5.5 | 78 | 4 |  |
| marker_4090 | P1_natcommun_lactone/s41467-024-52090-2 | main | no | 12 | 12 | 79.2 | 6.60 | 3.4 | 9.5 | 86 | 6 |  |
| marker_4090 | P1_natcommun_lactone/s41467-024-52090-2_SI | si | no | 34 | 34 | 8.6 | 0.25 | 3.8 | 6.6 | 40 | 37 |  |
| marker_4090 | P2_polym_star_raft/polymers-15-02002 | main | no | 14 | 14 | 51.5 | 3.68 | 3.8 | 9.6 | 84 | 12 |  |
| marker_4090 | P2_polym_star_raft/polymers-2246784-supplementary | si | no | 6 | 6 | 5.1 | 0.85 | 3.6 | 7.5 | 62 | 6 |  |
| marker_4090 | P3_polym_tmc_rop/polymers-15-00720 | main | no | 11 | 11 | 40.9 | 3.72 | 3.8 | 9.5 | 89 | 11 |  |
| marker_4090 | P4_polym_carbonate_scanned/polymers-16-03364_scanned | main | yes | 11 | 11 | 46.0 | 4.18 | 3.8 | 8.9 | 86 | 6 |  |
| marker_4090 | P5_pccp_flory_huggins/d4cp03690e | main | no | 12 | 12 | 24.5 | 2.04 | 3.8 | 8.4 | 81 | 7 |  |
| marker_4090 | P5_pccp_flory_huggins/d4cp03690e1_suppl | si | no | 6 | 6 | 2.3 | 0.38 | 3.7 | 5.6 | 59 | 4 |  |
| mineru_advanced | P1_natcommun_lactone/s41467-024-52090-2 | main | no | 12 | 12 | 197.4 | 16.45 | 3.0 | 3.5 | 72 | 6 |  |
| mineru_advanced | P1_natcommun_lactone/s41467-024-52090-2_SI | si | no | 34 | 34 | 286.6 | 8.43 | 4.1 | 3.4 | 71 | 37 |  |
| mineru_advanced | P2_polym_star_raft/polymers-15-02002 | main | no | 14 | 14 | 93.7 | 6.69 | 3.0 | 3.3 | 70 | 12 |  |
| mineru_advanced | P2_polym_star_raft/polymers-2246784-supplementary | si | no | 6 | 6 | 52.0 | 8.66 | 2.1 | 3.3 | 70 | 8 |  |
| mineru_advanced | P3_polym_tmc_rop/polymers-15-00720 | main | no | 11 | 11 | 183.9 | 16.72 | 2.7 | 3.3 | 70 | 11 |  |
| mineru_advanced | P4_polym_carbonate_scanned/polymers-16-03364_scanned | main | yes | 11 | 11 | 199.1 | 18.10 | 2.6 | 3.3 | 70 | 7 |  |
| mineru_advanced | P5_pccp_flory_huggins/d4cp03690e | main | no | 12 | 12 | 136.6 | 11.38 | 3.0 | 3.6 | 71 | 7 |  |
| mineru_advanced | P5_pccp_flory_huggins/d4cp03690e1_suppl | si | no | 6 | 6 | 34.1 | 5.68 | 2.0 | 3.2 | 68 | 4 |  |
| mineru_basic | P1_natcommun_lactone/s41467-024-52090-2 | main | no | 12 | 12 | 164.4 | 13.70 | 3.4 | 3.3 | 63 | 10 |  |
| mineru_basic | P1_natcommun_lactone/s41467-024-52090-2_SI | si | no | 34 | 34 | 85.5 | 2.52 | 4.0 | 3.2 | 61 | 40 |  |
| mineru_basic | P2_polym_star_raft/polymers-15-02002 | main | no | 14 | 14 | 47.9 | 3.42 | 3.2 | 3.3 | 60 | 31 |  |
| mineru_basic | P2_polym_star_raft/polymers-2246784-supplementary | si | no | 6 | 6 | 28.8 | 4.80 | 1.9 | 2.8 | 60 | 15 |  |
| mineru_basic | P3_polym_tmc_rop/polymers-15-00720 | main | no | 11 | 11 | 77.6 | 7.05 | 2.9 | 3.2 | 62 | 16 |  |
| mineru_basic | P4_polym_carbonate_scanned/polymers-16-03364_scanned | main | yes | 11 | 11 | 69.8 | 6.35 | 3.0 | 3.4 | 61 | 7 |  |
| mineru_basic | P5_pccp_flory_huggins/d4cp03690e | main | no | 12 | 12 | 84.2 | 7.02 | 3.1 | 3.3 | 63 | 17 |  |
| mineru_basic | P5_pccp_flory_huggins/d4cp03690e1_suppl | si | no | 6 | 6 | 12.8 | 2.13 | 1.5 | 1.3 | 64 | 7 |  |
| mineru_standard | P1_natcommun_lactone/s41467-024-52090-2 | main | no | 12 | 12 | 142.1 | 11.84 | 2.7 | 3.5 | 70 | 10 |  |
| mineru_standard | P1_natcommun_lactone/s41467-024-52090-2_SI | si | no | 34 | 34 | 92.0 | 2.71 | 3.3 | 3.4 | 68 | 40 |  |
| mineru_standard | P2_polym_star_raft/polymers-15-02002 | main | no | 14 | 14 | 38.5 | 2.75 | 2.5 | 3.2 | 64 | 31 |  |
| mineru_standard | P2_polym_star_raft/polymers-2246784-supplementary | si | no | 6 | 6 | 31.8 | 5.30 | 2.0 | 3.3 | 66 | 15 |  |
| mineru_standard | P3_polym_tmc_rop/polymers-15-00720 | main | no | 11 | 11 | 63.1 | 5.74 | 2.3 | 3.3 | 68 | 16 |  |
| mineru_standard | P4_polym_carbonate_scanned/polymers-16-03364_scanned | main | yes | 11 | 11 | 157.5 | 14.32 | 2.6 | 3.3 | 68 | 7 |  |
| mineru_standard | P5_pccp_flory_huggins/d4cp03690e | main | no | 12 | 12 | 83.6 | 6.97 | 2.7 | 3.5 | 68 | 17 |  |
| mineru_standard | P5_pccp_flory_huggins/d4cp03690e1_suppl | si | no | 6 | 6 | 15.9 | 2.65 | 1.9 | 3.1 | 60 | 7 |  |

*watched-GPU util %*: mean utilisation of the GPUs used, including other users' jobs (shared machine) — a contention indicator, not this backend's own load.

## Document reconstruction quality (automatic, vs. PDF text layer)

| config | page count match | token recall | token precision | page mapping | reading order | headings | tables | table cells | equations display/inline | image refs | broken refs | symbol retention | missing docs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| marker | 8/8 | 0.971 | 0.974 | 1.000 | 0.961 | 106 | 17 | 1758 | 164/364 | 89 | 0 | 0.937 | 0 |
| marker_4090 | 8/8 | 0.970 | 0.968 | 1.000 | 0.960 | 107 | 18 | 1844 | 164/513 | 89 | 0 | 0.940 | 0 |
| mineru_advanced | 8/8 | 0.971 | 0.877 | 0.984 | 0.958 | 106 | 50 | 3824 | 26/856 | 92 | 0 | 0.959 | 0 |
| mineru_basic | 8/8 | 0.957 | 0.932 | 0.991 | 0.958 | 102 | 17 | 1613 | 27/761 | 143 | 0 | 0.731 | 0 |
| mineru_standard | 8/8 | 0.966 | 0.964 | 0.991 | 0.963 | 102 | 17 | 1547 | 27/796 | 143 | 0 | 0.946 | 0 |

### Per document

| config | document | pages | recall | precision | page map | order | tables | cells | eq d/i | image refs | symbols |
|---|---|---|---|---|---|---|---|---|---|---|---|
| marker | P1_natcommun_lactone/s41467-024-52090-2 | 12/12 | 0.959 | 0.983 | 1.000 | 0.945 | 1 | 160 | 1/185 | 6 (0 broken) | 0.936 |
| marker | P1_natcommun_lactone/s41467-024-52090-2_SI | 34/34 | 0.964 | 0.964 | 1.000 | 0.995 | 4 | 334 | 7/0 | 37 (0 broken) | 1.000 |
| marker | P2_polym_star_raft/polymers-15-02002 | 14/14 | 0.996 | 0.939 | 1.000 | 0.988 | 1 | 135 | 76/1 | 12 (0 broken) | 0.882 |
| marker | P2_polym_star_raft/polymers-2246784-supplementary | 6/6 | 0.964 | 0.903 | 1.000 | 0.933 | 4 | 549 | 0/5 | 6 (0 broken) | 0.759 |
| marker | P3_polym_tmc_rop/polymers-15-00720 | 11/11 | 0.994 | 0.991 | 1.000 | 0.988 | 2 | 139 | 59/8 | 11 (0 broken) | 0.957 |
| marker | P4_polym_carbonate_scanned/polymers-16-03364_scanned | 11/11 | 0.986 | 0.987 | 1.000 | 0.977 | 2 | 276 | 1/95 | 6 (0 broken) | 0.968 |
| marker | P5_pccp_flory_huggins/d4cp03690e | 12/12 | 0.960 | 0.981 | 1.000 | 0.907 | 2 | 153 | 20/70 | 7 (0 broken) | 1.000 |
| marker | P5_pccp_flory_huggins/d4cp03690e1_suppl | 6/6 | 0.622 | 1.000 | 1.000 | 0.955 | 1 | 12 | 0/0 | 4 (0 broken) | – |
| marker_4090 | P1_natcommun_lactone/s41467-024-52090-2 | 12/12 | 0.959 | 0.945 | 1.000 | 0.945 | 1 | 160 | 1/331 | 6 (0 broken) | 0.926 |
| marker_4090 | P1_natcommun_lactone/s41467-024-52090-2_SI | 34/34 | 0.964 | 0.964 | 1.000 | 0.995 | 4 | 343 | 7/0 | 37 (0 broken) | 1.000 |
| marker_4090 | P2_polym_star_raft/polymers-15-02002 | 14/14 | 0.996 | 0.957 | 1.000 | 0.988 | 1 | 135 | 76/1 | 12 (0 broken) | 0.906 |
| marker_4090 | P2_polym_star_raft/polymers-2246784-supplementary | 6/6 | 0.959 | 0.906 | 1.000 | 0.933 | 4 | 555 | 0/5 | 6 (0 broken) | 0.828 |
| marker_4090 | P3_polym_tmc_rop/polymers-15-00720 | 11/11 | 0.994 | 0.991 | 1.000 | 0.988 | 2 | 139 | 59/11 | 11 (0 broken) | 0.957 |
| marker_4090 | P4_polym_carbonate_scanned/polymers-16-03364_scanned | 11/11 | 0.985 | 0.987 | 1.000 | 0.973 | 3 | 340 | 1/93 | 6 (0 broken) | 0.968 |
| marker_4090 | P5_pccp_flory_huggins/d4cp03690e | 12/12 | 0.959 | 0.981 | 1.000 | 0.908 | 2 | 160 | 20/72 | 7 (0 broken) | 1.000 |
| marker_4090 | P5_pccp_flory_huggins/d4cp03690e1_suppl | 6/6 | 0.617 | 1.000 | 1.000 | 0.955 | 1 | 12 | 0/0 | 4 (0 broken) | – |
| mineru_advanced | P1_natcommun_lactone/s41467-024-52090-2 | 12/12 | 0.966 | 0.971 | 1.000 | 0.946 | 2 | 174 | 0/273 | 6 (0 broken) | 0.966 |
| mineru_advanced | P1_natcommun_lactone/s41467-024-52090-2_SI | 34/34 | 0.991 | 0.516 | 0.941 | 0.997 | 30 | 2374 | 7/79 | 37 (0 broken) | 0.927 |
| mineru_advanced | P2_polym_star_raft/polymers-15-02002 | 14/14 | 0.985 | 0.896 | 0.929 | 0.984 | 1 | 135 | 0/69 | 12 (0 broken) | 0.977 |
| mineru_advanced | P2_polym_star_raft/polymers-2246784-supplementary | 6/6 | 0.991 | 0.870 | 1.000 | 0.932 | 7 | 452 | 0/23 | 8 (0 broken) | 0.931 |
| mineru_advanced | P3_polym_tmc_rop/polymers-15-00720 | 11/11 | 0.981 | 0.981 | 1.000 | 0.987 | 3 | 154 | 0/108 | 11 (0 broken) | 0.940 |
| mineru_advanced | P4_polym_carbonate_scanned/polymers-16-03364_scanned | 11/11 | 0.978 | 0.976 | 1.000 | 0.973 | 3 | 220 | 0/186 | 7 (0 broken) | 0.976 |
| mineru_advanced | P5_pccp_flory_huggins/d4cp03690e | 12/12 | 0.958 | 0.945 | 1.000 | 0.891 | 3 | 243 | 19/118 | 7 (0 broken) | 1.000 |
| mineru_advanced | P5_pccp_flory_huggins/d4cp03690e1_suppl | 6/6 | 0.694 | 0.800 | 1.000 | 0.956 | 1 | 72 | 0/0 | 4 (0 broken) | – |
| mineru_basic | P1_natcommun_lactone/s41467-024-52090-2 | 12/12 | 0.947 | 0.925 | 1.000 | 0.937 | 1 | 160 | 0/335 | 10 (0 broken) | 0.642 |
| mineru_basic | P1_natcommun_lactone/s41467-024-52090-2_SI | 34/34 | 0.964 | 0.930 | 1.000 | 0.997 | 4 | 304 | 8/61 | 40 (0 broken) | 0.746 |
| mineru_basic | P2_polym_star_raft/polymers-15-02002 | 14/14 | 0.980 | 0.879 | 0.929 | 0.975 | 1 | 154 | 0/37 | 31 (0 broken) | 0.788 |
| mineru_basic | P2_polym_star_raft/polymers-2246784-supplementary | 6/6 | 0.987 | 0.978 | 1.000 | 0.903 | 4 | 430 | 0/8 | 15 (0 broken) | 0.862 |
| mineru_basic | P3_polym_tmc_rop/polymers-15-00720 | 11/11 | 0.960 | 0.934 | 1.000 | 0.985 | 3 | 154 | 0/90 | 16 (0 broken) | 0.556 |
| mineru_basic | P4_polym_carbonate_scanned/polymers-16-03364_scanned | 11/11 | 0.971 | 0.982 | 1.000 | 0.967 | 2 | 276 | 0/111 | 7 (0 broken) | 0.960 |
| mineru_basic | P5_pccp_flory_huggins/d4cp03690e | 12/12 | 0.952 | 0.947 | 1.000 | 0.901 | 2 | 135 | 19/119 | 17 (0 broken) | 1.000 |
| mineru_basic | P5_pccp_flory_huggins/d4cp03690e1_suppl | 6/6 | 0.592 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0/0 | 7 (0 broken) | – |
| mineru_standard | P1_natcommun_lactone/s41467-024-52090-2 | 12/12 | 0.962 | 0.985 | 1.000 | 0.944 | 1 | 160 | 0/263 | 10 (0 broken) | 0.971 |
| mineru_standard | P1_natcommun_lactone/s41467-024-52090-2_SI | 34/34 | 0.971 | 0.967 | 1.000 | 0.999 | 4 | 304 | 8/123 | 40 (0 broken) | 0.909 |
| mineru_standard | P2_polym_star_raft/polymers-15-02002 | 14/14 | 0.984 | 0.890 | 0.929 | 0.976 | 1 | 154 | 0/21 | 31 (0 broken) | 0.977 |
| mineru_standard | P2_polym_star_raft/polymers-2246784-supplementary | 6/6 | 0.996 | 0.996 | 1.000 | 0.923 | 4 | 426 | 0/9 | 15 (0 broken) | 1.000 |
| mineru_standard | P3_polym_tmc_rop/polymers-15-00720 | 11/11 | 0.977 | 0.987 | 1.000 | 0.987 | 3 | 154 | 0/95 | 16 (0 broken) | 0.855 |
| mineru_standard | P4_polym_carbonate_scanned/polymers-16-03364_scanned | 11/11 | 0.975 | 0.989 | 1.000 | 0.975 | 2 | 196 | 0/183 | 7 (0 broken) | 0.976 |
| mineru_standard | P5_pccp_flory_huggins/d4cp03690e | 12/12 | 0.956 | 0.968 | 1.000 | 0.901 | 2 | 153 | 19/102 | 17 (0 broken) | 1.000 |
| mineru_standard | P5_pccp_flory_huggins/d4cp03690e1_suppl | 6/6 | 0.594 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0/0 | 7 (0 broken) | – |

Notes: the reference is the PDF text layer (PyMuPDF), which keeps running headers/footers and flattens tables, so recall < 1 and precision < 1 are expected for every backend. For the scanned case the reference is the original (born-digital) PDF's text layer. *Symbol retention* pools polymer notation (Đ, Mn, Mw, Tg, Tm, °C, g/mol, mol%, -co-, -b-, ×10, ±, Sn(Oct)2, Greek letters) after flattening LaTeX (e.g. `M_{n}` → Mn) and HTML sub/superscripts. Table cell accuracy is not scored automatically — see `reports/tables/<case>/<doc>.md` for side-by-side table dumps.

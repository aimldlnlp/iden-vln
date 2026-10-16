# Rekap eksperimen IDEN-VLN — 14 Oktober2026

## Charter
Rekap eksperimen navigasi utama yang terdokumentasi sampai ablation counter STOP. Ini inventaris keluarga eksperimen dan hasil, bukan hitungan setiap invocation/job. Audit data/geometri, training offline, oracle dan rescoring dipisahkan dari autonomous learned SR. Tidak menjalankan eksperimen baru.

## Canonical sources
Sumber utama: stage_metric_audit_20260928/{stage_metrics.json,cgt_audit.json}; balanced100_eval_20261013_v1/verified_results.json; balanced100_matched_contract_20261013_v1/summary.json; corrective_v3_comparison_20261010.json; final_verification/local_verification pilot CLIP8/counter. Semua berada di remote_ordered_eval. Laporan originating di paper_ieee_access_idenvln_final/docs menjadi sumber tambahan. Graphify lama hanya peta.

## Findings ledger
SR: seluruh rantai selesai. ISR: keberhasilan tahap; balanced100 memakai macro per task, dengan micro dilaporkan bila tersedia. CSR/CGT di sini adaptasi lokal. OSR mengabaikan kebutuhan STOP dalam pengukuran kunjungan; SPL menggabungkan success dan efisiensi rute; NE jarak error meter. Mean teacher-forced accuracy bukan SR. Tanda — berarti tidak dilaporkan di sumber yang dirangkum, bukan nol.

### A. Canonical historis / Tabel4
100task per policy, seluruhnya2tahap, AABB, horizon150, strict STOP<1m. SR,OSR,SPL seluruh baris0,00%.

| Policy | ISR (%) | CSR lokal (%) | CGT lokal (%) |
|---|---:|---:|---:|
| Random | 3,50 | 2,25 | 1,33 |
| Frame BC balanced | 2,00 | 1,50 | 0,60 |
| Target BC mild | 0,50 | 0,25 | 0,17 |
| Target BC strong | 0,50 | 0,50 | 0,02 |
| Target BC calibrated | 0,00 | 0,00 | 0,00 |
| Sequence BC mild | 3,50 | 2,00 | 1,47 |
| Sequence BC strong | 2,50 | 1,50 | 0,97 |
| Sequence BC calibrated | 0,50 | 0,50 | 0,52 |
| Sequence hybrid | 0,50 | 0,50 | 0,52 |
| DAgger-lite | 3,50 | 3,25 | 1,98 |

CGT memakai panjang acuan geodesik per tahap yang direkonstruksi; bukan panjang lintasan agent. Semua100reference sums cocok. NE canonical DAgger-lite9,47m sebagaimana main.tex. Audit tahap: docs/table4-stage-metrics-audit-20260928.md.

### B. Balanced100, sepuluh konfigurasi historis
25task per level2–5;350tahap;73scene;1000rollout. AABB/horizon150/strict<1m. SR dan SPL seluruh policy0,00%. OSR0,00%kecuali DAgger-lite1,00%. NE bersyarat pada99reference-valid episodes; satu invalid reference tetap kegagalan pada denominator100.

| Policy | ISR macro (%) | ISR micro (%) | CSR lokal (%) | NE (m) |
|---|---:|---:|---:|---:|
| Random | 3,68 | 3,14 | 2,48 | 11,42 |
| Frame BC balanced | 0,95 | 0,86 | 0,60 | 11,40 |
| Target BC mild | 0,75 | 0,57 | 0,50 | 11,72 |
| Target BC strong | 0,25 | 0,29 | 0,25 | 11,43 |
| Target BC calibrated | 0,00 | 0,00 | 0,00 | 12,12 |
| Sequence BC mild | 2,62 | 2,00 | 1,77 | 11,51 |
| Sequence BC strong | 1,98 | 1,71 | 1,38 | 11,70 |
| Sequence BC calibrated | 0,58 | 0,57 | 0,58 | 12,25 |
| Sequence hybrid | 0,00 | 0,00 | 0,00 | 12,11 |
| DAgger-lite | 1,98 | 1,71 | 1,22 | 11,63 |

CGT tidak dilaporkan dalam verified_results balanced100 ini. ISR macro/micro berbeda karena K2–5. Bank balanced100 berbeda dari canonical100 dua tahap.

### C. Kontrol matched: pipeline lama versus repair
Data4train/4dev,398aksi train,10STOP; GRU, window, seed, sampling, dan budget4000update sama. Perbedaan alignment observasi–aksi+yaw diuji sebagai paket. Kriteria checkpoint final sama; bukan retraining300trajectory historis.

| Cohort | Arm | Chain | Stage sukses | SR (%) | ISR macro (%) | CSR lokal (%) |
|---|---|---:|---:|---:|---:|---:|
| train | historical_contract | 0/4 | 0/10 | 0,00 | 0,00 | 0,00 |
| train | repaired_contract | 4/4 | 10/10 | 100,00 | 100,00 | 100,00 |
| train | random | 0/4 | 1/10 | 0,00 | 8,33 | 4,17 |
| dev | historical_contract | 0/4 | 0/10 | 0,00 | 0,00 | 0,00 |
| dev | repaired_contract | 0/4 | 0/10 | 0,00 | 0,00 | 0,00 |
| dev | random | 0/4 | 0/10 | 0,00 | 0,00 | 0,00 |
| balanced100 | historical_contract | 0/100 | 2/350 | 0,00 | 0,58 | 0,58 |
| balanced100 | repaired_contract | 0/100 | 12/350 | 0,00 | 3,65 | 3,30 |
| balanced100 | random | 0/100 | 11/350 | 0,00 | 3,68 | 2,48 |

ISR micro balanced100: repaired12/350 =3,43%, random11/350 =3,14%, historical2/350 =0,57%. Repaired macro3,65% tidak lebih tinggi daripada random3,68%; CSR lebih tinggi pada satu seed belum membuktikan superiority.

### D. Pilot IL / CLIP dan corrective pada train
Pilot memakai OBB atau approach goals, umumnya horizon400 dan dua STOP. Baris di bawah memiliki cohort berbeda; bukan leaderboard tunggal.

| Eksperimen | Hasil SR train | Hasil SR dev | Catatan |
|---|---:|---:|---|
| Stats/hash versus spatialCLIP, original/revised4task | 4/4 =100,00%setiap arm | 0/1 =0,00% | Kontrol empty/swapped berbeda; docs/spatial-clip-navigation-pilot-20261001.md |
| SpatialCLIP paired8 | 8/8 =100,00% | 0/2 =0,00% | Empty text0/8; swapped paired scored tujuan teks2/2 |
| Tambahan recovery lama | 8/8 =100,00% | 0/2 =0,00% | Dev belum membaik; docs/spatial-clip-recovery-il-20261010.md |
| Tambahan scene,10task | 10/10 =100,00% | 0/2 =0,00% | Perluasan coverage, bukan generalisasi terbukti |
| Combined corrective / v1 | 9/9 =100,00% | 0/2 =0,00% | Normal original starts |
| Early corrective v2 | 9/9 =100,00% | 0/2 =0,00% | Normal original starts |
| Learner corrective v3 | 9/9 =100,00% | 0/2 =0,00% | Transfer canonical tetap0 |
| Approach-v2 train4 early-stop | 4/4 =100,00% | accepted0/1 =0,00% | ISRtrain100,00%; full dev2dengan1held |
| Approach bank4 versus bank8, matched1200 | 4/4 dan8/8 =100,00% | masing-masing0/2 =0,00% | ISRtrain100,00%; accepteddev2dari5 |

#### Corrective pada tujuh perturbasi train yang sama
Teacher prefix dan forced perturbation diberikan; ini diagnostic train robustness, bukan unseen-scene autonomous benchmark.

| Model | Chain sukses / SR | Stage sukses / ISR |
|---|---:|---:|
| Tanpa late corrections | 4/7 =57,14% | 9/14 =64,29% |
| v1 | 4/7 =57,14% | 8/14 =57,14% |
| v2 | 4/7 =57,14% | 9/14 =64,29% |
| v3 | 6/7 =85,71% | 12/14 =85,71% |

Checkpoint v3 diuji pada canonical100 AABB/horizon150: SR0/100, ISR0/200, CSR/CGT/OCS/OSR/SPL0,00%; NE10,33m. Sumber: docs/spatial-clip-v3-canonical100-aabb-20261011.md.

### E. Human-reviewed recovery, budget dan LR schedule
Approach-v2,8train,8development perturbations train,2accepteddev, horizon400. BC original-only versus BC+697recovery; kontrak ini berbeda dari correctiveOBBdi atas. Setiap sel berikut SR/ISR(%).

| Kondisi | Original BC | Original BC+recovery | Perturbasi BC | Perturbasi BC+recovery | Dev kedua arm |
|---|---:|---:|---:|---:|---:|
| 1200update | 25,00 /56,25 | 0,00 /43,75 | 12,50 /31,25 | 0,00 /18,75 | 0,00 /0,00 |
| 3600update, LRtetap0,0012 | 12,50 /43,75 | 0,00 /0,00 | 0,00 /0,00 | 0,00 /0,00 | 0,00 /0,00 |
| 3600update, LRturun setelah1200 | 25,00 /56,25 | 0,00 /43,75 | 0,00 /6,25 | 12,50 /31,25 | 0,00 /0,00 |

LR drop memulihkan fitting; recovery membantu satu rantai perturbasi tetapi belum menyelesaikan original chain. Sumber: docs/human-clip8-recovery-pair-20261014.md, human-clip8-recovery-budget3600-20261014.md, human-clip8-recovery-lr-drop-20261014.md.

### F. Counter STOP terbaru
Kedua arm memakai BC+recovery2054samples; kanal tambahan berisi0 versus jumlah STOP sendiri. Initializer/draws sama,3600updates dengan LR drop. SR semuanya0,00%.

| Cohort | ISR counter_zero (%) | ISR counter_active (%) |
|---|---:|---:|
| Originaltrain8 | 43,75 | 37,50 |
| Perturbasitrain8 | 12,50 | 0,00 |
| Accepteddev2 | 0,00 | 0,00 |

Pada original, immediate invalid STOP tahap1 terjadi7/7 entered pada zero dan6/6 entered pada active; valid STOP tahap1 tetap0 pada keduanya. Counter active kehilangan satu keberhasilan tahap0. Seluruh36 replay dan verifier lokal passed; docs/human-clip8-stop-counter-20261014.md.

### G. Diagnostik gerak, STOP dan representasi awal

| Eksperimen | Hasil chain / SR | Status informasi |
|---|---:|---|
| Learnedmovement+learnedSTOP,dev30 | 0/30 =0,00% | Baseline learned |
| Tambah gold active-targetindex,dev30 | 0/30 =0,00% | Diagnostic goldfeature; firstsubgoal2/30menjadi0/30 |
| Learnedmovement+oracleSTOP,dev30 | 2/30 =6,67% | STOPoracle, bukan autonomousSR |
| Oraclemovement+learnedSTOP,dev30 | 0/30 =0,00% | Gerakoracle; firstsubgoal0/30 |
| Oraclemovement+oracleSTOP,dev30 | 24/30 =80,00% | Privileged feasibility |
| RGBD STOPverifier+learnedmovement,dev30 | 0/30 =0,00% | Firstsubgoal1/30 |
| RGBD STOPverifier+oraclemovement,dev30 | 0/30 =0,00% | Firstsubgoal0/30 |
| Movementbaseline train20+oracleSTOP | 9/20 =45,00% | Firsttarget15/20; tidaksetara autonomousSR |
| Fixedright/depthstuckrecovery train20+oracleSTOP | 9/20 =45,00% | Tidak menaikkan chain |
| SpatialRGB versusRGBD3x3 train20+oracleSTOP | keduanya0/20 =0,00% | Firsttarget2/20 |
| RGBD+instruction train60,evaluate20+oracleSTOP | 0/20 =0,00% | Firsttarget2/20 |
| RGBD8x8 train20+oracleSTOP | 1/20 =5,00% | Firsttarget2/20 |

Sumber: oracle-navigation-dev30, rgbd-stop-verifier-dev30, navigation-failure-audit, rgbd-navigation-matched-ablation, rgbd-instruction-train60, spatial8-navigation-pilot; seluruhnya pada docs. Metrik first-target success tidak otomatis menjadi ISR karena denominator chain/stage berbeda.

### H. Sensitivitas evaluator dan single-goal

- Rescore canonical100 pada endpoint <3m: DAgger-lite dan random masing-masing15/100 =15,00%. Dengan dua ordered STOP <3m, keduanya6/100 =6,00%. Target, radius, dan aturan berbeda; ini tidak mengganti SR canonical0. Sumber: docs/vln-protocol-sensitivity-20260929.md.
- Single-goal v2 transfer100: Sequence mild STOP success2/100 =2,00%, final-position success3/100 =3,00%; DAgger-lite STOP success2/100 =2,00%, final-position success4/100 =4,00%. Mean nDTW masing-masing0,12 dan0,14; sDTW0,01 keduanya. Policy lain memiliki STOP success0. Sumber: docs/single-goal-v2-transfer-100-20260930.md.
- Single-goal live pilot4task: stationary STOP dan random masing-masing1/4 =25,00%; tiered BC dan target BC masing-masing0/4. Beberapa start sudah dekat goal; pilot ini bukan long-horizon SR.
- Single-goal Sequence BC offline: accuracy train/dev/internal-test88,61/54,02/56,25%; STOP precision48,62/5,88/6,25%. Ini klasifikasi offline, bukan rollout SR. Sumber: docs/single-goal-sequence-baseline-20260930.md.

### I. Controller feasibility di paper dan teacher terbaru

| Controller / sumber target | Chain / SR | OSR (%) | NE (m) |
|---|---:|---:|---:|
| Gold PathFinder | 87/100 =87,00% |87,00|1,29|
| Headless gold | 93/100 =93,00% |93,00|1,17|
| Headless XLM-R target | 61/100 =61,00% |61,00|1,19|
| Rendered gold | 87/100 =87,00% |87,00|1,29|
| Rendered Qwen predicted target | 13/100 =13,00% |13,00|0,85|
| Rendered XLM-R predicted target | 65/100 =65,00% |65,00|0,86|
| Actual teacher balanced100 | 56/100 =56,00% |—|—|

Controller memakai target gold/predicted dan geometry/follower; bukan learned action policy. NE paper bersyarat pada jumlah valid records yang berbeda. Teacher balanced100 menyelesaikan244/350 tahap =69,71% micro;44task kehabisan horizon. Sumber: generated/tables_main_vln_feasibility.tex dan docs/balanced100-teacher-contract-audit-20261013.md.

### J. Audit kesiapan data, bukan SR policy

- Historical expansion:22/25 replay numerik valid,0/25 diterima penuh pada audit RGB awal.
- Audit approach resolver, banister, mesh, pitch, dan visibility menghasilkan bank reviewed4 lalu8task. Construction/replay acceptance tidak dilaporkan sebagai learned SR.
- Review manusia mencakup14 terminal recovery STOP views dari7task oleh Zidan; bukan validasi seluruh rute/dataset.
- Batch release2500 dihentikan untuk balanced100; tidak ada hasil learned full2500 yang diklaim di sini.

## Scope filter

Canonical dua tahap AABB/horizon150, balanced100 K2–5 AABB/horizon150, OBB corrective/approach-v2 horizon400, endpoint3m rescoring, dan oracle controllers memiliki kontrak berbeda. Tabel disusun untuk catatan, bukan ranking lintas kontrak. CSR/CGT/NE yang tidak dilaporkan tidak diisi dengan nol.

## Object filter

SR train100,00% dan perturbasi train85,71% tidak membuktikan generalisasi. Accepted dev2/5 atau1/2 dicatat eksplisit. Satu seed, cohort kecil, teacher states yang ditinjau, dan exposure membatasi kesimpulan. Random bukan learned policy; oracle feasibility bukan autonomous policy. Offline accuracy bukan SR.

## Conclusions

1. Ada learned SR nonzero, terutama pada imitasi train dan perturbasi train (D–F).
2. Pada canonical dan balanced100, full-chain learned SR tetap0,00%; repaired ISR3,65% belum melampaui random macro3,68% (A–C).
3. Dev pilot yang dirangkum tetap0,00%; belum ada bukti generalisasi kuat (C–F).
4. Recovery, LR schedule, dan counter tidak memberikan perbaikan monoton; counter terakhir tidak membantu (E–F).
5. SR nonzero pada oracle/rescoring tidak boleh mengganti headline learned score (G–I).

## Next steps / Risks

Prioritas saat ini: audit konsistensi fitur dan label STOP/non-STOP pada boundary, kemudian rancang satu ablation objective/sampling dengan kontrol tetap. Belum dilakukan multi-seed, random approach-v2 pada kontrak yang sama, atau evaluasi generalisasi final baru. Rekap berdasarkan artefak/laporan lokal; bukan audit setiap arsip run atau status SSH terbaru. Tidak ada training baru dalam rekap ini.

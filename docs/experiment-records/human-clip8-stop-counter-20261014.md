# CLIP8 + recovery: ablation counter STOP, hasil final

14 Oktober2026. Pipeline selesai exit0; comparison verified;36fresh actor replay passed; archive diunduh dan verifier lokal passed. Tidak ada training atau rollout tambahan pada kelanjutan ini.

## Charter

Pertanyaan: apakah counter STOP yang berasal dari aksi agen sendiri memperbaiki transisi tahap kedua tanpa mengurangi kemampuan tahap pertama?

Jenis: ablation development satu seed, dua arm matched. Scope: pilot visible_approach_framed_v2, bukan balanced100 atau Tabel4. Kelanjutan dari handover menutup final verification, archive/download, laporan, dan menentukan diagnosis berikutnya.

Sumber bukti diprioritaskan: kontrak beku dan kode; trace/checkpoint; verifier server/lokal; diagnosis CPU; laporan dan handover sebagai konteks. Graphify414nodes dipakai sebagai peta lama, bukan sumber angka terbaru.

## Canonical sources

Root lokal: `D:/Kuliah/Semester 7/Proskrip/Template_Skripsi_latex/Template_Skripsi/remote_ordered_eval/approach_clip8_stop_counter_20261014_v1/`.

Root remote: `/home/aimldl/LH-VLN/runs/approach_clip8_stop_counter_20261014_v1/`.

- `contract_frozen.json`, `counter_self_check.json`, `experiment/protocol_frozen.json`.
- `experiment/training_summary.json`, `experiment/fitting_verification.json`.
- `experiment/evaluation_rollouts.json`, `experiment/evaluation_summary.json`.
- `experiment/independent_evaluation_verification.json`, `experiment/final_verification.json`.
- `experiment/diagnosis/diagnostic_results.json`, `counter_comparison.json`.
- `pipeline_status.json`, `exit`, `snapshot.tgz`, `local_verification.json`.
- Audit lanjutan lokal: `remote_ordered_eval/counter_boundary_review_20261014_v1/boundary_evidence.json`; script `remote_ordered_eval/audit_counter_boundary_evidence.py`.

Konteks: `docs/human-clip8-recovery-lr-drop-20261014.md` dan `docs/handover-stop-counter-ablation-20261014.md` pada folder paper. Snapshot status pending dalam handover kini sudah ditutup oleh bukti final; handover lama dipertahankan sebagai catatan historis.

## Findings ledger

| ID | Temuan | Bukti / batas |
|---|---|---|
| F1 | Pipeline phasecompleted/verifiedtrue, exit0,36rollout selesai; session tmux counter sudah berakhir. | Status remote diperiksa14Oktober19.38WITA. Session lain tidak dihentikan. |
| F2 | Kedua arm BC+recovery memakai2054window identik:1357original+697recovery,8unique train tasks,7task dengan recovery tambahan. | Kontrak, training summary, fitting verification. counter_zero bukan original-only BC. |
| F3 | Inisialisasi dan draws matched; counter adalah satu-satunya perbedaan fitur kedua arm baru. | Shared initializer/draw hashes, self-check dan fitting verifier. Control6768tidak dianggap reproduksi bitwise model6767lama. |
| F4 | Counter aktif tidak meningkatkan SR pada cohort mana pun; stage success turun pada original dan perturbasi. | Final36trace dan metrics dihitung ulang lokal. |
| F5 | Pada original, stage1validSTOP0 pada kedua arm; semua task yang memasuki stage1 langsung STOP invalid:7/7kontrol,6/6aktif. | counter_comparison.stage_transition_metrics. Penurunan7menjadi6bukan perbaikan boundary. |
| F6 | Original index3 kehilangan stage0success pada counter aktif. | Paired final outcomes; aktif menghabiskan horizon tanpa STOP. |
| F7 | Common-set original STOPFP13dan recoverySTOPFP9pada kedua arm; STOPFN0. | diagnosis.common_set_prediction_metrics. Ini relatif label teacher, bukan geodesic precision. |
| F8 | Pada4/8original tiap arm, first action divergence tepat pada teacher boundary dan berupa STOP invalid; prediksi cache jugaSTOP. | Audit boundary seluruh8task×2arm, cached probabilities, pose/action prefix, pre_geo_m. |
| F9 | Empat original lain tiap arm mengalami divergence lebih awal; seluruh16first-divergence top1cache cocok dengan aksi live dan prefix position error0. | boundary_evidence.json dan diagnosis. Tidak membuktikan seluruh live feature numerically identical. |
| F10 | Archive hash cocok; extract menolak traversal, link/nonregular member, dan overwrite file berbeda; local verifier passed. |40archive members; local_verification.json. |

### Kontrak yang dipertahankan

- Frozen CLIP ViT-B/16 spatial2×2 left/front/right dan text vector; GRU128/window16; parent6767features tetap, counter appended scalar menjadi6768.
- Counter adalah jumlahSTOPyang sudah dieksekusi sebelum frame, termasuk STOP invalid. Initial observation STOP tidak dimasukkan sebagai aksi dinilai. Tidak membaca successes, sim.stage, semantic mask, jarak goal, atau NavMesh untuk fitur counter.
- Padding counter0; nilai bertahan setelah STOP keluar dari window. Kontrol memberi0pada kanal tambahan. Normalisasi parent tetap; counter mean0/std1.
- Seed20260930; uniform sampling with replacement; AdamW weight decay0,0001; batch128; gradient clip5.
-3600updates: LR0,0012hingga1200, lalu0,00012. Loss weights berasal dari distribusi aksi original bank. Final checkpoint3600saja; tidak memilih dari dev.
- Dua ordered actual actor STOP, geodesik strict<1m terhadap frozen approach goals, horizon400termasuk prefix/forced actions. Heading tidak menjadi syarat scoring; STOP invalid tetap memajukan tahap sesuai kontrak.
- Kasus sama per arm:8original train,8perturbasi train yang sudah pernah diperiksa,2accepted dev. Total36rollout.

### Hasil final

| Cohort | Arm | Chain sukses | Stage sukses | SR (%) | ISR (%) |
|---|---|---:|---:|---:|---:|
| Original train | counter_zero |0/8|7/16|0,00|43,75|
| Original train | counter_active |0/8|6/16|0,00|37,50|
| Perturbasi train | counter_zero |0/8|2/16|0,00|12,50|
| Perturbasi train | counter_active |0/8|0/16|0,00|0,00|
| Accepted dev | counter_zero |0/2|0/4|0,00|0,00|
| Accepted dev | counter_active |0/2|0/4|0,00|0,00|

Penurunan ISR active:6,25percentage points pada original dan12,50pada perturbasi. Ini perubahan cohort satu seed, bukan estimasi efek populasi.

### Denominator transisi tahap

| Cohort | Arm | Stage0 STOP valid | Stage1 entered | Stage1 STOP valid | Immediate invalid STOP / entered |
|---|---|---:|---:|---:|---:|
| Original | zero |7/8|7/8|0|7/7|
| Original | active |6/8|6/8|0|6/6|
| Perturbasi | zero |2/8|6/8|0|4/6|
| Perturbasi | active |0/8|6/8|0|2/6|
| Accepted dev | zero |0/2|2/2|0|1/2|
| Accepted dev | active |0/2|2/2|0|1/2|

Jumlah immediate STOP yang lebih kecil pada perturbasi tidak menghasilkan keberhasilan tahap kedua. STOP yang gagal juga dapat membuka tahap1; stage1entered tidak berarti stage0berhasil.

### Audit boundary setelah download

Script lokal read-only merekonstruksi first divergence pada seluruh original teacher traces dan memastikan source hashes tidak berubah. Tidak ada pemilihan kasus berdasarkan keberhasilan dan tidak ada inference/model fitting baru.

Kasus first divergence tepat pada boundary:

| Arm | Indeks original | Prediksi cacheSTOP pada kasus tersebut |
|---|---|---|
| zero |1,3,6,7|Seluruhnya sama dengan aksi liveSTOPinvalid|
| active |1,5,6,7|Seluruhnya sama dengan aksi liveSTOPinvalid|

Pada kasus tersebut, prefix aksi sebelum boundary masih sama dengan teacher, pose prefix cocok, ownSTOPcount1, dan fitur counter active1/control0. Probabilitas cacheSTOP berkisar82,08–96,32%pada zero dan69,29–95,94%pada active. Actor menunjukkan jarak target tahap kedua>=1m; STOPtersebut memang invalid menurut kontrak, bukan sekadar alternatif aksi berbeda dari teacher.

Ini mendukung prioritas audit teacher-state boundary fitting. Tidak meniadakan distribution shift setelah divergence, dan belum membuktikan mekanisme objective, alias fitur, atau teacher-search history.

## Scope filter

- Hasil counter dibandingkan terutama zero versus active yang matched; skor archivedLRdrop6767hanya konteks.
- Tidak melaporkan kemenangan terhadap random atau generalisasi. Belum ada random pada kontrak approach-v2yang sama.
- Accepted dev2bukan full dev5;3task held tetap dicatat. Perturbasi memakai teacher prefix pada train scenes, bukan unseen-scene autonomous SR.
- Teacher-labelSTOPFP dan recall100% tidak disamakan dengan geodesicSTOPprecision.
- Tidak menyimpulkan counter tidak pernah berguna; bukti hanya menolak manfaat pada konfigurasi/cohort/seed ini.

## Object filter

- Review manusia oleh Zidan mencakup14terminal recoverySTOPviews dari7task, bukan seluruh rute/dataset. Dua export reviewer sama tidak menjadi dua reviewer independen.
- Counter menghitung aksi, bukan keberhasilan. Ia tidak menghapus kesalahan semantik target atau memberi kemampuan parsing instruksi aktif secara otomatis.
- Turunnya immediateSTOPcount tidak dianggap improvement jika stage0retention turun atau stage1valid tetap0.
- Cache/live top1agreement dan poseprefix0lebih sempit daripada verifikasi seluruh fitur live. Penyebab numerik/informasi perlu diperiksa sebelum ablation baru.
- Historical balanced100 memakai AABB/horizon150/K2–5dan kontrak berbeda. Hasilnya tidak diganti dengan counter pilot; paper/Tabel4tidak diubah.

## Conclusions

1. Final pipeline dan pencatatan hasil counter telah ditutup, termasuk verifikasi lokal (F1,F10).
2. Counter active tidak memperbaiki transisi tahap kedua dan mengurangi stage success pada dua train cohorts yang diperiksa (F4–F6). Tidak menjadi kandidat untuk diperluas sekarang.
3. STOPfalsepositives tetap22pada common teacher-label set setiap arm, dengan FN0; fitting agregat tinggi menutupi kesalahan lokal penting (F7).
4. Sebagian boundary failures juga diprediksi pada cached teacher states sebelum action-prefix drift. Memperbesar recovery bank saja belum menargetkan temuan ini secara spesifik (F8–F9).
5. Next step paling konkret adalah audit boundary data/feature consistency dan fitting objective sebelum training baru. Counter tambahan atau larangan STOPberurutan belum didukung (F4–F9).

## Next steps / Risks

Urutan audit berikutnya, tanpa mengubah scoring:

1. Gunakan semua8teacher boundaries dan seluruh2054windows. Periksa causal/cache/live feature consistency pada boundary, serta exactduplicate input dengan STOP/nonSTOPlabels yang konflik. Jangan mensyaratkan100%single-labelaccuracy tanpa mengecek aksi alternatif yang valid.
2. Ukur error/loss/margin pada boundary versus terminalSTOPdan navigation samples; gunakan fixed set, classweights dan normalisasi yang sama. Teacher-labelFPberbeda dari geodesicinvalid STOP; keduanya harus dicatat.
3. Bila tidak ada konflik/inconsistency yang menjelaskan error, rancang satu ablation objective/sampling boundary dengan kontrak baru. Pertahankan data, budget, initializer, LR, evaluator dan task sama. Jangan sekaligus mengubah teacher history, representasi, dan bobot STOP.
4. Teacher-search history tetap hipotesis alternatif. Bila diuji, demonstrasi baru wajib menggunakan aksi simulator sah dan direplay; jangan menghapus aksi dari cache untuk membuat hasil cocok.
5. Sebelum klaim manfaat populasi: seed berulang, dev baru/scene-disjoint yang belum dipakai memilih model, serta random pada kontrak yang sama.

Belum dilakukan: exactwindowlabel-conflict audit, numerical boundary livefeature equality baru, ablation objective/sampling/searchhistory, seed tambahan, random approach-v2, atau perubahan paper. Tidak ada job counter milik pekerjaan ini yang sengaja ditinggalkan berjalan.

### Identitas akhir

```text
Contract: 24bdfa4d883bf05f0f7bd163a08e9263c8e9cc07d4c3bfde15cc28f05ced1c6b
Shared initializer: bf0481bafe3d0d3a1f1b57beca54f5f2a3acfcefed9cf107a91db6798dd58b53
Uniform draws: f42978f04abee6c42cab1a079e878fd1f7c8a013e1cc8d2addefab5a4e6dad66
counter_zero checkpoint: 2310daf29199fc8bb8649b99fd93bf8ff3d4c33ece7b2baf424bf05a10ef2196
counter_active checkpoint: 516da15a7b7a3af021f68aa4f02686ae7537363da84695e63cfc58f2c1678ded
Snapshot local/remote: e0ef6578b4d32115caf5a2e18e2cd050eb7db4dfcce9c5793e1e900121b7143d
```

Review internal: angka sesuai comparison/trace/verifier; perubahan kontrol dibanding armbaru yang matched; klaim kausal mekanisme dan generalisasi dibatasi. Hasil negatif dipertahankan.

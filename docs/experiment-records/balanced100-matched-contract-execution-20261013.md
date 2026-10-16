# Matched contract balanced100 — catatan eksekusi

13 Oktober2026. Training selesai; evaluasi balanced100 berjalan saat catatan ini ditulis. Hasil final belum tersedia.

## Charter

Mengukur pengaruh paket alignment observasi–aksi dan yaw runtime. Arm historical memakai frame/pose sesudah aksi untuk label aksi tersebut dan unsigned yaw saat runtime. Arm repaired memakai observasi sebelum aksi dan signed yaw.

Kedua arm memakai empat task train,398aksi,10STOP, empat dev terpisah scene, GRU128/window8/input175, seed57007, batch128,4000update, AdamW lr0,0012, weight decay0,0001, class-weight power0,45 dan STOPfactor0,65. Checkpoint final dipakai untuk keduanya; tidak ada early stopping atau seleksi menggunakan dev.

## Canonical sources

Remote: `/home/aimldl/LH-VLN/runs/balanced100_matched_contract_20261013_v1/`.

- `protocol_frozen.json`: cohort, kontrak, source hashes dan budget.
- `training/data_verification.json`, `training/matched_verification.json`.
- `evaluation/{train,dev,balanced100}/{historical_contract,repaired_contract,random}/`.
- Saat seluruh evaluasi selesai: `independent_verification.json`, `artifact_hashes.json`, `result_report.md`.

Skrip lokal: `remote_ordered_eval/run_balanced100_matched_contract.py`, `launch_balanced100_matched_contract.sh`, `verify_balanced100_matched_contract.py`, `finish_balanced100_matched_contract.sh`.

## Findings ledger

| ID | Temuan yang sudah selesai | Bukti |
|---|---|---|
| F1 | Seluruh inisialisasi parameter dan urutan sampling4000update identik antara arm. | Hash training/matched_verification.json |
| F2 | Kedua arm offline train accuracy100,00%, recallSTOP100,00%. | Final training curve update4000 |
| F3 | Rollout train historical0/4chain dan0/10stage; repaired4/4chain dan10/10stage. | evaluation/train/*/metrics.json dan trace |
| F4 | Rollout dev historical0/4chain dan0/10stage; repaired0/4chain dan0/10stage. | evaluation/dev/*/metrics.json dan trace |
| F5 | Balanced100 telah dimulai; tidak ada hasil final yang diklaim di catatan ini. | status.json dan per-episode trace; progres berubah setelah catatan dibuat |

Inisialisasi SHA256: `dac9e65d521aa300dcc16f59f3646909deb180d8cc9a1eb74c4867584f3b4ffa`.

Urutan draws SHA256: `51c600976ee1203d4935ebb1c1898a242230e8bb4b3769fa5cf51efad4a8e944`.

Checkpoint historical SHA256: `38f8e9b3c956d8f0a2a6fcdf89722fbe3981472061ccb80416e7ebe59d95c7be`.

Checkpoint repaired SHA256: `306bc840d06abc3527605f9353eac3729bcdab7c806e5afba512406549f529c0`.

## Scope filter

Efek yang diuji adalah alignment dan yaw sebagai satu paket. Pemilihan checkpoint final dan snapshot clone sama untuk kedua arm; eksperimen tidak mengisolasi bug snapshot, tidak mereplikasi checkpoint historis300trajectory, dan tidak memilih model dari hasil dev/balanced100.

## Object filter

Cohort pilot memiliki grounding yang masih held. Protokol baru menyatakannya sebagai diagnosis numerik; gate/status bank lama tidak diubah. Empat task train tidak mewakili level4–5 atau dataset2500task. Balanced100 tetap cohort pengembangan, dengan scoring AABB strict<1m, horizon150, dan aturan STOP historis.

## Conclusions

F1–F3: kontrak repaired membantu imitasi closed-loop pada cohort train ini meskipun kedua arm sama-sama sempurna dalam klasifikasi offline. F4: belum ada perbaikan pada dev4. F5: kenaikan learned score balanced100 belum diketahui.

## Next steps / Risks

Job utama dan verifier berjalan detached dalam tmux. Verifier menunggu exit utama hingga maksimum1jam; kemudian memeriksa324episode, hash input/frame, checkpoint, matched sampling, reset/start/reference, urutan tahap, actual STOP, aritmetika skor, serta pasangan task. Hasil laporan final dihasilkan otomatis di server. Tidak perlu laptop tetap terhubung agar job berjalan.

Log utama: `/home/aimldl/LH-VLN/runs/balanced100_matched_contract_20261013_v1.log`.

Exit utama: `/home/aimldl/LH-VLN/runs/balanced100_matched_contract_20261013_v1.exit`.

Log/exit verifier: sibling `_verify.log` dan `_verify.exit`. Exit0 berarti proses/pemeriksaan selesai; bukan otomatis SR meningkat.

Belum ada hasil final balanced100 atau verifikasi final dalam catatan ini. Satu seed dan grounding yang belum divalidasi manusia membatasi kesimpulan. Tidak mengubah paper, Tabel4, checkpoint lama, atau proses pengguna lain. Training CPU2threads; simulasi serial.

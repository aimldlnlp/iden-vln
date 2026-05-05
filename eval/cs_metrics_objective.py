"""
cs_metrics_objective.py

Menghitung:
- M-index
- Language Entropy (LE)
- I-index
- Burstiness
- Memory
- Span Entropy (SE)
- CMI (Code-Mixing Index)
- T-index (berbasis MarianMT)

Strategi LID:
1) Kamus: indonesian_words.txt, english_words.txt
2) Fallback: langid (hanya id/en)
3) Kalau tetap tidak yakin -> 'unk'
Semua metrik hanya pakai token dengan label 'id' atau 'en' (unk diabaikan).
"""

import json
import math
import re
import statistics
import collections
from typing import List, Dict, Tuple

import langid
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# =====================================================================
# Konfigurasi umum
# =====================================================================

JSON_PATH = "logs/925_all_result.json"          # ganti kalau file-mu beda path
INDO_WORDLIST_PATH = "indonesian_words.txt"
EN_WORDLIST_PATH = "english_words.txt"

# Threshold minimum confidence langid, di bawah ini dianggap 'unk'
LID_MIN_CONF = 0.80

# Model MT untuk T-index (id<->en)
ID_EN_MODEL_NAME = "Helsinki-NLP/opus-mt-id-en"
EN_ID_MODEL_NAME = "Helsinki-NLP/opus-mt-en-id"


# =====================================================================
# Tokenisasi
# =====================================================================

def tokenize(text: str) -> List[str]:
    """
    Tokenisasi sederhana: ambil rangkaian huruf a-z/A-Z dan lowercase.
    """
    return re.findall(r"[a-zA-Z]+", text.lower())


# =====================================================================
# Load kamus kata Indonesia & Inggris
# =====================================================================

def load_wordlist(path: str) -> set:
    """
    Memuat daftar kata dari file teks:
    - satu kata per baris
    - baris kosong / diawali '#' di-skip
    - semua kata di-lowercase
    """
    words = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                words.add(line.lower())
    except FileNotFoundError:
        print(f"[WARNING] File wordlist '{path}' tidak ditemukan. "
              f"Menggunakan set kosong.\n")
    return words


INDONESIAN_WORDS = load_wordlist(INDO_WORDLIST_PATH)
ENGLISH_WORDS = load_wordlist(EN_WORDLIST_PATH)


# =====================================================================
# Model LID (langid) - fallback setelah kamus
# =====================================================================

# Batasi langid hanya ke id/en untuk lebih stabil
langid.set_languages(["id", "en"])


def lid_predict_lang(token: str) -> str:
    """
    Prediksi bahasa token dengan langid, dibatasi ke {id, en}.
    Kalau confidence < LID_MIN_CONF -> 'unk'.
    """
    if not token:
        return "unk"
    lang, score = langid.classify(token)
    if lang in ("id", "en") and score >= LID_MIN_CONF:
        return lang
    return "unk"


def detect_lang(token: str) -> str:
    """
    Strategi LID:
    1) Kalau token ada di kamus Indonesia -> 'id'
    2) Kalau token ada di kamus Inggris   -> 'en'
    3) Kalau tidak, gunakan langid
       - kalau confident -> 'id' / 'en'
       - kalau tidak -> 'unk'
    """
    t = token.lower()
    if t in INDONESIAN_WORDS:
        return "id"
    if t in ENGLISH_WORDS:
        return "en"
    return lid_predict_lang(t)


# =====================================================================
# Util: sequences & filtering
# =====================================================================

def sequences_by_utterance(texts: List[str]) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Dari list teks -> (tokens_per_utt, tags_per_utt).
    tags bisa berisi 'id', 'en', 'unk'.
    """
    tokens_per_utt = []
    tags_per_utt = []
    for t in texts:
        toks = tokenize(t)
        if not toks:
            continue
        tags = [detect_lang(tok) for tok in toks]
        tokens_per_utt.append(toks)
        tags_per_utt.append(tags)
    return tokens_per_utt, tags_per_utt


def flatten(seqs: List[List[str]]) -> List[str]:
    """Gabungkan semua sequence jadi satu list."""
    return [tag for seq in seqs for tag in seq]


def filter_id_en(seq: List[str]) -> List[str]:
    """Ambil hanya label 'id' dan 'en' (abaikan 'unk')."""
    return [tag for tag in seq if tag in ("id", "en")]


def lang_counts(seq: List[str]) -> Dict[str, int]:
    """
    Hitung frekuensi 'id' dan 'en' saja (unk diabaikan).
    """
    filtered = filter_id_en(seq)
    return dict(collections.Counter(filtered))


# =====================================================================
# M-index & Language Entropy
# =====================================================================

def m_index(counts: Dict[str, int]) -> float:
    """
    M-index (2 bahasa):
      p_i = proporsi token bahasa i
      s   = sum_i p_i^2
      M   = (1 - s) / s
    """
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0

    ps = [c / total for c in counts.values()]
    s = sum(p * p for p in ps)
    if s == 0:
        return 0.0
    return (1.0 - s) / s


def language_entropy(counts: Dict[str, int]) -> float:
    """
    Shannon entropy distribusi bahasa (bit), hanya 'id' & 'en'.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0

    H = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / total
        H -= p * math.log2(p)
    return H


# =====================================================================
# I-index (probabilitas switch)
# =====================================================================

def i_index(lang_seq: List[str]) -> float:
    """
    I-index: probabilitas bahwa token i != token i-1, hanya untuk id/en.
    'unk' diabaikan total.
    """
    seq = filter_id_en(lang_seq)
    if len(seq) <= 1:
        return 0.0
    switches = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    return switches / (len(seq) - 1)


# =====================================================================
# Spans, Burstiness, Memory, Span Entropy
# =====================================================================

def compute_spans(lang_seq: List[str]) -> List[Tuple[str, int]]:
    """
    Hitung language span hanya untuk id/en.
      input: ['id','unk','en','en','id']
      filter -> ['id','en','en','id']
      output spans: [('id',1), ('en',2), ('id',1)]
    """
    seq = filter_id_en(lang_seq)
    spans = []
    if not seq:
        return spans

    cur_lang = seq[0]
    cur_len = 1
    for tag in seq[1:]:
        if tag == cur_lang:
            cur_len += 1
        else:
            spans.append((cur_lang, cur_len))
            cur_lang = tag
            cur_len = 1
    spans.append((cur_lang, cur_len))
    return spans


def burstiness_and_memory(
    spans: List[Tuple[str, int]],
    k: float = 1.0,
    english_lang: str = "en",
) -> Tuple[float, float]:
    """
    Hitung Burstiness & Memory dari panjang span.
    Panjang span bahasa non-English bisa dikali faktor k (default 1).
    """
    if len(spans) < 2:
        return float("nan"), float("nan")

    lengths_aug = [
        length * (k if lang != english_lang else 1.0)
        for lang, length in spans
    ]

    n = len(lengths_aug)
    mean_all = statistics.mean(lengths_aug)
    stdev_all = statistics.pstdev(lengths_aug) if n > 1 else 0.0

    # Burstiness: (σ - μ) / (σ + μ)
    if mean_all + stdev_all == 0:
        burst = 0.0
    else:
        burst = (stdev_all - mean_all) / (stdev_all + mean_all)

    # Memory: lag-1 autocorrelation
    if n < 2:
        return burst, float("nan")

    first = lengths_aug[:-1]
    second = lengths_aug[1:]

    mean1 = statistics.mean(first)
    mean2 = statistics.mean(second)
    std1 = statistics.pstdev(first) if len(first) > 1 else 0.0
    std2 = statistics.pstdev(second) if len(second) > 1 else 0.0

    if std1 == 0 or std2 == 0:
        memory = 0.0
    else:
        num = 0.0
        for x, y in zip(first, second):
            num += (x - mean1) * (y - mean2)
        memory = num / ((n - 1) * std1 * std2)

    return burst, memory


def span_entropy(spans: List[Tuple[str, int]]) -> float:
    """
    Entropi distribusi panjang span.
    """
    if not spans:
        return 0.0
    length_counts = collections.Counter(length for _, length in spans)
    total_spans = sum(length_counts.values())

    H = 0.0
    for cnt in length_counts.values():
        p = cnt / total_spans
        H -= p * math.log2(p)
    return H


# =====================================================================
# CMI (Code-Mixing Index)
# =====================================================================

def cmi_for_sequence(lang_seq: List[str]) -> float:
    """
    CMI per utterance, hanya id/en.
      CMI = ((N - N_max) / N) * 100
    """
    seq = filter_id_en(lang_seq)
    if not seq:
        return 0.0
    counts = collections.Counter(seq)
    if len(counts) <= 1:
        return 0.0
    total = sum(counts.values())
    max_lang = max(counts.values())
    return (total - max_lang) / total * 100.0


def cmi_corpus(seqs: List[List[str]]) -> Tuple[float, float, float]:
    """
    Mengembalikan:
      - CMI_corpus_one_utt : CMI kalau seluruh korpus dianggap 1 utterance
      - CMI_avg_all        : rata-rata CMI di semua utterance (yg punya id/en)
      - CMI_avg_mixed      : rata-rata CMI hanya utterance dgn CMI>0
    """
    # Flatten & filter id/en
    flat_tags = filter_id_en(flatten(seqs))
    cmi_corpus_one_utt = cmi_for_sequence(flat_tags)

    cmi_utt_all = []
    for seq in seqs:
        cmi_val = cmi_for_sequence(seq)
        cmi_utt_all.append(cmi_val)

    cmi_avg_all = sum(cmi_utt_all) / len(cmi_utt_all) if cmi_utt_all else 0.0
    cmi_mixed = [c for c in cmi_utt_all if c > 0.0]
    cmi_avg_mixed = sum(cmi_mixed) / len(cmi_mixed) if cmi_mixed else 0.0

    return cmi_corpus_one_utt, cmi_avg_all, cmi_avg_mixed


# =====================================================================
# Switch words & T-index (MarianMT)
# =====================================================================

def extract_switch_words_per_utt(tokens_per_utt, tags_per_utt):
    """
    Ambil kata tepat setelah switch id<->en.
    Skip jika salah satu sisi 'unk'.
    Return: list of (word, src_lang, prev_lang)
    """
    all_switch_words = []
    for toks, tags in zip(tokens_per_utt, tags_per_utt):
        assert len(toks) == len(tags)
        for i in range(1, len(toks)):
            if tags[i] == "unk" or tags[i - 1] == "unk":
                continue
            if tags[i] != tags[i - 1]:
                all_switch_words.append((toks[i], tags[i], tags[i - 1]))
    return all_switch_words


# ---- Load model MT ----

print("[INFO] Memuat model MT Marian (id<->en) untuk T-index...")

try:
    tokenizer_id_en = AutoTokenizer.from_pretrained(ID_EN_MODEL_NAME)
    model_id_en = AutoModelForSeq2SeqLM.from_pretrained(ID_EN_MODEL_NAME)
    tokenizer_en_id = AutoTokenizer.from_pretrained(EN_ID_MODEL_NAME)
    model_en_id = AutoModelForSeq2SeqLM.from_pretrained(EN_ID_MODEL_NAME)
except Exception as e:
    raise RuntimeError(
        "Gagal memuat model MT. Pastikan 'transformers', 'sentencepiece', dan 'torch' sudah terinstall.\n"
        + str(e)
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_id_en.to(DEVICE)
model_en_id.to(DEVICE)

_MT_CACHE = {}


def mt_avg_logprob(word: str, src_lang: str, tgt_lang: str) -> float:
    """
    Hitung rata-rata log-probabilitas token terjemahan untuk 'word'
    menggunakan MarianMT (id<->en).

    - src_lang: 'id' atau 'en'
    - tgt_lang: 'id' atau 'en'

    Langkah:
      1. Pilih model/tokenizer sesuai arah (id->en atau en->id).
      2. Generate terjemahan dengan output_scores=True.
      3. Ambil log-prob token yang benar-benar dipilih model di setiap langkah.
      4. Rata-ratakan log-prob → skor MT untuk kata ini.
    """
    if not word:
        return float("nan")

    word_norm = word.strip().lower()
    key = (word_norm, src_lang, tgt_lang)
    if key in _MT_CACHE:
        return _MT_CACHE[key]

    # Kalau bahasa sama, tidak ada terjemahan bermakna
    if src_lang == tgt_lang:
        _MT_CACHE[key] = 0.0
        return 0.0

    # Pilih model & tokenizer
    if src_lang == "id" and tgt_lang == "en":
        tokenizer = tokenizer_id_en
        model = model_id_en
    elif src_lang == "en" and tgt_lang == "id":
        tokenizer = tokenizer_en_id
        model = model_en_id
    else:
        _MT_CACHE[key] = float("nan")
        return float("nan")

    # Tokenisasi input
    inputs = tokenizer(word_norm, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generate terjemahan
    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + 10,
            num_beams=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    sequences = gen_out.sequences  # (1, seq_len_dec)
    scores = gen_out.scores        # list panjang = # langkah decode

    if len(sequences) == 0 or len(scores) == 0:
        _MT_CACHE[key] = float("nan")
        return float("nan")

    seq = sequences[0]
    log_probs = []

    # scores[i] -> logits untuk token seq[i+1]
    for step, logits in enumerate(scores):
        token_id = seq[step + 1]
        log_prob = torch.log_softmax(logits[0], dim=-1)[token_id].item()
        log_probs.append(log_prob)

    if not log_probs:
        _MT_CACHE[key] = float("nan")
        return float("nan")

    avg_log_prob = float(sum(log_probs) / len(log_probs))
    _MT_CACHE[key] = avg_log_prob
    return avg_log_prob


def compute_t_index(switch_words) -> float:
    """
    T-index: rata-rata skor log-prob terjemahan pada kata setelah switch id<->en.
    """
    if not switch_words:
        return float("nan")

    scores = []
    for word, src_lang, prev_lang in switch_words:
        score = mt_avg_logprob(word, src_lang=src_lang, tgt_lang=prev_lang)
        if math.isnan(score):
            continue
        scores.append(score)

    if not scores:
        return float("nan")

    return sum(scores) / len(scores)


# =====================================================================
# MAIN
# =====================================================================

def main():
    # Load data JSON
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    success_tasks = data["success tasks"]

    # LID per utterance
    tokens_per_utt, tags_per_utt = sequences_by_utterance(success_tasks)

    # Flatten tags (termasuk unk) & hitung distribusi id/en
    flat_tags_all = flatten(tags_per_utt)
    flat_tags_id_en = filter_id_en(flat_tags_all)
    counts = lang_counts(flat_tags_all)

    unk_count = sum(1 for t in flat_tags_all if t == "unk")

    # M-index & Language Entropy
    M = m_index(counts)
    LE = language_entropy(counts)

    # I-index
    I = i_index(flat_tags_all)

    # Spans, Burstiness, Memory, Span Entropy
    spans = compute_spans(flat_tags_all)
    B, MEM = burstiness_and_memory(spans)
    SE = span_entropy(spans)

    # CMI
    CMI_corpus, CMI_avg_all, CMI_avg_mixed = cmi_corpus(tags_per_utt)

    # T-index
    switch_words = extract_switch_words_per_utt(tokens_per_utt, tags_per_utt)
    T = compute_t_index(switch_words)

    # Print hasil
    print("===== STATISTIK DASAR =====")
    print("Jumlah utterance:", len(tokens_per_utt))
    print("Jumlah token total (incl. unk):", len(flat_tags_all))
    print("Jumlah token id/en saja       :", len(flat_tags_id_en))
    print("Jumlah token 'unk'            :", unk_count)
    print("Distribusi bahasa (id/en saja):", counts)
    print()

    print("===== METRIK CODE-MIXING (hanya id/en) =====")
    print(f"M-index          : {M:.6f}")
    print(f"Language Entropy : {LE:.6f} bit")
    print(f"I-index          : {I:.6f}")
    print(f"Burstiness       : {B:.6f}")
    print(f"Memory           : {MEM:.6f}")
    print(f"Span Entropy     : {SE:.6f} bit")
    print()
    print("CMI (Gambäck & Das, hanya id/en):")
    print(f"  CMI korpus (1 utterance) : {CMI_corpus:.4f}")
    print(f"  CMI rata-rata semua utt  : {CMI_avg_all:.4f}")
    print(f"  CMI rata-rata utt mixed  : {CMI_avg_mixed:.4f}")
    print()
    print("===== T-INDEX =====")
    print(f"T-index (avg logprob MT)    : {T}")


if __name__ == "__main__":
    main()

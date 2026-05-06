"""
Compute Indonesian-English code-switching metrics for IDEN-VLN outputs.

Metrics:
- M-index
- Language Entropy (LE)
- I-index
- Burstiness
- Memory
- Span Entropy (SE)
- CMI (Code-Mixing Index)
- T-index based on MarianMT

Language identification strategy:
1. Wordlist lookup with indonesian_words.txt and english_words.txt.
2. langid fallback restricted to Indonesian and English.
3. Low-confidence tokens are labeled as 'unk'.

All aggregate metrics use only tokens labeled 'id' or 'en'. Unknown tokens are
kept for diagnostics but ignored by the metric formulas.
"""

import collections
import json
import math
import re
import statistics
from typing import Dict, List, Tuple

import langid
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# =====================================================================
# General configuration
# =====================================================================

JSON_PATH = "logs/925_all_result.json"
INDO_WORDLIST_PATH = "indonesian_words.txt"
EN_WORDLIST_PATH = "english_words.txt"

LID_MIN_CONF = 0.80

ID_EN_MODEL_NAME = "Helsinki-NLP/opus-mt-id-en"
EN_ID_MODEL_NAME = "Helsinki-NLP/opus-mt-en-id"


# =====================================================================
# Tokenization and wordlists
# =====================================================================

def tokenize(text: str) -> List[str]:
    """Tokenize alphabetic spans and lowercase them."""
    return re.findall(r"[a-zA-Z]+", text.lower())


def load_wordlist(path: str) -> set:
    """Load one lowercased word per line, skipping blanks and comments."""
    words = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                words.add(line.lower())
    except FileNotFoundError:
        print(f"[WARNING] Wordlist file '{path}' was not found. Using an empty set.\n")
    return words


INDONESIAN_WORDS = load_wordlist(INDO_WORDLIST_PATH)
ENGLISH_WORDS = load_wordlist(EN_WORDLIST_PATH)


# =====================================================================
# Language identification
# =====================================================================

langid.set_languages(["id", "en"])


def lid_predict_lang(token: str) -> str:
    """Predict token language with langid; low confidence becomes 'unk'."""
    if not token:
        return "unk"
    lang, score = langid.classify(token)
    if lang in ("id", "en") and score >= LID_MIN_CONF:
        return lang
    return "unk"


def detect_lang(token: str) -> str:
    """Classify a token as Indonesian, English, or unknown."""
    t = token.lower()
    if t in INDONESIAN_WORDS:
        return "id"
    if t in ENGLISH_WORDS:
        return "en"
    return lid_predict_lang(t)


def sequences_by_utterance(texts: List[str]) -> Tuple[List[List[str]], List[List[str]]]:
    """Convert text strings into token and language-tag sequences."""
    tokens_per_utt = []
    tags_per_utt = []
    for text in texts:
        tokens = tokenize(text)
        if not tokens:
            continue
        tags = [detect_lang(token) for token in tokens]
        tokens_per_utt.append(tokens)
        tags_per_utt.append(tags)
    return tokens_per_utt, tags_per_utt


def flatten(seqs: List[List[str]]) -> List[str]:
    """Flatten nested sequences into one list."""
    return [tag for seq in seqs for tag in seq]


def filter_id_en(seq: List[str]) -> List[str]:
    """Keep only Indonesian and English tags."""
    return [tag for tag in seq if tag in ("id", "en")]


def lang_counts(seq: List[str]) -> Dict[str, int]:
    """Count Indonesian and English tags only."""
    return dict(collections.Counter(filter_id_en(seq)))


# =====================================================================
# Core metrics
# =====================================================================

def m_index(counts: Dict[str, int]) -> float:
    """Compute M-index for a two-language distribution."""
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0

    proportions = [count / total for count in counts.values()]
    squared_sum = sum(p * p for p in proportions)
    if squared_sum == 0:
        return 0.0
    return (1.0 - squared_sum) / squared_sum


def language_entropy(counts: Dict[str, int]) -> float:
    """Compute Shannon entropy in bits for the id/en distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        if count == 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def i_index(lang_seq: List[str]) -> float:
    """Compute switch probability between adjacent id/en tags."""
    seq = filter_id_en(lang_seq)
    if len(seq) <= 1:
        return 0.0
    switches = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
    return switches / (len(seq) - 1)


def compute_spans(lang_seq: List[str]) -> List[Tuple[str, int]]:
    """Return contiguous language spans after removing unknown tags."""
    seq = filter_id_en(lang_seq)
    if not seq:
        return []

    spans = []
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
    """Compute burstiness and lag-1 memory from language-span lengths."""
    if len(spans) < 2:
        return float("nan"), float("nan")

    lengths = [
        length * (k if lang != english_lang else 1.0)
        for lang, length in spans
    ]

    mean_all = statistics.mean(lengths)
    stdev_all = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
    burst = 0.0 if mean_all + stdev_all == 0 else (stdev_all - mean_all) / (stdev_all + mean_all)

    first = lengths[:-1]
    second = lengths[1:]
    mean1 = statistics.mean(first)
    mean2 = statistics.mean(second)
    std1 = statistics.pstdev(first) if len(first) > 1 else 0.0
    std2 = statistics.pstdev(second) if len(second) > 1 else 0.0

    if std1 == 0 or std2 == 0:
        memory = 0.0
    else:
        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(first, second))
        memory = numerator / ((len(lengths) - 1) * std1 * std2)

    return burst, memory


def span_entropy(spans: List[Tuple[str, int]]) -> float:
    """Compute entropy of the span-length distribution."""
    if not spans:
        return 0.0
    length_counts = collections.Counter(length for _, length in spans)
    total_spans = sum(length_counts.values())

    entropy = 0.0
    for count in length_counts.values():
        p = count / total_spans
        entropy -= p * math.log2(p)
    return entropy


def cmi_for_sequence(lang_seq: List[str]) -> float:
    """Compute CMI for one utterance or sequence."""
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
    """Return corpus-level CMI, all-utterance mean CMI, and mixed-only mean CMI."""
    flat_tags = filter_id_en(flatten(seqs))
    cmi_corpus_one_utt = cmi_for_sequence(flat_tags)

    cmi_all = [cmi_for_sequence(seq) for seq in seqs]
    cmi_avg_all = sum(cmi_all) / len(cmi_all) if cmi_all else 0.0

    cmi_mixed = [cmi for cmi in cmi_all if cmi > 0.0]
    cmi_avg_mixed = sum(cmi_mixed) / len(cmi_mixed) if cmi_mixed else 0.0

    return cmi_corpus_one_utt, cmi_avg_all, cmi_avg_mixed


# =====================================================================
# Switch words and T-index
# =====================================================================

def extract_switch_words_per_utt(tokens_per_utt, tags_per_utt):
    """Extract words immediately after id/en switches."""
    all_switch_words = []
    for tokens, tags in zip(tokens_per_utt, tags_per_utt):
        assert len(tokens) == len(tags)
        for i in range(1, len(tokens)):
            if tags[i] == "unk" or tags[i - 1] == "unk":
                continue
            if tags[i] != tags[i - 1]:
                all_switch_words.append((tokens[i], tags[i], tags[i - 1]))
    return all_switch_words


print("[INFO] Loading MarianMT models (id<->en) for T-index...")

try:
    tokenizer_id_en = AutoTokenizer.from_pretrained(ID_EN_MODEL_NAME)
    model_id_en = AutoModelForSeq2SeqLM.from_pretrained(ID_EN_MODEL_NAME)
    tokenizer_en_id = AutoTokenizer.from_pretrained(EN_ID_MODEL_NAME)
    model_en_id = AutoModelForSeq2SeqLM.from_pretrained(EN_ID_MODEL_NAME)
except Exception as e:
    raise RuntimeError(
        "Failed to load MT models. Ensure that transformers, sentencepiece, and torch are installed.\n"
        + str(e)
    )

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_id_en.to(DEVICE)
model_en_id.to(DEVICE)

_MT_CACHE = {}


def mt_avg_logprob(word: str, src_lang: str, tgt_lang: str) -> float:
    """Compute average translation-token log probability for one word."""
    if not word:
        return float("nan")

    word_norm = word.strip().lower()
    key = (word_norm, src_lang, tgt_lang)
    if key in _MT_CACHE:
        return _MT_CACHE[key]

    if src_lang == tgt_lang:
        _MT_CACHE[key] = 0.0
        return 0.0

    if src_lang == "id" and tgt_lang == "en":
        tokenizer = tokenizer_id_en
        model = model_id_en
    elif src_lang == "en" and tgt_lang == "id":
        tokenizer = tokenizer_en_id
        model = model_en_id
    else:
        _MT_CACHE[key] = float("nan")
        return float("nan")

    inputs = tokenizer(word_norm, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + 10,
            num_beams=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    sequences = gen_out.sequences
    scores = gen_out.scores
    if len(sequences) == 0 or len(scores) == 0:
        _MT_CACHE[key] = float("nan")
        return float("nan")

    seq = sequences[0]
    log_probs = []
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
    """Compute mean MT log-probability over post-switch words."""
    if not switch_words:
        return float("nan")

    scores = []
    for word, src_lang, prev_lang in switch_words:
        score = mt_avg_logprob(word, src_lang=src_lang, tgt_lang=prev_lang)
        if not math.isnan(score):
            scores.append(score)

    if not scores:
        return float("nan")
    return sum(scores) / len(scores)


# =====================================================================
# Main entry point
# =====================================================================

def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    success_tasks = data["success tasks"]
    tokens_per_utt, tags_per_utt = sequences_by_utterance(success_tasks)

    flat_tags_all = flatten(tags_per_utt)
    flat_tags_id_en = filter_id_en(flat_tags_all)
    counts = lang_counts(flat_tags_all)
    unk_count = sum(1 for tag in flat_tags_all if tag == "unk")

    m_value = m_index(counts)
    le_value = language_entropy(counts)
    i_value = i_index(flat_tags_all)

    spans = compute_spans(flat_tags_all)
    burstiness, memory = burstiness_and_memory(spans)
    se_value = span_entropy(spans)

    cmi_corpus_value, cmi_avg_all, cmi_avg_mixed = cmi_corpus(tags_per_utt)

    switch_words = extract_switch_words_per_utt(tokens_per_utt, tags_per_utt)
    t_value = compute_t_index(switch_words)

    print("===== BASIC STATISTICS =====")
    print("Utterance count:", len(tokens_per_utt))
    print("Total token count (incl. unk):", len(flat_tags_all))
    print("Token count for id/en only   :", len(flat_tags_id_en))
    print("Token count for 'unk'        :", unk_count)
    print("Language distribution (id/en):", counts)
    print()

    print("===== CODE-MIXING METRICS (id/en only) =====")
    print(f"M-index          : {m_value:.6f}")
    print(f"Language Entropy : {le_value:.6f} bit")
    print(f"I-index          : {i_value:.6f}")
    print(f"Burstiness       : {burstiness:.6f}")
    print(f"Memory           : {memory:.6f}")
    print(f"Span Entropy     : {se_value:.6f} bit")
    print()
    print("CMI (Gamback and Das, id/en only):")
    print(f"  Corpus CMI (1 utterance) : {cmi_corpus_value:.4f}")
    print(f"  Average CMI, all utt     : {cmi_avg_all:.4f}")
    print(f"  Average CMI, mixed utt   : {cmi_avg_mixed:.4f}")
    print()
    print("===== T-INDEX =====")
    print(f"T-index (avg MT logprob): {t_value}")


if __name__ == "__main__":
    main()


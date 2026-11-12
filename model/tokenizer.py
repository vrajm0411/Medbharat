"""Dual-Script Medical Tokenizer for MedBharat-1.

Trains a BPE tokenizer on the medical corpus that handles:
- English medical terminology
- Hindi / Devanagari text
- Mixed Hinglish medical text
- Medical symbols and abbreviations
"""

import json
import gzip
from pathlib import Path
from typing import List

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from tokenizers.normalizers import Sequence, NFKC, Lowercase


def build_tokenizer(
    corpus_files: List[str],
    vocab_size: int = 48_000,
    output_path: str = "medbharat_tokenizer.json",
    min_frequency: int = 3,
) -> Tokenizer:
    """Train a BPE tokenizer on medical corpus files.

    Args:
        corpus_files: list of .txt.gz or .txt or .jsonl.gz file paths
        vocab_size: target vocabulary size
        output_path: where to save the trained tokenizer
        min_frequency: minimum token frequency to include

    Returns:
        Trained tokenizer
    """
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Normalizer: Unicode NFKC normalization (handles Hindi + English)
    tokenizer.normalizer = Sequence([NFKC()])

    # Pre-tokenizer: split on whitespace and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Punctuation(),
    ])

    # Special tokens
    special_tokens = [
        "<pad>",      # 0 - padding
        "<s>",        # 1 - beginning of sequence
        "</s>",       # 2 - end of sequence
        "<unk>",      # 3 - unknown
        "<mask>",     # 4 - mask (for potential MLM later)
        "<en>",       # 5 - English marker
        "<hi>",       # 6 - Hindi marker
        "<mix>",      # 7 - Hinglish/Mixed marker
        "<question>", # 8 - Q&A structure
        "<answer>",   # 9 - Q&A structure
        "<drug>",     # 10 - drug entity
        "<disease>",  # 11 - disease entity
        "<treatment>",# 12 - treatment entity
    ]

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Build text iterator from files
    def text_iterator():
        for fpath in corpus_files:
            p = Path(fpath)
            try:
                if p.suffix == ".gz":
                    if ".jsonl" in p.name:
                        with gzip.open(p, "rt", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    record = json.loads(line)
                                    # Extract all text fields
                                    for field in ["text", "title", "question", "answer",
                                                   "explanation", "long_answer", "context"]:
                                        val = record.get(field, "")
                                        if val and len(val) > 10:
                                            yield val
                                except json.JSONDecodeError:
                                    continue
                    else:
                        # .txt.gz corpus file
                        with gzip.open(p, "rt", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line and len(line) > 10:
                                    yield line
                else:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and len(line) > 10:
                                yield line
            except Exception as e:
                print(f"Warning: Could not read {fpath}: {e}")
                continue

    # Train
    print(f"Training tokenizer with vocab_size={vocab_size} on {len(corpus_files)} files...")
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

    # Post-processor: add BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )

    # Decoder
    tokenizer.decoder = decoders.BPEDecoder()

    # Save
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    """Load a trained tokenizer from file."""
    return Tokenizer.from_file(path)


def detect_language(text: str) -> int:
    """Simple language detection: 0=English, 1=Hindi, 2=Mixed.

    Uses Unicode range check for Devanagari characters.
    """
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_alpha = sum(1 for c in text if c.isalpha())

    if total_alpha == 0:
        return 0

    hindi_ratio = hindi_chars / total_alpha

    if hindi_ratio > 0.5:
        return 1  # Hindi
    elif hindi_ratio > 0.1:
        return 2  # Mixed/Hinglish
    else:
        return 0  # English

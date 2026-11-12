"""Streaming dataset for MedBharat-1 training.

Memory-efficient: reads data line-by-line, never loads full dataset.
Supports both JSONL.gz (structured) and TXT.gz (corpus) formats.
"""

import json
import gzip
import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import IterableDataset

from model.tokenizer import detect_language


class MedBharatDataset(IterableDataset):
    """Streaming dataset that tokenizes text on-the-fly.

    Reads from .jsonl.gz or .txt.gz files and produces
    fixed-length token sequences for causal language modeling.
    """

    def __init__(
        self,
        data_files: List[str],
        tokenizer,
        max_seq_len: int = 2048,
        shuffle_files: bool = True,
    ):
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shuffle_files = shuffle_files

        self.pad_id = tokenizer.token_to_id("<pad>") or 0
        self.bos_id = tokenizer.token_to_id("<s>") or 1
        self.eos_id = tokenizer.token_to_id("</s>") or 2

    def _read_texts(self):
        """Yield text strings from data files."""
        files = list(self.data_files)
        if self.shuffle_files:
            random.shuffle(files)

        for fpath in files:
            p = Path(fpath)
            try:
                if ".jsonl" in p.name:
                    open_fn = gzip.open if p.suffix == ".gz" else open
                    with open_fn(p, "rt", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                                text = self._extract_text(record)
                                if text and len(text) > 50:
                                    yield text
                            except json.JSONDecodeError:
                                continue
                else:
                    # .txt or .txt.gz corpus files
                    open_fn = gzip.open if p.suffix == ".gz" else open
                    with open_fn(p, "rt", encoding="utf-8") as f:
                        doc_buffer = []
                        for line in f:
                            line = line.strip()
                            if not line:
                                # Empty line = document boundary
                                if doc_buffer:
                                    yield " ".join(doc_buffer)
                                    doc_buffer = []
                            else:
                                doc_buffer.append(line)
                        if doc_buffer:
                            yield " ".join(doc_buffer)

            except Exception as e:
                print(f"Warning: Could not read {fpath}: {e}")
                continue

    def _extract_text(self, record: dict) -> str:
        """Extract text from a JSONL record."""
        parts = []
        for field in ["title", "text", "question", "answer", "explanation",
                       "long_answer", "context"]:
            val = record.get(field, "")
            if val:
                parts.append(str(val))
        return " ".join(parts)

    def __iter__(self):
        """Yield tokenized fixed-length sequences with language IDs."""
        token_buffer = []
        lang_buffer = []

        for text in self._read_texts():
            # Tokenize
            encoded = self.tokenizer.encode(text)
            tokens = encoded.ids

            # Detect language for the whole text
            lang_id = detect_language(text)

            token_buffer.extend(tokens)
            lang_buffer.extend([lang_id] * len(tokens))

            # Yield full sequences from buffer
            while len(token_buffer) >= self.max_seq_len + 1:
                input_ids = token_buffer[: self.max_seq_len]
                labels = token_buffer[1 : self.max_seq_len + 1]
                language_ids = lang_buffer[: self.max_seq_len]

                yield {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "language_ids": torch.tensor(language_ids, dtype=torch.long),
                }

                # Slide buffer
                token_buffer = token_buffer[self.max_seq_len:]
                lang_buffer = lang_buffer[self.max_seq_len:]


class CurriculumDataManager:
    """Manages data files for each curriculum phase.

    Phase 1 (Pre-Med):      wiki_medical + ncbi_books
    Phase 2 (Pathology):     pubmed (disease) + icd11
    Phase 3 (Pharmacology):  nlem_2025 + drugbank + pubmed (drug)
    Phase 4 (Clinical):      medqa + pubmedqa + who_india
    Phase 5 (Indian Spec.):  medmcqa + pubmed (india) + nlem
    Phase 6 (Residency):     ALL data mixed
    """

    PHASES = {
        1: {
            "name": "Pre-Med (Basic Sciences)",
            "sources": ["wiki_medical", "ncbi_books"],
            "epochs": 2,
            "lr_schedule": "warmup",
        },
        2: {
            "name": "Pathology (Diseases & Diagnosis)",
            "sources": ["pubmed", "icd11"],
            "epochs": 2,
            "lr_schedule": "peak",
        },
        3: {
            "name": "Pharmacology (Drugs & Treatment)",
            "sources": ["nlem_2025", "drugbank"],
            "epochs": 3,
            "lr_schedule": "peak",
        },
        4: {
            "name": "Clinical (Q&A & Reasoning)",
            "sources": ["medqa", "pubmedqa", "who_india"],
            "epochs": 2,
            "lr_schedule": "decay",
        },
        5: {
            "name": "Indian Specialization",
            "sources": ["medmcqa", "pubmed", "nlem_2025"],
            "epochs": 3,
            "lr_schedule": "decay",
        },
        6: {
            "name": "Residency (Full Integration)",
            "sources": "ALL",
            "epochs": 1,
            "lr_schedule": "minimum",
        },
    }

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def get_phase_files(self, phase: int) -> List[str]:
        """Get data files for a curriculum phase."""
        phase_config = self.PHASES[phase]
        sources = phase_config["sources"]

        files = []
        if sources == "ALL":
            # Phase 6: use everything
            for f in self.data_dir.rglob("*.jsonl.gz"):
                files.append(str(f))
            for f in self.data_dir.rglob("*_corpus.txt.gz"):
                files.append(str(f))
        else:
            for source in sources:
                source_dir = self.data_dir / source
                if source_dir.exists():
                    for f in source_dir.rglob("*.jsonl.gz"):
                        files.append(str(f))
                    for f in source_dir.rglob("*_corpus.txt.gz"):
                        files.append(str(f))
                # Also check exported directory
                export_dir = self.data_dir.parent / "exported" / source
                if export_dir.exists():
                    for f in export_dir.rglob("*.jsonl.gz"):
                        files.append(str(f))
                    for f in export_dir.rglob("*_corpus.txt.gz"):
                        files.append(str(f))

        return files

    def get_phase_config(self, phase: int) -> dict:
        return self.PHASES[phase]

---
license: apache-2.0
language:
  - en
  - hi
tags:
  - medical
  - mixture-of-experts
  - healthcare
  - indian-medical
  - dual-script
  - ayush
  - pytorch
library_name: pytorch
pipeline_tag: text-generation
---

# MedBharat-1

> **Model weights** (`checkpoints/*/model.pt`, ~451 MB each) are too large for GitHub and live on Hugging Face: https://huggingface.co/Vraj0411/Medbharat

A ~500M parameter **Mixture of Medical Experts** language model with dual-script (English + Hindi) support, purpose-built for Indian healthcare.

## Model Details

| Property | Value |
|----------|-------|
| **Total Parameters** | ~500M |
| **Active Parameters** | ~260M per forward pass (top-2 experts) |
| **Architecture** | Curriculum-trained Mixture of Medical Experts |
| **Languages** | English, Hindi, Hinglish |
| **Vocab Size** | 48,000 (32K English + 12K Hindi + 4K medical symbols) |
| **Max Sequence Length** | 256 |
| **Embedding Dim** | 512 |
| **Shared Layers** | 6 |
| **Expert Layers** | 4 |
| **Attention Heads** | 8 (+ 2 medical-specific heads) |

## Architecture

MedBharat-1 uses a **Mixture of 6 Medical Experts** with top-2 routing:

1. **Diagnosis** - Disease identification and differential diagnosis
2. **Pharmacology** - Drug information, interactions, Indian drug brands
3. **Treatment** - Treatment protocols and clinical guidelines
4. **AYUSH** - Ayurveda, Yoga, Unani, Siddha, Homeopathy
5. **Anatomy** - Human anatomy and physiology
6. **General** - General medical knowledge

Key architectural features:
- **Medical Sparse Attention** with dedicated medical attention heads
- **RoPE** positional encoding
- **SwiGLU** activation in feed-forward layers
- **RMSNorm** for layer normalization
- **Language embeddings** for dual-script support

## Benchmark Results

| Category | Accuracy | Samples |
|----------|----------|---------|
| Clinical | 100.0% | 2 |
| Hindi Medical | 50.0% | 2 |
| Tropical Disease | 40.0% | 5 |
| Indian Drugs | 20.0% | 5 |
| AYUSH | 25.0% | 4 |
| Indian Exam | 25.0% | 4 |
| **Overall** | **36.4%** | **22** |

## Training

The model was trained using a 6-phase curriculum:
- Curriculum-based training with progressive difficulty
- Trained on medical corpus including Indian medical data
- Supports medical Q&A in English and Hindi

## Usage

```python
import torch
from model.config import MedBharatConfig
from model.transformer import MedBharatModel

config = MedBharatConfig()
model = MedBharatModel(config)
model.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=True))
model.eval()
```

## Limitations

- This is a research model and should **NOT** be used for actual medical diagnosis or treatment
- Benchmark accuracy is limited; the model needs further training and evaluation
- Sequence length is limited to 256 tokens

## Author

**Vraj Modi** ([Vraj0411](https://huggingface.co/Vraj0411))

"""MedBharat-1 Inference Server for OCI deployment.

FastAPI server with:
  - /generate — text generation endpoint
  - /medical_qa — structured medical Q&A
  - /health — health check
  - /benchmark — run evaluation benchmark
"""

import os
import time
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model.config import MedBharatConfig
from model.transformer import MedBharatModel
from model.tokenizer import load_tokenizer, detect_language

# --- Configuration ---
WEIGHTS_DIR = os.environ.get("MEDBHARAT_WEIGHTS", "/app/weights")
TOKENIZER_PATH = os.environ.get("MEDBHARAT_TOKENIZER", "/app/weights/medbharat_tokenizer.json")
DEVICE = os.environ.get("MEDBHARAT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS = int(os.environ.get("MEDBHARAT_MAX_TOKENS", "512"))

app = FastAPI(
    title="MedBharat-1 API",
    description="Indian Healthcare LLM — Bilingual Medical AI",
    version="1.0.0",
)

# --- Global model state ---
model: Optional[MedBharatModel] = None
tokenizer = None
config: Optional[MedBharatConfig] = None


# --- Request/Response schemas ---
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_k: int = Field(default=50, ge=1, le=500)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)


class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    language_detected: str
    tokens_generated: int
    time_seconds: float


class MedicalQARequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2048)
    max_tokens: int = Field(default=512, ge=1, le=1024)


class MedicalQAResponse(BaseModel):
    question: str
    answer: str
    language: str
    confidence_note: str
    time_seconds: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    parameters: str


# --- Startup ---
@app.on_event("startup")
async def load_model():
    global model, tokenizer, config

    print(f"Loading MedBharat-1 on {DEVICE}...")

    config = MedBharatConfig()

    # Load tokenizer
    tok_path = Path(TOKENIZER_PATH)
    if not tok_path.exists():
        print(f"WARNING: Tokenizer not found at {tok_path}")
        return
    tokenizer = load_tokenizer(str(tok_path))

    # Load model
    model = MedBharatModel(config)
    weights_path = Path(WEIGHTS_DIR) / "model.pt"
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"WARNING: No weights at {weights_path}, using random init")

    model = model.to(DEVICE)
    model.eval()

    params = model.count_parameters()
    print(f"Model loaded: {params['total'] / 1e6:.0f}M params on {DEVICE}")


# --- Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health():
    loaded = model is not None and tokenizer is not None
    params = model.count_parameters() if model else {}
    return HealthResponse(
        status="healthy" if loaded else "model_not_loaded",
        model_loaded=loaded,
        device=DEVICE,
        parameters=f"{params.get('total', 0) / 1e6:.0f}M" if params else "N/A",
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    # Detect language
    lang_id = detect_language(req.prompt)
    lang_names = {0: "English", 1: "Hindi", 2: "Mixed/Hinglish"}

    # Tokenize
    encoded = tokenizer.encode(req.prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(DEVICE)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )

    # Decode
    generated_ids = output_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    # Strip the prompt portion
    prompt_text = tokenizer.decode(encoded.ids)
    answer_text = generated_text[len(prompt_text):].strip()

    elapsed = time.time() - start
    return GenerateResponse(
        generated_text=answer_text,
        prompt=req.prompt,
        language_detected=lang_names.get(lang_id, "Unknown"),
        tokens_generated=len(generated_ids) - len(encoded.ids),
        time_seconds=round(elapsed, 3),
    )


@app.post("/medical_qa", response_model=MedicalQAResponse)
async def medical_qa(req: MedicalQARequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    lang_id = detect_language(req.question)
    lang_names = {0: "English", 1: "Hindi", 2: "Mixed/Hinglish"}

    # Format as medical Q&A prompt
    prompt = f"Medical Question: {req.question}\nAnswer:"

    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=req.max_tokens,
            temperature=0.6,  # lower temp for factual answers
            top_k=40,
            top_p=0.85,
        )

    generated_ids = output_ids[0].tolist()
    full_text = tokenizer.decode(generated_ids)
    prompt_text = tokenizer.decode(encoded.ids)
    answer = full_text[len(prompt_text):].strip()

    elapsed = time.time() - start
    return MedicalQAResponse(
        question=req.question,
        answer=answer,
        language=lang_names.get(lang_id, "Unknown"),
        confidence_note="This is an AI-generated response. Always consult a qualified healthcare professional.",
        time_seconds=round(elapsed, 3),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

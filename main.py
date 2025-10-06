from __future__ import annotations

import base64
import io
import os
import time
import uuid
import json
import threading
from typing import Any, Dict, List, Optional, Tuple, Iterable

import requests
import torch
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from transformers import (
    Qwen3VLMoeForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
    AutoModel,
    AutoTokenizer,
)

# =========================
# Config
# =========================
MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"     # vision+chat
EMBED_MODEL_ID = "intfloat/e5-small-v2"         # text embeddings (change if you prefer)
CONTEXT_TOKENS = 32000
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = "auto"                                  # or torch.float16 to save memory on Apple Silicon
MAX_IMAGE_BYTES = 25 * 1024 * 1024
ALLOW_NO_AUTH = True                             # flip to False and add auth if exposing publicly

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# =========================
# FastAPI app & globals
# =========================
app = FastAPI(title="OpenAI-compatible Qwen3-VL (MPS)",
)

# CORS for all origins (tighten as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # e.g., ["http://localhost:3000"]
    allow_credentials=False,
    allow_methods=["*"],        # includes OPTIONS
    allow_headers=["*"],
    expose_headers=["*"],
)

# vision+chat
model = None
processor = None

# embeddings
emb_model = None
emb_tokenizer = None
emb_device = DEVICE  # small model, MPS or CPU both fine


# =========================
# Helpers
# =========================
def _now_unix() -> int:
    return int(time.time())


def _load_image_from_bytes(data: bytes) -> Image.Image:
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def _fetch_image(url: str) -> Image.Image:
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return _load_image_from_bytes(r.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")


def _image_from_openai_image_url(obj: Dict[str, Any]) -> Image.Image:
    """
    OpenAI messages:
      {"type":"image_url","image_url":{"url":"https://..."}}
    Also supports data URLs: data:image/png;base64,....
    """
    url = obj.get("image_url", {}).get("url")
    if not url:
        raise HTTPException(status_code=400, detail="image_url.url missing")

    if url.startswith("data:"):
        try:
            _, b64 = url.split(",", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail="Malformed data URL")
        try:
            data = base64.b64decode(b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 in data URL")
        return _load_image_from_bytes(data)
    else:
        return _fetch_image(url)


def _map_openai_messages_to_qwen(
    messages: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
    """
    Converts OpenAI ChatCompletions messages to Qwen chat schema and collects PIL images.
    """
    qwen_messages: List[Dict[str, Any]] = []
    pil_images: List[Image.Image] = []

    for m in messages:
        role = m.get("role")
        content = m.get("content")

        if isinstance(content, str):
            qwen_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue

        if not isinstance(content, list):
            raise HTTPException(status_code=400, detail="message.content must be string or list")

        merged: List[Dict[str, Any]] = []
        for part in content:
            ptype = part.get("type")
            if ptype == "text":
                merged.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image_url":
                img = _image_from_openai_image_url(part)
                pil_images.append(img)
                merged.append({"type": "image", "image": img})
            else:
                continue

        if not merged:
            merged = [{"type": "text", "text": ""}]

        qwen_messages.append({"role": role, "content": merged})

    return qwen_messages, pil_images


def _pack_qwen_inputs(qwen_messages: List[Dict[str, Any]], pil_images: List[Image.Image]):
    prompt = processor.apply_chat_template(
        qwen_messages, add_generation_prompt=True, tokenize=False
    )
    proc_inputs = processor(
        text=[prompt],
        images=pil_images if pil_images else None,
        return_tensors="pt",
    )
    prompt_tokens = int(proc_inputs["input_ids"].shape[-1])
    proc_inputs = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in proc_inputs.items()}
    return proc_inputs, prompt_tokens


def _sse_format(data_obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data_obj, ensure_ascii=False)}\n\n"


# =========================
# Model loading
# =========================
@app.on_event("startup")
def _load_models():
    global model, processor, emb_model, emb_tokenizer
    t0 = time.time()

    # vision chat
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map=None,
    )
    model.config.max_position_embeddings = CONTEXT_TOKENS
    processor.tokenizer.model_max_length = CONTEXT_TOKENS
    model.to(DEVICE)
    model.eval()

    # embeddings (small)
    emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    emb_model = AutoModel.from_pretrained(EMBED_MODEL_ID)
    emb_model.to(emb_device)
    emb_model.eval()

    print(f"[startup] Loaded chat={MODEL_ID} and emb={EMBED_MODEL_ID} on {DEVICE} in {time.time()-t0:.1f}s")


# =========================
# OpenAI-compatible schemas
# =========================
class ChatMessage(BaseModel):
    role: str
    content: Any  # str OR list of content parts (text / image_url)


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=128, ge=1)
    temperature: Optional[float] = Field(default=0.7, ge=0.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0)
    top_k: Optional[int] = Field(default=50, ge=0)
    repetition_penalty: Optional[float] = Field(default=1.0, ge=0.0)
    stream: Optional[bool] = False
    n: Optional[int] = 1
    stop: Optional[Any] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int
    message: ChoiceMessage
    finish_reason: str = "stop"
    logprobs: Optional[Any] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionsResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class ModelData(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "owner"


class ListModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelData]


# Embeddings
class EmbeddingRequest(BaseModel):
    model: str
    input: Any  # str | List[str]
    encoding_format: Optional[str] = None  # we always return float lists
    user: Optional[str] = None


class EmbeddingItem(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingItem]
    model: str
    usage: Usage


# =========================
# Endpoints
# =========================
@app.get("/v1/models", response_model=ListModelsResponse)
def list_models():
    return ListModelsResponse(
        data=[
            ModelData(id=MODEL_ID, created=_now_unix()),
            ModelData(id=EMBED_MODEL_ID, created=_now_unix()),
        ]
    )


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionsRequest = Body(...)):
    if not ALLOW_NO_AUTH:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # We accept any 'model' string but serve local MODEL_ID
    qwen_msgs, pil_images = _map_openai_messages_to_qwen([m.dict() for m in req.messages])
    proc_inputs, prompt_tokens = _pack_qwen_inputs(qwen_msgs, pil_images)

    chat_id = "chatcmpl-" + uuid.uuid4().hex
    created = _now_unix()

    if req.stream:
        # Streaming via TextIteratorStreamer
        streamer = TextIteratorStreamer(
            tokenizer=processor.tokenizer if hasattr(processor, "tokenizer") else processor,  # AutoProcessor exposes .tokenizer
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        gen_kwargs = dict(
            **proc_inputs,
            max_new_tokens=req.max_tokens or 128,
            do_sample=True,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.9,
            top_k=req.top_k or 50,
            repetition_penalty=req.repetition_penalty or 1.0,
            streamer=streamer,
        )

        def _worker():
            with torch.no_grad():
                model.generate(**gen_kwargs)

        thread = threading.Thread(target=_worker)
        thread.start()

        def event_stream() -> Iterable[str]:
            # Initial header chunk (mirrors OpenAI shape loosely)
            yield _sse_format({
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            })

            # Accumulate text to compute completion_tokens post-hoc if desired
            collected = ""
            for piece in streamer:
                if not piece:
                    continue
                collected += piece
                yield _sse_format({
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                })

            # Final chunk with finish_reason and usage (approximate completion_tokens by re-tokenizing)
            comp_ids = (processor.tokenizer if hasattr(processor, "tokenizer") else processor).encode(
                collected, add_special_tokens=False
            )
            completion_tokens = int(len(comp_ids))
            usage = {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": completion_tokens,
                "total_tokens": int(prompt_tokens) + completion_tokens,
            }

            yield _sse_format({
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": usage,
            })
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming
    with torch.no_grad():
        out_ids = model.generate(
            **proc_inputs,
            max_new_tokens=req.max_tokens or 128,
            do_sample=True,
            temperature=req.temperature or 0.7,
            top_p=req.top_p or 0.9,
            top_k=req.top_k or 50,
            repetition_penalty=req.repetition_penalty or 1.0,
        )

    trimmed = [o[len(proc_inputs["input_ids"][0]):] for o in out_ids]
    completion_tokens = int(trimmed[0].shape[-1])
    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    resp = ChatCompletionsResponse(
        id=chat_id,
        created=created,
        model=MODEL_ID,
        choices=[
            Choice(index=0, message=ChoiceMessage(content=text), finish_reason="stop")
        ],
        usage=Usage(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(prompt_tokens) + int(completion_tokens),
        ),
    )
    return resp


# ---------- Embeddings ----------
def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # mean pooling with attention mask
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def embeddings(req: EmbeddingRequest = Body(...)):
    if not ALLOW_NO_AUTH:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Normalize inputs to list[str]
    if isinstance(req.input, str):
        texts = [req.input]
    elif isinstance(req.input, list):
        # OpenAI allows list of strings; if nested lists are sent, flatten strings only
        texts = []
        for x in req.input:
            if isinstance(x, str):
                texts.append(x)
            else:
                raise HTTPException(status_code=400, detail="Only string inputs are supported")
    else:
        raise HTTPException(status_code=400, detail="Invalid input type")

    with torch.no_grad():
        tok = emb_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tok = {k: v.to(emb_device) for k, v in tok.items()}
        out = emb_model(**tok)
        sent_emb = _mean_pool(out.last_hidden_state, tok["attention_mask"])
        # Normalize (common practice for retrieval)
        sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)

    data = []
    for i, vec in enumerate(sent_emb.cpu().tolist()):
        data.append(EmbeddingItem(embedding=vec, index=i))

    usage = Usage(
        prompt_tokens=int(tok["input_ids"].numel()),  # rough count across batch
        completion_tokens=0,
        total_tokens=int(tok["input_ids"].numel()),
    )

    return EmbeddingResponse(
        data=data,
        model=EMBED_MODEL_ID,
        usage=usage,
    )


# Minimal health
@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model": MODEL_ID, "embeddings": EMBED_MODEL_ID}

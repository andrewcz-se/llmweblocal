import json
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import httpx
from pydantic import BaseModel

OLLAMA_URL = "http://localhost:11434"
last_used_model = None

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    options: Optional[Dict[str, Any]] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Shutdown logic: Unload the last used model from VRAM
    if last_used_model:
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": last_used_model, "keep_alive": 0}
                )
        except Exception:
            pass

app = FastAPI(title="Local Ollama Chat", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/models")
async def get_models():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vram/{model_name}")
async def estimate_vram(model_name: str, num_ctx: int = 4096):
    try:
        print(f"[VRAM_ESTIMATE] Model: {model_name}, Context: {num_ctx}")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/show",
                json={"name": model_name}
            )
            if resp.status_code != 200:
                print(f"[VRAM_ESTIMATE] Ollama API Error: {resp.status_code} - {resp.text}")
                raise HTTPException(status_code=resp.status_code, detail=f"Ollama error: {resp.text}")
            info = resp.json()

        model_info = info.get("model_info", {})
        details = info.get("details", {})
        quantization = details.get("quantization_level", "Q4_0")
        
        # Log metadata for debugging
        print(f"[VRAM_METADATA] Arch: {model_info.get('general.architecture')}")
        print(f"[VRAM_METADATA] Params: {model_info.get('general.parameter_count')}")
        
        # 1. Get the architecture prefix
        arch = model_info.get("general.architecture") or "llama"
        
        # 2. Get parameter count with strict None handling
        exact_param_count = model_info.get("general.parameter_count")
        if exact_param_count is not None:
            param_billions = float(exact_param_count) / 1e9
        else:
            param_str = str(details.get("parameter_size", "0B")).strip().upper()
            try:
                if "M" in param_str:
                    param_billions = float(param_str.replace("M", "").replace("B", "")) / 1000
                else:
                    param_billions = float(param_str.replace("B", ""))
            except Exception:
                param_billions = 0

        bits_map = {
            "IQ2_XXS": 2.3, "IQ2_XS": 2.4, "IQ2_S": 2.5, "IQ2_M": 2.7,
            "Q2_K": 2.6, 
            "IQ3_XXS": 3.1, "IQ3_S": 3.2, "IQ3_M": 3.3,
            "Q3_K_S": 3.0, "Q3_K_M": 3.3, "Q3_K_L": 3.5,
            "IQ4_NL": 4.3, "IQ4_XS": 4.3,
            "Q4_0": 4.0, "Q4_1": 4.5, "Q4_K_S": 4.0, "Q4_K_M": 4.5,
            "Q5_0": 5.0, "Q5_1": 5.5, "Q5_K_S": 5.0, "Q5_K_M": 5.5,
            "Q6_K": 6.0, "Q8_0": 8.0, "F16": 16.0, "BF16": 16.0, "F32": 32.0
        }
        
        bits_per_weight = bits_map.get(quantization.upper(), 4.5)
        model_weights_gib = (param_billions * 1e9 * bits_per_weight / 8) / (1024**3)
        
        # 3. KV Cache variables with strict None handling (using 'or' for fallback)
        num_kv_layers = model_info.get(f"{arch}.block_count") or \
                        model_info.get("llama.block_count") or 32
        
        num_kv_heads = model_info.get(f"{arch}.attention.head_count_kv") or \
                       model_info.get(f"{arch}.attention.head_count") or \
                       model_info.get("llama.attention.head_count_kv") or 8
        
        head_dim = model_info.get(f"{arch}.attention.key_length")
        if not head_dim:
            embed_len = model_info.get(f"{arch}.embedding_length") or \
                        model_info.get("llama.embedding_length") or 4096
            head_count = model_info.get(f"{arch}.attention.head_count") or \
                         model_info.get("llama.attention.head_count") or 32
            head_dim = embed_len // head_count if head_count else 128
            
        bytes_per_el = 2 
        kv_cache_bytes = (2 * int(num_kv_layers) * int(num_kv_heads) * int(head_dim) * int(num_ctx) * bytes_per_el)
        kv_cache_gib = kv_cache_bytes / (1024**3)
        
        compute_overhead_gib = model_weights_gib * 0.10
        total_gib = model_weights_gib + kv_cache_gib + compute_overhead_gib
        
        print(f"[VRAM_RESULT] Weights: {model_weights_gib:.2f}, KV: {kv_cache_gib:.2f}, Total: {total_gib:.2f}")

        return {
            "model": model_name,
            "architecture": arch,
            "quantization": quantization,
            "parameters_billions": round(param_billions, 2),
            "num_ctx": num_ctx,
            "estimates": {
                "model_weights_gib": round(model_weights_gib, 2),
                "kv_cache_gib": round(kv_cache_gib, 2),
                "compute_overhead_gib": round(compute_overhead_gib, 2),
                "total_gib": round(total_gib, 2)
            }
        }
    except Exception as e:
        import traceback
        print(f"[VRAM_ESTIMATE] Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    global last_used_model
    last_used_model = request.model

    # Technical Console Logging
    print(f"\n[OLLAMA_PROXY] REQUEST")
    print(f"MODEL: {request.model}")
    print(f"OPTIONS: {json.dumps(request.options, indent=2)}")
    print(f"MESSAGES: {json.dumps(request.messages, indent=2)}")
    print("-" * 40)
    
    async def event_generator():
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": request.model,
                        "messages": request.messages,
                        "options": request.options,
                        "stream": True
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            # Ollama returns a JSON object per line.
                            # We wrap it in SSE format and yield bytes to prevent Uvicorn buffering
                            yield f"data: {line}\n\n".encode("utf-8")
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n".encode("utf-8")

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/api/unload")
async def unload_model(request: Dict[str, str]):
    model_name = request.get("model")
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name required")
    
    try:
        async with httpx.AsyncClient() as client:
            # Sending keep_alive: 0 unloads the model immediately after completion
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model_name,
                    "messages": [],
                    "stream": False,
                    "keep_alive": 0
                }
            )
            resp.raise_for_status()
            return {"status": "unloaded", "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

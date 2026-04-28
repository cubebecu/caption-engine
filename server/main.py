"""
caption-engine: FastAPI service for image captioning with Gemma-3-4B multimodal.

Embeds a Web UI with:
  - Tab 1: Caption (image upload + paste)
  - Tab 2: Configuration (system prompt, output format)

GPU enforcement: requires >=4GB VRAM, refuses CPU fallback.
"""

import asyncio
import base64
import io
import json
import logging
import logging.handlers
import mimetypes
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel
import aiohttp

logger = logging.getLogger("caption-engine")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/gemma-3-4b-it/gemma-3-4b-it-Q4_K_M.gguf")
MMPROJ_PATH = os.getenv("MMPROJ_PATH", "/app/models/gemma-3-4b-it/mmproj-model-f16.gguf")
LLAMA_API_PORT = int(os.getenv("LLAMA_API_PORT", "8080"))
LLAMA_HOST = f"http://127.0.0.1:{LLAMA_API_PORT}"
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "32"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "512"))
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", "4096"))
WORKERS = int(os.getenv("WORKERS", "1"))

# Default system prompt for technical documentation
_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent.parent / "system_prompt_default.md"
with _DEFAULT_PROMPT_PATH.open("r", encoding="utf-8") as _f:
    DEFAULT_SYSTEM_PROMPT = _f.read()

# Custom prompt file — loaded on startup, saved on config update
_CUSTOM_PROMPT_PATH = Path(__file__).resolve().parent.parent / "system_prompt_custom.md"

SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
if _CUSTOM_PROMPT_PATH.exists():
    try:
        SYSTEM_PROMPT = _CUSTOM_PROMPT_PATH.read_text(encoding="utf-8")
        logger.info("Loaded custom system prompt from %s", _CUSTOM_PROMPT_PATH)
    except Exception as e:
        logger.error("Failed to load custom prompt, using default: %s", e)
else:
    logger.info("No custom prompt found, using default")

# ── Job storage ──
JOBS_DIR = Path(os.getenv("JOBS_DIR", "/app/jobs"))

# ── Image validation ──
MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "16"))
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp"}
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "16384"))  # prevent zip bombs

# ── Batch processing ──
BATCH_MAX_IMAGES = int(os.getenv("BATCH_MAX_IMAGES", "100"))
BATCH_MAX_IMAGE_SIZE_MB = int(os.getenv("BATCH_MAX_IMAGE_SIZE_MB", "8"))

# ── Thumbnail ──
THUMBNAIL_MAX_SIZE = (320, 240)

# ── Log file ──
LOG_FILE = Path(os.getenv("LOG_FILE", "./logs/server.log"))
LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10 MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job registry: job_id → job metadata
_jobs: dict[str, dict] = {}


def _load_jobs():
    """Scan JOBS_DIR and rebuild _jobs from existing directories on disk."""
    if not JOBS_DIR.exists():
        return
    loaded = 0
    for job_dir in sorted(JOBS_DIR.iterdir()):
        if not job_dir.is_dir():
            continue
        job_id = job_dir.name
        md_path = next(job_dir.glob("*.md"), None)
        img_path = next(job_dir.glob("image*"), None)
        thumb_path = job_dir / "thumb.jpg"
        if not md_path:
            logger.warning("Job dir without .md file, skipping: %s", job_id)
            continue
        caption = md_path.read_text(encoding="utf-8")
        img_size = {}
        if img_path and img_path.exists():
            try:
                img = Image.open(img_path)
                img_size = {"width": img.width, "height": img.height}
            except Exception:
                pass
        _jobs[job_id] = {
            "id": job_id,
            "filename": md_path.name,
            "image_path": str(img_path) if img_path else "",
            "thumb_path": str(thumb_path) if thumb_path.exists() else "",
            "md_path": str(md_path),
            "caption": caption,
            "model": "unknown",
            "processing_time_ms": 0.0,
            "image_size": img_size,
            "created_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(job_dir.stat().st_mtime)),
        }
        loaded += 1
    logger.info("Loaded %d jobs from disk", loaded)


class JobEntry(BaseModel):
    id: str
    filename: str
    image_path: str
    thumb_path: str = ""
    md_path: str
    caption: str
    model: str
    processing_time_ms: float
    image_size: dict[str, int]
    created_at: str


class JobListResponse(BaseModel):
    jobs: list[JobEntry]


class JobDeleteResponse(BaseModel):
    deleted: bool
    message: str

# ---------------------------------------------------------------------------
# GPU enforcement
# ---------------------------------------------------------------------------

def check_gpu_requirements():
    """Verify GPU is available with >=4GB VRAM. Exit if not."""
    try:
        import torch
    except ImportError:
        # torch not installed — try nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print("FATAL: nvidia-smi failed — GPU not available.", file=sys.stderr)
                sys.exit(1)

            gpus = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            if not gpus:
                print("FATAL: No GPUs detected.", file=sys.stderr)
                sys.exit(1)

            total_vram = sum(int(line.split()[0]) for line in gpus)
            if total_vram < 4096:
                print(
                    f"FATAL: Insufficient VRAM. Total: {total_vram}MB, "
                    f"required: 4096MB (4GB). GPU-only mode enforced.",
                    file=sys.stderr,
                )
                sys.exit(1)

            print(f"✅ GPU check passed: {total_vram}MB total VRAM")
            return
        except FileNotFoundError:
            print(
                "FATAL: Neither torch nor nvidia-smi available. "
                "GPU-only mode cannot be enforced.",
                file=sys.stderr,
            )
            sys.exit(1)

    # torch path: check per-GPU VRAM
    try:
        gpu_count = torch.cuda.device_count()
    except AssertionError:
        gpu_count = 0

    if gpu_count == 0:
        # torch has no CUDA or no GPUs — fall back to nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                gpus = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
                total_vram = sum(int(line.split()[0]) for line in gpus)
                if total_vram < 4096:
                    print(
                        f"FATAL: Insufficient VRAM. Total: {total_vram}MB, "
                        f"required: 4096MB (4GB). GPU-only mode enforced.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                print(f"✅ GPU check passed (nvidia-smi fallback): {total_vram}MB total VRAM")
                return
        except FileNotFoundError:
            pass
        print("FATAL: No CUDA GPUs detected. GPU-only mode enforced.", file=sys.stderr)
        sys.exit(1)

    total_vram = 0
    for i in range(gpu_count):
        vram = torch.cuda.get_device_properties(i).total_mem // (1024 ** 2)
        total_vram += vram
        print(f"  GPU {i}: {vram}MB VRAM")

    if total_vram < 4096:
        print(
            f"FATAL: Insufficient VRAM. Total: {total_vram}MB, "
            f"required: 4096MB (4GB). GPU-only mode enforced.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"✅ GPU check passed: {total_vram}MB total VRAM across {gpu_count} GPU(s)")


# ── Job management ──

def _save_job(image_bytes: bytes, caption_text: str, model_name: str,
              processing_time_ms: float, img_size: dict, filename: str) -> str:
    """Save a job's image, thumbnail, and caption, return job_id."""
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save original image
    img_path = job_dir / "image"
    img_ext = Path(filename).suffix or ".png"
    img_path = img_path.with_suffix(img_ext)
    img_path.write_bytes(image_bytes)

    # Generate and save thumbnail
    thumb_path = job_dir / "thumb.jpg"
    try:
        thumb_bytes = _generate_thumbnail(image_bytes)
        thumb_path.write_bytes(thumb_bytes)
    except Exception as e:
        logger.warning("Thumbnail generation failed for job %s: %s", job_id, e)

    # Save markdown
    md_name = Path(filename).stem + ".md"
    md_path = job_dir / md_name
    md_path.write_text(caption_text, encoding="utf-8")

    _jobs[job_id] = {
        "id": job_id,
        "filename": md_name,
        "image_path": str(img_path),
        "thumb_path": str(thumb_path),
        "md_path": str(md_path),
        "caption": caption_text,
        "model": model_name,
        "processing_time_ms": processing_time_ms,
        "image_size": img_size,
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    logger.info("Job saved: id=%s file=%s size=%d bytes", job_id, md_name, len(image_bytes))
    return job_id


def _delete_job(job_id: str) -> bool:
    """Delete a job and its files. Returns True if deleted."""
    entry = _jobs.pop(job_id, None)
    if not entry:
        return False
    job_dir = JOBS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    logger.info("Job deleted: id=%s", job_id)
    return True


def _get_job_entry(job_id: str) -> JobEntry:
    """Get a JobEntry for a job_id."""
    entry = _jobs[job_id]
    return JobEntry(
        id=entry["id"],
        filename=entry["filename"],
        image_path=entry["image_path"],
        thumb_path=entry.get("thumb_path", ""),
        md_path=entry["md_path"],
        caption=entry["caption"],
        model=entry["model"],
        processing_time_ms=entry["processing_time_ms"],
        image_size=entry["image_size"],
        created_at=entry["created_at"],
    )


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

def _validate_image(image_bytes: bytes, filename: str | None) -> None:
    """Validate an uploaded image before processing.

    Checks:
      1. File size <= MAX_IMAGE_SIZE_MB
      2. Extension in ALLOWED_IMAGE_EXTENSIONS
      3. Pillow can open it (not a corrupted/fake file)
      4. Dimensions <= MAX_IMAGE_DIMENSION (prevent zip bombs)

    Raises HTTPException on any failure.
    """
    # 1. Size check
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large: {size_mb:.1f} MB. Maximum: {MAX_IMAGE_SIZE_MB} MB",
        )

    # 2. Extension check
    ext = ""
    if filename:
        ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        # Fallback: try MIME type detection
        guessed_type = mimetypes.guess_type(filename or "")[0]
        if not guessed_type or not guessed_type.startswith("image/"):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported image format: '{ext}'. Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}",
            )

    # 3. Pillow can open it
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # actually parse, don't just read header
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image: {e}",
        )

    # 4. Dimension check (reopen after verify)
    img = Image.open(io.BytesIO(image_bytes))
    if img.width > MAX_IMAGE_DIMENSION or img.height > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions ({img.width}x{img.height}) exceed maximum ({MAX_IMAGE_DIMENSION}px)",
        )


# ---------------------------------------------------------------------------
# Thumbnail
# ---------------------------------------------------------------------------

def _generate_thumbnail(image_bytes: bytes) -> bytes:
    """Generate a JPEG thumbnail from image bytes."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "RGBX", "PA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "PA":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    img.thumbnail(THUMBNAIL_MAX_SIZE, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Llama-server health monitor
# ---------------------------------------------------------------------------

_llama_healthy: bool = True
_llama_restart_count: int = 0
LLAMA_MAX_RESTARTS = int(os.getenv("LLAMA_MAX_RESTARTS", "5"))
LLAMA_RESTART_COOLDOWN = int(os.getenv("LLAMA_RESTART_COOLDOWN", "30"))  # seconds


async def _check_llama_health() -> bool:
    """Check if llama-server is responding."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{LLAMA_HOST}/v1/models",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                return resp.status == 200
    except Exception:
        return False


async def _restart_llama_server() -> bool:
    """Attempt to restart llama-server. Returns True on success."""
    global _llama_healthy, _llama_restart_count
    _llama_restart_count += 1
    logger.warning(
        "llama-server health check failed, attempting restart (%d/%d)",
        _llama_restart_count,
        LLAMA_MAX_RESTARTS,
    )

    if _llama_restart_count > LLAMA_MAX_RESTARTS:
        logger.critical(
            "llama-server exceeded max restarts (%d), giving up",
            LLAMA_MAX_RESTARTS,
        )
        _llama_healthy = False
        return False

    await _stop_llama_server()
    try:
        await _start_llama_server()
        _llama_healthy = True
        logger.info("llama-server restarted successfully")
        return True
    except Exception as e:
        logger.error("llama-server restart failed: %s", e)
        _llama_healthy = False
        return False


async def _llama_health_monitor():
    """Background task: periodically check llama-server health and restart if needed."""
    global _llama_healthy, _llama_restart_count
    check_interval = int(os.getenv("LLAMA_HEALTH_INTERVAL", "15"))  # seconds

    while True:
        await asyncio.sleep(check_interval)

        if _llama_healthy:
            if not await _check_llama_health():
                logger.warning("llama-server health check failed")
                _llama_healthy = False
                await _restart_llama_server()
        else:
            # Already unhealthy, try periodic recovery
            if await _check_llama_health():
                logger.info("llama-server recovered")
                _llama_healthy = True
                _llama_restart_count = 0  # reset counter on recovery


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

_llama_process: Optional[subprocess.Popen] = None


async def _start_llama_server():
    """Start llama-server as a subprocess with GPU-only settings."""
    global _llama_process

    # Find llama-server binary (works in both build-from-source and ghcr.io images)
    candidates = [
        "./llama-server",
        "/opt/llama.cpp/build/bin/llama-server",
        "/app/llama-server",
        "/usr/local/bin/llama-server",
    ]
    server_bin = next((p for p in candidates if Path(p).exists()), None)
    if not server_bin:
        logger.error("llama-server binary not found in: %s", candidates)
        raise RuntimeError("llama-server binary not found")

    cmd = [
        server_bin,
        "-m", MODEL_PATH,
        "--mmproj", MMPROJ_PATH,
        "--host", "127.0.0.1",
        "--port", str(LLAMA_API_PORT),
        "--ctx-size", str(CONTEXT_SIZE),
        "--gpu-layers", str(GPU_LAYERS),
        "-b", str(BATCH_SIZE),
        "-t", str(WORKERS),
        "--log-disable",
        "--flash-attn", "auto",     # flash attention for VRAM efficiency
    ]

    logger.info("Starting llama-server: %s", " ".join(cmd))
    _llama_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    logger.info("llama-server started with PID: %d", _llama_process.pid)

    # Read output lines as they come
    import threading
    def drain_output():
        for line in iter(_llama_process.stdout.readline, ''):
            logger.debug("llama-server: %s", line.rstrip())
    threading.Thread(target=drain_output, daemon=True).start()

    # Wait for server to be ready (aiohttp health check)
    for attempt in range(180):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{LLAMA_HOST}/v1/models") as resp:
                    if resp.status == 200:
                        logger.info("llama-server is ready (attempt %d/180)", attempt + 1)
                        return
        except Exception as e:
            logger.debug("Waiting for llama-server (attempt %d/180): %s", attempt + 1, str(e))
            pass
        await asyncio.sleep(1)

    logger.error("llama-server did not become ready after 180 attempts")
    raise RuntimeError("llama-server did not become ready in 180s")


async def _stop_llama_server():
    global _llama_process
    if _llama_process and _llama_process.poll() is None:
        logger.info("Stopping llama-server (PID: %d)", _llama_process.pid)
        _llama_process.send_signal(signal.SIGTERM)
        try:
            _llama_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server didn't stop gracefully, killing")
            _llama_process.kill()
    logger.info("llama-server stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # GPU check at startup
    logger.info("=== Starting caption-engine ===")
    logger.info("MODEL_PATH=%s MMPROJ_PATH=%s GPU_LAYERS=%d", MODEL_PATH, MMPROJ_PATH, GPU_LAYERS)
    _load_jobs()
    check_gpu_requirements()
    await _start_llama_server()

    # Start background health monitor
    monitor_task = asyncio.create_task(_llama_health_monitor())
    logger.info("Health monitor started (interval=%ds)", int(os.getenv("LLAMA_HEALTH_INTERVAL", "15")))

    logger.info("caption-engine ready on port 8000")
    yield
    logger.info("=== Shutting down caption-engine ===")
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass
    await _stop_llama_server()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="caption-engine",
    description="Image captioning with Gemma-3-4B multimodal via llama.cpp",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Logging middleware
# ---------------------------------------------------------------------------

async def request_logging_middleware(request: Request, call_next):
    """Log all requests with method, path, status code, and duration."""
    start_time = time.perf_counter()
    logger.info("REQUEST %s %s", request.method, request.url.path)

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info("RESPONSE %s %s -> %d (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
        return response
    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error("ERROR %s %s -> FAILED (%.1fms): %s", request.method, request.url.path, duration_ms, str(e))
        raise


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Log HTTP exceptions."""
    logger.warning("HTTP %d: %s - %s", exc.status_code, exc.detail, request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "type": "http_error", "code": exc.status_code}},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Log and return unhandled exceptions."""
    logger.error("UNHANDLED ERROR: %s - %s - Path: %s", type(exc).__name__, str(exc), request.url.path, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error", "type": "internal_error", "code": 500}},
    )

app.middleware("http")(request_logging_middleware)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _image_to_base64(image: bytes) -> str:
    """Resize oversized images and encode as base64."""
    img = Image.open(io.BytesIO(image))

    # Convert RGBA/PA/P modes to RGB (JPEG doesn't support alpha)
    if img.mode in ("RGBA", "RGBX", "PA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "PA":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1])  # use last alpha channel
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Limit to 2048px on the longest side for inference efficiency
    if max(img.size) > 2048:
        ratio = 2048 / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class CaptionResponse(BaseModel):
    caption: str
    model: str
    processing_time_ms: float
    image_size: dict[str, int]
    job_id: str


class HealthResponse(BaseModel):
    status: str
    model: str
    gpu_layers: int
    context_size: int
    workers: int
    batch_size: int


class ConfigResponse(BaseModel):
    system_prompt: str
    model_path: str
    gpu_layers: int
    context_size: int
    workers: int
    batch_size: int


class ConfigUpdate(BaseModel):
    system_prompt: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    logger.info("Web UI requested")
    return HTMLResponse(content=WEB_UI_HTML)


async def _process_single_image(image_bytes: bytes, filename: str, prompt: str) -> CaptionResponse:
    """Process a single image through llama-server and save a job.

    Returns CaptionResponse on success.
    Raises HTTPException on validation or inference failure.
    """
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    _validate_image(image_bytes, filename)

    img = Image.open(io.BytesIO(image_bytes))
    img_size = {"width": img.width, "height": img.height}
    logger.info("Processing image: %s (%dx%d, %d bytes)", filename, img_size["width"], img_size["height"], len(image_bytes))

    b64 = await _image_to_base64(image_bytes)

    start = time.time()

    payload = {
        "model": MODEL_PATH,
        "messages": [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        },
                    },
                    {"type": "text", "text": "Describe this image in detail. Focus on technical content."},
                ],
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.3,
        "stream": False,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{LLAMA_HOST}/v1/chat/completions", json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("llama-server error HTTP %d: %s", resp.status, body[:500])
                    raise HTTPException(status_code=502, detail=f"llama.cpp error (HTTP {resp.status}): {body}")
                result = await resp.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("llama-server request failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=502, detail=f"llama.cpp error: {e}")

    elapsed_ms = (time.time() - start) * 1000
    caption_text = result["choices"][0]["message"]["content"]
    model_name = result.get("model") or "Gemma-3-4B"

    logger.info("Caption generated: %s tokens, %.1fms, model=%s", len(caption_text.split()), elapsed_ms, model_name)

    # Save job
    job_id = _save_job(
        image_bytes=image_bytes,
        caption_text=caption_text,
        model_name=model_name,
        processing_time_ms=round(elapsed_ms, 1),
        img_size=img_size,
        filename=filename or "image.png",
    )

    return CaptionResponse(
        caption=caption_text,
        model=model_name,
        processing_time_ms=round(elapsed_ms, 1),
        image_size=img_size,
        job_id=job_id,
    )


@app.post("/caption", response_model=CaptionResponse)
async def caption(
    image: UploadFile = File(...),
    system_prompt: Optional[str] = Form(None),
):
    """Generate a caption from an image."""
    prompt = system_prompt or SYSTEM_PROMPT

    # ── Health gate: reject if llama-server is down ──
    if not _llama_healthy:
        raise HTTPException(
            status_code=503,
            detail="Vision model is unavailable (restarting). Retry in a few seconds.",
        )

    content = await image.read()
    return await _process_single_image(content, image.filename, prompt)


@app.post("/caption/batch")
async def caption_batch(
    images: list[UploadFile] = File(...),
    system_prompt: Optional[str] = Form(None),
):
    """Process multiple images sequentially, streaming results via SSE."""
    prompt = system_prompt or SYSTEM_PROMPT

    if not _llama_healthy:
        raise HTTPException(
            status_code=503,
            detail="Vision model is unavailable (restarting). Retry in a few seconds.",
        )

    if len(images) > BATCH_MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many images: {len(images)}. Maximum: {BATCH_MAX_IMAGES}",
        )

    async def event_generator():
        processed = 0
        failed = 0
        for idx, image in enumerate(images):
            filename = image.filename or f"image_{idx}.png"
            try:
                yield f"event: progress\ndata: {json.dumps({'current': idx + 1, 'total': len(images), 'filename': filename})}\n\n"
                content = await image.read()
                resp = await _process_single_image(content, filename, prompt)
                yield f"event: result\ndata: {json.dumps(resp.model_dump())}\n\n"
                processed += 1
            except Exception as e:
                logger.error("Batch image %d (%s) failed: %s", idx + 1, filename, str(e))
                failed += 1
                yield f"event: error\ndata: {json.dumps({'filename': filename, 'detail': str(e)})}\n\n"
        yield f"event: done\ndata: {json.dumps({'processed': processed, 'failed': failed})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    logger.info("Health check requested")
    return HealthResponse(
        status="ok",
        model=MODEL_PATH,
        gpu_layers=GPU_LAYERS,
        context_size=CONTEXT_SIZE,
        workers=WORKERS,
        batch_size=BATCH_SIZE,
    )


@app.get("/health/model")
async def health_model():
    """Check if llama-server (vision model) is reachable."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{LLAMA_HOST}/v1/models", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                return {"status": "ok" if resp.status == 200 else "error"}
    except Exception as e:
        logger.debug("Model health check failed: %s", str(e))
        return {"status": "error"}


@app.get("/config")
async def get_config():
    logger.info("Config requested")
    return ConfigResponse(
        system_prompt=SYSTEM_PROMPT,
        model_path=MODEL_PATH,
        gpu_layers=GPU_LAYERS,
        context_size=CONTEXT_SIZE,
        workers=WORKERS,
        batch_size=BATCH_SIZE,
    )


@app.get("/config/default")
async def get_default_config():
    """Return the default system prompt."""
    logger.info("Default config requested")
    return {"system_prompt": DEFAULT_SYSTEM_PROMPT}


@app.delete("/config")
async def reset_config():
    """Delete custom prompt and return defaults."""
    global SYSTEM_PROMPT
    if _CUSTOM_PROMPT_PATH.exists():
        _CUSTOM_PROMPT_PATH.unlink()
        logger.info("Custom system prompt deleted, reverting to default")
    SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
    return ConfigResponse(
        system_prompt=SYSTEM_PROMPT,
        model_path=MODEL_PATH,
        gpu_layers=GPU_LAYERS,
        context_size=CONTEXT_SIZE,
        workers=WORKERS,
        batch_size=BATCH_SIZE,
    )


@app.get("/logs")
async def get_logs(lines: int = 200):
    """Return the last N lines of the application log."""
    log_file = LOG_FILE
    if not log_file.exists():
        return {"lines": []}
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        tail = [l.rstrip("\n") for l in all_lines[-lines:]]
        return {"lines": tail, "total": len(all_lines), "shown": len(tail)}
    except Exception as e:
        logger.error("Failed to read logs: %s", str(e))
        return {"lines": [], "total": 0, "shown": 0, "error": str(e)}


@app.put("/config")
async def update_config(config: ConfigUpdate):
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = config.system_prompt

    # Persist custom prompt to disk so it survives restarts
    try:
        _CUSTOM_PROMPT_PATH.write_text(config.system_prompt, encoding="utf-8")
        logger.info("Custom system prompt saved to %s", _CUSTOM_PROMPT_PATH)
    except Exception as e:
        logger.error("Failed to save custom prompt to disk: %s", e)

    logger.info("Config updated: system_prompt changed (length=%d)", len(config.system_prompt))
    return ConfigResponse(
        system_prompt=SYSTEM_PROMPT,
        model_path=MODEL_PATH,
        gpu_layers=GPU_LAYERS,
        context_size=CONTEXT_SIZE,
        workers=WORKERS,
        batch_size=BATCH_SIZE,
    )


# ── Job endpoints ──

@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    jobs = [_get_job_entry(jid) for jid in sorted(_jobs.keys(), reverse=True)]
    logger.info("Jobs listed: %d total", len(jobs))
    return {"jobs": jobs}


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its files."""
    if job_id not in _jobs:
        logger.warning("Delete job not found: %s", job_id)
        raise HTTPException(status_code=404, detail="Job not found")
    _delete_job(job_id)
    logger.info("Job deleted via API: id=%s", job_id)
    return {"deleted": True, "message": "Job deleted"}


@app.get("/jobs/{job_id}/image")
async def download_job_image(job_id: str):
    """Download the image for a job."""
    entry = _jobs.get(job_id)
    if not entry:
        logger.warning("Image download not found: %s", job_id)
        raise HTTPException(status_code=404, detail="Job not found")
    img_path = Path(entry["image_path"])
    if not img_path.exists():
        logger.error("Image file not found: %s", img_path)
        raise HTTPException(status_code=404, detail="Image file not found")

    logger.info("Image downloaded: job_id=%s file=%s", job_id, img_path.name)

    async def file_iterator():
        async with aiofiles.open(img_path, "rb") as f:
            while chunk := await f.read(8192):
                yield chunk

    return StreamingResponse(
        file_iterator(),
        media_type="image/jpeg",
        headers={"Content-Disposition": f'attachment; filename="{img_path.name}"'},
    )


@app.get("/jobs/{job_id}/thumbnail")
async def get_job_thumbnail(job_id: str):
    """Get a thumbnail for a job's image."""
    entry = _jobs.get(job_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Job not found")

    thumb_path = Path(entry.get("thumb_path", ""))
    if not thumb_path.exists():
        # Fallback: generate on-the-fly from original image
        img_path = Path(entry.get("image_path", ""))
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")
        try:
            thumb_bytes = _generate_thumbnail(img_path.read_bytes())
        except Exception as e:
            logger.error("Thumbnail generation failed: %s", e)
            raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
        return StreamingResponse(
            io.BytesIO(thumb_bytes),
            media_type="image/jpeg",
        )

    async def file_iterator():
        async with aiofiles.open(thumb_path, "rb") as f:
            while chunk := await f.read(8192):
                yield chunk

    return StreamingResponse(
        file_iterator(),
        media_type="image/jpeg",
    )


@app.get("/jobs/{job_id}/caption.md")
async def download_job_caption(job_id: str):
    """Download the markdown caption for a job."""
    entry = _jobs.get(job_id)
    if not entry:
        logger.warning("Caption download not found: %s", job_id)
        raise HTTPException(status_code=404, detail="Job not found")
    md_path = Path(entry["md_path"])
    if not md_path.exists():
        logger.error("Caption file not found: %s", md_path)
        raise HTTPException(status_code=404, detail="Caption file not found")

    logger.info("Caption downloaded: job_id=%s file=%s", job_id, md_path.name)

    async def file_iterator():
        async with aiofiles.open(md_path, "rb") as f:
            while chunk := await f.read(8192):
                yield chunk

    return StreamingResponse(
        file_iterator(),
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{md_path.name}"'},
    )


# ---------------------------------------------------------------------------
# Web UI (embedded)
# ---------------------------------------------------------------------------

WEB_UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Image caption creator</title>
<style>
  :root {
    --bg: #0a0c10;
    --surface: #131620;
    --surface2: #1a1e2b;
    --surface3: #222736;
    --border: #262b3a;
    --border-hover: #3a4060;
    --text: #d8dbe6;
    --text-dim: #6b7194;
    --text-bright: #f0f2fa;
    --accent: #6c8cff;
    --accent-dim: rgba(108,140,255,0.12);
    --accent-glow: rgba(108,140,255,0.25);
    --green: #34d399;
    --green-dim: rgba(52,211,153,0.15);
    --red: #f87171;
    --red-dim: rgba(248,113,113,0.12);
    --radius: 12px;
    --radius-sm: 8px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    height: 56px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--text-bright);
  }
  .logo-icon {
    width: 28px; height: 28px;
    background: var(--accent);
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
  }
  .logo span { color: var(--accent); }
  .status-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 14px;
    border-radius: 20px;
    background: var(--green-dim);
    font-size: 12px;
    font-weight: 600;
    color: var(--green);
  }
  .status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 6px var(--green);
  }
  .status-dot.error { background: var(--red); box-shadow: 0 0 6px var(--red); }
  .tab-bar {
    display: flex;
    gap: 0;
    padding: 0 28px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }
  .tab {
    padding: 14px 22px;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-dim);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.15s;
    user-select: none;
    display: flex;
    align-items: center;
    gap: 7px;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .tab-badge {
    background: var(--surface3);
    color: var(--text-dim);
    font-size: 11px;
    padding: 1px 7px;
    border-radius: 10px;
    font-weight: 600;
  }
  .tab.active .tab-badge { background: var(--accent-dim); color: var(--accent); }
  .page { padding: 24px 28px; flex: 1; overflow: auto; }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .card:hover { border-color: var(--border-hover); }
  .card-head {
    padding: 14px 18px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    background: var(--surface2);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .card-body { padding: 18px; flex: 1; }
  .input-layout {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    min-height: 500px;
  }
  .dropzone {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 32px 24px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    min-height: 280px;
    justify-content: center;
  }
  .dropzone:hover { border-color: var(--accent); background: var(--accent-dim); }
  .dropzone-icon { font-size: 48px; opacity: 0.4; }
  .dropzone-title { font-size: 14px; font-weight: 600; color: var(--text); }
  .dropzone-sub { font-size: 13px; color: var(--text-dim); }
  .dropzone-sub strong { color: var(--accent); }
  .paste-box {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 20px;
    text-align: center;
    color: var(--text-dim);
    font-size: 13px;
    background: var(--surface2);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    min-height: 56px;
  }
  kbd {
    background: var(--surface3);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 2px 7px;
    font-size: 11px;
    font-family: inherit;
    color: var(--text);
    box-shadow: 0 1px 0 var(--border);
  }
  .file-select-btn {
    padding: 8px 18px;
    border-radius: var(--radius-sm);
    background: var(--accent-dim);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    font-family: inherit;
  }
  .file-select-btn:hover { background: var(--accent-glow); }
  .preview-wrap {
    display: none;
    flex-direction: column;
    gap: 12px;
  }
  .preview-wrap.visible { display: flex; }
  .preview-img-wrap {
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--surface2);
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 180px;
  }
  .preview-img-wrap img { max-width: 100%; max-height: 280px; object-fit: contain; }
  .preview-meta {
    display: flex;
    gap: 14px;
    font-size: 12px;
    color: var(--text-dim);
  }
  .preview-meta span { display: flex; align-items: center; gap: 4px; }
  .btn {
    padding: 10px 20px;
    border: none;
    border-radius: var(--radius-sm);
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: inherit;
  }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover { background: #5a7af0; }
  .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-ghost {
    background: transparent;
    color: var(--text-dim);
    border: 1px solid var(--border);
  }
  .btn-ghost:hover { border-color: var(--text-dim); color: var(--text); }
  .btn-sm { padding: 5px 10px; font-size: 12px; border-radius: 6px; }
  .btn-danger-sm {
    padding: 4px 8px;
    font-size: 11px;
    border-radius: 6px;
    background: var(--red-dim);
    border: 1px solid transparent;
    color: var(--red);
    cursor: pointer;
    font-family: inherit;
    font-weight: 600;
    transition: all 0.15s;
  }
  .btn-danger-sm:hover { border-color: var(--red); }
  .btn-row { display: flex; gap: 8px; margin-top: 12px; }
  .output {
    font-size: 14px;
    line-height: 1.75;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text);
    min-height: 150px;
    flex: 1;
    overflow: auto;
  }
  .output-placeholder { color: var(--text-dim); font-style: italic; }
  .output-actions {
    display: flex;
    gap: 8px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
  }
  .output-tags {
    display: flex;
    gap: 12px;
    margin-top: 12px;
    font-size: 11px;
    color: var(--text-dim);
  }
  .output-tag {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    background: var(--surface2);
    border-radius: 6px;
  }
  /* Results */
  .results-page { max-width: 1200px; }
  .results-empty {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-dim);
  }
  .results-empty-icon { font-size: 48px; opacity: 0.3; margin-bottom: 12px; }
  .results-empty-text { font-size: 15px; font-weight: 600; margin-bottom: 4px; color: var(--text); }
  .results-empty-sub { font-size: 13px; }
  .job-list { display: flex; flex-direction: column; gap: 12px; }
  .job-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    display: flex;
    overflow: hidden;
    transition: border-color 0.15s;
  }
  .job-card:hover { border-color: var(--border-hover); }
  .job-thumb {
    width: 160px;
    min-height: 120px;
    background: var(--surface2);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    overflow: hidden;
    position: relative;
  }
  .job-thumb img { max-width: 100%; max-height: 100%; object-fit: contain; }
  .job-thumb-overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 6px 8px;
    background: linear-gradient(transparent, rgba(0,0,0,0.7));
    font-size: 10px;
    color: var(--text-dim);
    display: flex;
    gap: 6px;
  }
  .job-thumb-overlay span { display: flex; align-items: center; gap: 3px; }
  .job-body { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  .job-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  .job-name {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-bright);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 400px;
  }
  .job-actions { display: flex; gap: 6px; flex-shrink: 0; }
  .job-meta {
    padding: 8px 16px;
    font-size: 11px;
    color: var(--text-dim);
    display: flex;
    gap: 16px;
    background: var(--surface2);
  }
  .job-meta span { display: flex; align-items: center; gap: 4px; }
  .job-preview {
    flex: 1;
    padding: 14px 16px;
    overflow: auto;
    font-size: 13px;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
    color: var(--text);
    min-height: 80px;
    max-height: 200px;
  }
  /* Config */
  .config-page { max-width: 720px; }
  .config-section-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-bright);
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 4px;
  }
  .field { display: flex; flex-direction: column; gap: 8px; }
  .field-label {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-dim);
  }
  .field textarea {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px;
    color: var(--text);
    font-size: 13px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    line-height: 1.65;
    resize: vertical;
    min-height: 360px;
    outline: none;
    transition: border-color 0.2s;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .field textarea:focus { border-color: var(--accent); }
  .field-hint { font-size: 12px; color: var(--text-dim); }
  .info-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 16px;
    font-size: 13px;
    line-height: 1.7;
    color: var(--text-dim);
  }
  .info-card strong { color: var(--text); }
  .info-card code {
    background: var(--surface3);
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 12px;
    color: var(--accent);
  }
  .format-table {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    overflow: hidden;
  }
  .format-row {
    display: flex;
    border-bottom: 1px solid var(--border);
  }
  .format-row:last-child { border-bottom: none; }
  .format-header {
    background: var(--surface3);
    font-weight: 600;
    color: var(--text-bright);
  }
  .format-col { padding: 10px 14px; font-size: 13px; }
  .format-name { font-weight: 500; color: var(--text); width: 80px; }
  .format-status { text-align: center; font-size: 15px; }
  .format-ok { color: var(--green); }
  .format-no { color: var(--red); }
  .toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    padding: 12px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    z-index: 1000;
    transform: translateY(100px);
    opacity: 0;
    transition: all 0.3s ease;
  }
  .toast.show { transform: translateY(0); opacity: 1; }
  .toast.success { background: var(--green); color: #000; }
  .toast.error { background: var(--red); color: #000; }
  .spinner {
    width: 16px; height: 16px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.6s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  /* Batch queue */
  .batch-queue { display: none; flex-direction: column; gap: 8px; }
  .batch-queue.visible { display: flex; }
  .batch-header { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; }
  .batch-count { font-size: 13px; font-weight: 600; color: var(--text); }
  .batch-count .ok { color: var(--green); }
  .batch-count .skip { color: var(--red); }
  .batch-list { max-height: 240px; overflow: auto; display: flex; flex-direction: column; gap: 4px; }
  .batch-item { display: flex; align-items: center; gap: 8px; padding: 6px 10px; background: var(--surface2); border-radius: 6px; font-size: 12px; }
  .batch-item .status { font-size: 14px; flex-shrink: 0; width: 20px; text-align: center; }
  .batch-item .name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); }
  .batch-item .size { color: var(--text-dim); flex-shrink: 0; }
  .batch-item.skipped { opacity: 0.5; }
  .batch-item.processing { background: var(--accent-dim); }
  .batch-item.done { border-left: 3px solid var(--green); }
  .batch-item.error { border-left: 3px solid var(--red); }
  .batch-progress { height: 4px; background: var(--surface3); border-radius: 2px; overflow: hidden; }
  .batch-progress-bar { height: 100%; background: var(--accent); transition: width 0.3s; width: 0%; }
  .batch-status { font-size: 12px; color: var(--text-dim); text-align: center; padding: 4px 0; }
  @media (max-width: 900px) {
    .input-layout { grid-template-columns: 1fr; }
    .job-card { flex-direction: column; }
    .job-thumb { width: 100%; min-height: 100px; max-height: 200px; }
  }
</style>
</head>
<body>
<div class="topbar">
  <div class="logo">
    <div class="logo-icon">🔍</div>
    <span>Image caption creator</span>
  </div>
  <div>
    <div class="status-badge">
      <div class="status-dot" id="model-dot"></div>
      <span id="model-text">Model</span>
    </div>
  </div>
</div>
<div class="tab-bar">
  <div class="tab active" data-tab="caption"><span class="tab-icon">📷</span> Caption</div>
  <div class="tab" data-tab="results"><span class="tab-icon">📋</span> Results <span class="tab-badge" id="results-count">0</span></div>
  <div class="tab" data-tab="logs"><span class="tab-icon">📜</span> Logs <span class="tab-badge" id="logs-count">0</span></div>
  <div class="tab" data-tab="config"><span class="tab-icon">⚙️</span> Configuration</div>
</div>
<div class="page" style="padding:0;flex:1;overflow:hidden;">
  <!-- Caption Tab -->
  <div class="tab-content active" id="tab-caption" style="display:flex;padding:24px;gap:20px;min-height:500px;">
    <div class="input-layout" style="flex:1;min-height:0;">
      <!-- Left: Input -->
      <div style="display:flex;flex-direction:column;gap:16px;">
        <div class="card">
          <div class="card-head">📁 Image Input</div>
          <div class="card-body">
            <div class="dropzone" id="dropzone">
              <div class="dropzone-icon">🖼️</div>
              <div class="dropzone-title">Drop image here</div>
              <div class="dropzone-sub">or <strong>click to browse</strong></div>
            </div>
            <input type="file" id="file-input" accept="image/png,image/jpeg,image/gif,image/bmp" multiple style="display:none">
            <input type="file" id="folder-input" webkitdirectory multiple style="display:none">
            <div class="paste-box">💡 Paste with <kbd>Ctrl</kbd>+<kbd>V</kbd></div>
            <div class="btn-row">
              <button class="file-select-btn" id="file-select-btn">📂 Select files</button>
              <button class="file-select-btn" id="folder-select-btn">📁 Select folder</button>
            </div>
            <!-- Batch queue panel -->
            <div class="batch-queue" id="batch-queue">
              <div class="batch-header">
                <div class="batch-count" id="batch-count"></div>
                <div class="btn-row" style="margin-top:0;">
                  <button class="btn btn-primary btn-sm" id="batch-process-btn" disabled>🚀 Process All</button>
                  <button class="btn btn-ghost btn-sm" id="batch-clear-btn">✕ Clear</button>
                </div>
              </div>
              <div class="batch-progress"><div class="batch-progress-bar" id="batch-progress-bar"></div></div>
              <div class="batch-status" id="batch-status"></div>
              <div class="batch-list" id="batch-list"></div>
            </div>
            <div class="preview-wrap" id="preview-container">
              <div class="preview-img-wrap">
                <img id="preview-image" alt="Preview">
              </div>
              <div class="preview-meta">
                <span id="preview-filename"></span>
                <span id="preview-size"></span>
                <span id="preview-dims"></span>
              </div>
              <div class="btn-row">
                <button class="btn btn-primary" id="caption-btn" disabled>🚀 Generate Caption</button>
                <button class="btn btn-ghost" id="clear-btn">✕ Clear</button>
              </div>
            </div>
          </div>
        </div>
        <!-- Info card -->
        <div class="card">
          <div class="card-body" style="font-size:13px;color:var(--text-dim);line-height:1.6;">
            <div style="font-weight:600;color:var(--text-bright);margin-bottom:6px;">Image caption generator for technical documentation</div>
            <div>PNG, JPEG, GIF, BMP · Single image or batch (up to 100)</div>
          </div>
        </div>
      </div>
      <!-- Right: Output -->
      <div style="display:flex;flex-direction:column;gap:16px;">
        <div class="card" style="flex:1;min-height:0;">
          <div class="card-head">
            📝 Output
            <span style="margin-left:auto;font-size:11px;text-transform:none;letter-spacing:0;font-weight:500;color:var(--accent)">Markdown</span>
          </div>
          <div class="card-body" style="display:flex;flex-direction:column;gap:0;">
            <div class="output placeholder" id="output-area">Caption will appear here...</div>
            <div class="output-actions">
              <button class="btn btn-primary" id="copy-btn" style="flex:1;justify-content:center;" disabled>📋 Copy</button>
              <button class="btn btn-ghost" id="save-btn" style="flex:1;justify-content:center;" disabled>💾 Save .md</button>
            </div>
            <div class="output-tags" id="output-meta" style="display:none">
              <div class="output-tag" id="meta-model"></div>
              <div class="output-tag" id="meta-time"></div>
              <div class="output-tag" id="meta-dims"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
  <!-- Results Tab -->
  <div class="tab-content" id="tab-results" style="display:none;padding:24px;overflow:auto;">
    <div class="results-page">
      <div class="job-list" id="job-list"></div>
      <div class="results-empty" id="results-empty">
        <div class="results-empty-icon">📭</div>
        <div class="results-empty-text">No results yet</div>
        <div class="results-empty-sub">Generate a caption to see results here</div>
      </div>
    </div>
  </div>
  <!-- Logs Tab -->
  <div class="tab-content" id="tab-logs" style="display:none;padding:24px;overflow:auto;">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
      <div style="font-size:14px;font-weight:600;color:var(--text-bright);">Application Logs</div>
      <div style="display:flex;gap:8px;align-items:center;">
        <span style="font-size:12px;color:var(--text-dim);" id="logs-total">0 lines total</span>
        <button class="btn btn-ghost btn-sm" id="logs-refresh-btn" onclick="loadLogs()">↻ Refresh</button>
        <button class="btn btn-ghost btn-sm" id="logs-download-btn" onclick="downloadLogs()">💾 Download</button>
      </div>
    </div>
    <div class="card" style="max-height:calc(100vh - 220px);overflow:auto;">
      <pre id="logs-output" style="padding:16px;font-size:12px;line-height:1.6;color:var(--text);white-space:pre;overflow-x:auto;font-family:'JetBrains Mono','Fira Code',monospace;margin:0;max-height:calc(100vh - 220px);"></pre>
    </div>
  </div>
  <!-- Config Tab -->
  <div class="tab-content" id="tab-config" style="display:none;padding:24px;overflow:auto;">
    <div class="config-page">
      <div class="config-section-title">System Prompt</div>
      <div class="field">
        <div class="field-label">System Prompt</div>
        <textarea id="config-system-prompt"></textarea>
        <div class="field-hint">This prompt is sent to the model as the system message for every caption request.</div>
      </div>
      <div class="btn-row" style="margin-top:20px;">
        <button class="btn btn-primary" id="save-config-btn">💾 Save Configuration</button>
        <button class="btn btn-ghost" id="reset-config-btn">↺ Reset to Defaults</button>
      </div>
      <div class="config-section-title" style="margin-top:32px;">Model &amp; Runtime Info</div>
      <div class="info-card" id="model-info">Loading...</div>
      <div class="config-section-title" style="margin-top:32px;">Supported Image Formats</div>
      <div class="format-table">
        <div class="format-row format-header">
          <span class="format-col format-name">Format</span>
          <span class="format-col format-status">Status</span>
        </div>
        <div class="format-row">
          <span class="format-col format-name">JPEG</span>
          <span class="format-col format-status format-ok">✓</span>
        </div>
        <div class="format-row">
          <span class="format-col format-name">PNG</span>
          <span class="format-col format-status format-ok">✓</span>
        </div>
        <div class="format-row">
          <span class="format-col format-name">GIF</span>
          <span class="format-col format-status format-ok">✓</span>
        </div>
        <div class="format-row">
          <span class="format-col format-name">BMP</span>
          <span class="format-col format-status format-ok">✓</span>
        </div>
        <div class="format-row">
          <span class="format-col format-name">WebP</span>
          <span class="format-col format-status format-no">✗</span>
        </div>
        <div class="format-row">
          <span class="format-col format-name">AVIF</span>
          <span class="format-col format-status format-no">✗</span>
        </div>
        <div class="format-row">
          <span class="format-col format-name">TIFF</span>
          <span class="format-col format-status format-no">✗</span>
        </div>
        <div class="format-row">
          <span class="format-col format-name">SVG</span>
          <span class="format-col format-status format-no">✗</span>
        </div>
      </div>
    </div>
  </div>
</div>
<div class="toast" id="toast"></div>
<script>
let currentImage = null;
let currentImageName = '';
let lastCaption = '';
let jobs = {};
let batchFiles = [];
let batchAbort = null;

// DOM refs
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const folderInput = document.getElementById('folder-input');
const previewContainer = document.getElementById('preview-container');
const previewImage = document.getElementById('preview-image');
const previewFilename = document.getElementById('preview-filename');
const previewSize = document.getElementById('preview-size');
const previewDims = document.getElementById('preview-dims');
const captionBtn = document.getElementById('caption-btn');
const copyBtn = document.getElementById('copy-btn');
const saveBtn = document.getElementById('save-btn');
const outputArea = document.getElementById('output-area');
const outputMeta = document.getElementById('output-meta');
const modelDot = document.getElementById('model-dot');
const modelText = document.getElementById('model-text');
const toast = document.getElementById('toast');
const jobList = document.getElementById('job-list');
const resultsEmpty = document.getElementById('results-empty');
const resultsCount = document.getElementById('results-count');
const logsOutput = document.getElementById('logs-output');
const logsTotal = document.getElementById('logs-total');
const logsCount = document.getElementById('logs-count');
const batchQueue = document.getElementById('batch-queue');
const batchCount = document.getElementById('batch-count');
const batchProgressBar = document.getElementById('batch-progress-bar');
const batchStatus = document.getElementById('batch-status');
const batchList = document.getElementById('batch-list');
const batchProcessBtn = document.getElementById('batch-process-btn');
let lastLogsLines = [];

const ALLOWED_EXTS = {'.png': true, '.jpg': true, '.jpeg': true, '.gif': true, '.bmp': true};
const BATCH_MAX_SIZE = 8 * 1024 * 1024;

// Tabs
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => { c.style.display = 'none'; });
    tab.classList.add('active');
    const content = document.getElementById('tab-' + tab.dataset.tab);
    if (tab.dataset.tab === 'caption') { content.style.display = 'flex'; }
    else { content.style.display = 'block'; }
    if (tab.dataset.tab === 'results') loadJobs();
    if (tab.dataset.tab === 'logs') loadLogs();
  });
});

// File select
document.getElementById('file-select-btn').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFiles(fileInput.files); });

// Folder select
document.getElementById('folder-select-btn').addEventListener('click', () => folderInput.click());
folderInput.addEventListener('change', () => { if (folderInput.files.length) handleFiles(folderInput.files); });

// Drop zone
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.style.borderColor = 'var(--accent)'; dropzone.style.background = 'var(--accent-dim)'; });
dropzone.addEventListener('dragleave', () => { dropzone.style.borderColor = ''; dropzone.style.background = ''; });
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.style.borderColor = ''; dropzone.style.background = '';
  const files = e.dataTransfer.files;
  if (files.length) handleFiles(files);
});

// Paste
document.addEventListener('paste', e => {
  const items = e.clipboardData.items;
  for (const item of items) {
    if (item.type.startsWith('image/')) { handleFiles([item.getAsFile()]); break; }
  }
});

function isSupportedImage(file) {
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  return ALLOWED_EXTS[ext] && file.size <= BATCH_MAX_SIZE;
}

function getBaseName(fullPath) {
  return fullPath.replace(/^.*[\\/]/, '');
}

function handleFiles(fileList) {
  const files = Array.from(fileList).map(f => ({
    file: f,
    name: getBaseName(f.name),
    size: f.size,
    supported: isSupportedImage(f),
  }));

  if (files.length === 1 && files[0].supported) {
    handleFile(files[0].file);
    return;
  }

  batchFiles = files;
  showBatchQueue();
}

function handleFile(file) {
  currentImage = file;
  currentImageName = file.name;
  const reader = new FileReader();
  reader.onload = e => {
    previewImage.src = e.target.result;
    previewFilename.textContent = file.name;
    previewSize.textContent = formatBytes(file.size);
    const img = new Image();
    img.onload = () => { previewDims.textContent = img.width + ' × ' + img.height; };
    img.src = e.target.result;
    dropzone.style.display = 'none';
    document.querySelector('.paste-box').style.display = 'none';
    document.getElementById('file-select-btn').parentElement.style.display = 'none';
    previewContainer.classList.add('visible');
    captionBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function showBatchQueue() {
  const ok = batchFiles.filter(f => f.supported).length;
  const skip = batchFiles.length - ok;
  batchCount.innerHTML = `<span class="ok">${ok} supported</span>${skip ? ' <span class="skip">(' + skip + ' skipped)</span>' : ''}`;
  batchProgressBar.style.width = '0%';
  batchStatus.textContent = '';
  batchList.innerHTML = '';
  batchProcessBtn.disabled = false;
  batchProcessBtn.textContent = '🚀 Process All';

  batchFiles.forEach((f, i) => {
    const item = document.createElement('div');
    item.className = 'batch-item' + (f.supported ? '' : ' skipped');
    item.id = 'batch-item-' + i;
    item.innerHTML = `
      <span class="status">${f.supported ? '⏳' : '✗'}</span>
      <span class="name" title="${f.name}">${f.name}</span>
      <span class="size">${formatBytes(f.size)}</span>
    `;
    batchList.appendChild(item);
  });

  dropzone.style.display = 'none';
  document.querySelector('.paste-box').style.display = 'none';
  document.getElementById('file-select-btn').parentElement.style.display = 'none';
  previewContainer.classList.remove('visible');
  batchQueue.classList.add('visible');
}

document.getElementById('batch-clear-btn').addEventListener('click', clearBatch);

function clearBatch() {
  batchFiles = [];
  batchQueue.classList.remove('visible');
  dropzone.style.display = '';
  document.querySelector('.paste-box').style.display = '';
  document.getElementById('file-select-btn').parentElement.style.display = '';
  fileInput.value = '';
  folderInput.value = '';
}

// Batch processing with SSE
batchProcessBtn.addEventListener('click', async () => {
  const systemPrompt = document.getElementById('config-system-prompt').value;
  const formData = new FormData();
  const supported = batchFiles.filter(f => f.supported);
  supported.forEach(f => formData.append('images', f.file));

  if (!supported.length) {
    showToast('No supported images', 'error');
    return;
  }

  batchProcessBtn.disabled = true;
  batchAbort = new AbortController();

  try {
    const resp = await fetch('/caption/batch', {
      method: 'POST',
      body: formData,
      signal: batchAbort.signal,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || 'Batch failed');
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop();

      let eventType = '';
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7);
          continue;
        }
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          handleBatchEvent(eventType, data);
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      batchStatus.textContent = 'Cancelled';
      showToast('Batch cancelled', 'error');
    } else {
      batchStatus.textContent = 'Error: ' + err.message;
      showToast('Batch failed: ' + err.message, 'error');
    }
  } finally {
    batchAbort = null;
    batchProcessBtn.disabled = false;
    batchProcessBtn.textContent = '🚀 Process All';
  }
});

function handleBatchEvent(type, data) {
  if (type === 'progress') {
    const pct = Math.round((data.current / data.total) * 100);
    batchProgressBar.style.width = pct + '%';
    batchStatus.textContent = `Processing ${data.current}/${data.total} — ${data.filename}`;
    const prevItem = document.getElementById('batch-item-' + (data.current - 2));
    if (prevItem) {
      prevItem.querySelector('.status').textContent = '✓';
      prevItem.classList.add('done');
    }
    const curItem = document.getElementById('batch-item-' + (data.current - 1));
    if (curItem) {
      curItem.querySelector('.status').textContent = '⏳';
      curItem.classList.add('processing');
    }
  } else if (type === 'result') {
    lastCaption = data.caption;
    outputArea.textContent = data.caption;
    outputArea.classList.remove('placeholder');
    outputMeta.style.display = 'flex';
    copyBtn.disabled = false;
    saveBtn.disabled = false;
    document.getElementById('meta-model').textContent = '🤖 ' + data.model;
    document.getElementById('meta-time').textContent = '⏱ ' + data.processing_time_ms.toFixed(0) + 'ms';
    document.getElementById('meta-dims').textContent = '📐 ' + data.image_size.width + '×' + data.image_size.height;
  } else if (type === 'error') {
    const idx = batchFiles.findIndex(f => f.name === data.filename);
    if (idx >= 0) {
      const item = document.getElementById('batch-item-' + idx);
      if (item) {
        item.querySelector('.status').textContent = '✗';
        item.classList.add('error');
      }
    }
  } else if (type === 'done') {
    batchStatus.textContent = `Done — ${data.processed} processed, ${data.failed} failed`;
    batchProgressBar.style.width = '100%';
    showToast(`Batch complete: ${data.processed} processed, ${data.failed} failed`, data.failed ? 'error' : 'success');
    loadJobs();
  }
}

// Cancel batch
document.getElementById('batch-clear-btn').addEventListener('click', () => {
  if (batchAbort) {
    batchAbort.abort();
  } else {
    clearBatch();
  }
});

document.getElementById('clear-btn').addEventListener('click', () => {
  currentImage = null;
  currentImageName = '';
  fileInput.value = '';
  folderInput.value = '';
  dropzone.style.display = '';
  document.querySelector('.paste-box').style.display = '';
  document.getElementById('file-select-btn').parentElement.style.display = '';
  previewContainer.classList.remove('visible');
  captionBtn.disabled = true;
  outputArea.textContent = 'Caption will appear here...';
  outputArea.classList.add('placeholder');
  outputMeta.style.display = 'none';
  copyBtn.disabled = true;
  saveBtn.disabled = true;
  lastCaption = '';
  clearBatch();
});

// Caption
captionBtn.addEventListener('click', async () => {
  if (!currentImage) return;
  captionBtn.disabled = true;
  captionBtn.textContent = 'Generating...';
  outputArea.classList.add('placeholder');
  outputArea.textContent = 'Processing image...';
  const systemPrompt = document.getElementById('config-system-prompt').value;
  const formData = new FormData();
  formData.append('image', currentImage);
  if (systemPrompt) formData.append('system_prompt', systemPrompt);
  try {
    const resp = await fetch('/caption', { method: 'POST', body: formData });
    if (!resp.ok) { const err = await resp.json().catch(() => ({ detail: resp.statusText })); throw new Error(err.detail || 'Caption failed'); }
    const data = await resp.json();
    lastCaption = data.caption;
    outputArea.textContent = data.caption;
    outputArea.classList.remove('placeholder');
    outputMeta.style.display = 'flex';
    copyBtn.disabled = false;
    saveBtn.disabled = false;
    document.getElementById('meta-model').textContent = '🤖 ' + data.model;
    document.getElementById('meta-time').textContent = '⏱ ' + data.processing_time_ms.toFixed(0) + 'ms';
    document.getElementById('meta-dims').textContent = '📐 ' + data.image_size.width + '×' + data.image_size.height;
    showToast('Caption generated', 'success');
  } catch (err) {
    outputArea.textContent = 'Error: ' + err.message;
    outputArea.classList.remove('placeholder');
    showToast('Failed: ' + err.message, 'error');
  } finally {
    captionBtn.disabled = false;
    captionBtn.textContent = '🚀 Generate Caption';
  }
});

// Copy
copyBtn.addEventListener('click', async () => {
  if (!lastCaption) return;
  try { await navigator.clipboard.writeText(lastCaption); showToast('Copied to clipboard', 'success'); }
  catch { showToast('Copy failed', 'error'); }
});

// Save as .md
saveBtn.addEventListener('click', () => {
  if (!lastCaption) return;
  const baseName = currentImageName ? currentImageName.replace(/\.[^.]+$/, '') : 'caption';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lastCaption], { type: 'text/markdown' }));
  a.download = baseName + '.md';
  a.click();
  URL.revokeObjectURL(a.href);
  showToast('Saved as ' + baseName + '.md', 'success');
});

// Jobs
async function loadJobs() {
  try {
    const resp = await fetch('/jobs');
    if (!resp.ok) return;
    const data = await resp.json();
    jobs = {};
    data.jobs.forEach(j => jobs[j.id] = j);
    resultsCount.textContent = data.jobs.length;
    resultsEmpty.style.display = data.jobs.length === 0 ? 'block' : 'none';
    jobList.innerHTML = '';
    data.jobs.forEach(j => {
      const card = document.createElement('div');
      card.className = 'job-card';
      card.innerHTML = `
        <div class="job-thumb">
          <img src="/jobs/${j.id}/thumbnail" alt="Job image">
          <div class="job-thumb-overlay">
            <span>📐 ${j.image_size.width}×${j.image_size.height}</span>
          </div>
        </div>
        <div class="job-body">
          <div class="job-header">
            <div class="job-name">${j.filename}</div>
            <div class="job-actions">
              <button class="btn btn-ghost btn-sm" onclick="copyJob('${j.id}')">📋 Copy</button>
              <button class="btn btn-ghost btn-sm" onclick="downloadJobMd('${j.id}')">💾 .md</button>
              <button class="btn btn-ghost btn-sm" onclick="downloadJobImage('${j.id}')">🖼️ Image</button>
              <button class="btn-danger-sm" onclick="deleteJob('${j.id}')">✕ Delete</button>
            </div>
          </div>
          <div class="job-meta">
            <span>🤖 ${j.model}</span>
            <span>⏱ ${j.processing_time_ms.toFixed(0)}ms</span>
            <span>📅 ${j.created_at}</span>
          </div>
          <div class="job-preview">${escapeHtml(j.caption)}</div>
        </div>`;
      jobList.appendChild(card);
    });
  } catch {}
}

window.copyJob = async (id) => {
  const j = jobs[id];
  if (!j) return;
  try { await navigator.clipboard.writeText(j.caption); showToast('Copied', 'success'); }
  catch { showToast('Copy failed', 'error'); }
};

window.downloadJobMd = (id) => {
  const j = jobs[id];
  if (!j) return;
  const a = document.createElement('a');
  a.href = '/jobs/' + id + '/caption.md';
  a.download = j.filename;
  a.click();
};

window.downloadJobImage = (id) => {
  const j = jobs[id];
  if (!j) return;
  const a = document.createElement('a');
  a.href = '/jobs/' + id + '/image';
  a.download = j.image_path.split('/').pop();
  a.click();
};

window.deleteJob = async (id) => {
  if (!confirm('Delete this job (image + caption)?')) return;
  const resp = await fetch('/jobs/' + id, { method: 'DELETE' });
  if (resp.ok) { delete jobs[id]; loadJobs(); showToast('Job deleted', 'success'); }
  else { showToast('Delete failed', 'error'); }
};

// Config
async function loadConfig() {
  try {
    const resp = await fetch('/config');
    const config = await resp.json();
    document.getElementById('config-system-prompt').value = config.system_prompt;
    document.getElementById('model-info').innerHTML =
      '<strong>Model:</strong> ' + config.model_path + '<br>' +
      '<strong>GPU Layers:</strong> <code>' + config.gpu_layers + '/' + config.gpu_layers + '</code><br>' +
      '<strong>Context Size:</strong> <code>' + config.context_size + '</code> tokens<br>' +
      '<strong>Workers:</strong> ' + config.workers + ' | <strong>Batch Size:</strong> ' + config.batch_size;
  } catch {}
}

document.getElementById('save-config-btn').addEventListener('click', async () => {
  const systemPrompt = document.getElementById('config-system-prompt').value;
  const resp = await fetch('/config', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ system_prompt: systemPrompt }),
  });
  if (resp.ok) showToast('Configuration saved', 'success');
  else showToast('Failed to save', 'error');
});

document.getElementById('reset-config-btn').addEventListener('click', async () => {
  try {
    const resp = await fetch('/config', { method: 'DELETE' });
    if (!resp.ok) throw new Error('Failed to reset');
    const data = await resp.json();
    document.getElementById('config-system-prompt').value = data.system_prompt;
    showToast('Configuration reset to defaults', 'success');
  } catch {
    showToast('Failed to reset', 'error');
  }
});

// Logs
async function loadLogs() {
  try {
    const resp = await fetch('/logs?lines=200');
    if (!resp.ok) return;
    const data = await resp.json();
    lastLogsLines = data.lines || [];
    logsTotal.textContent = data.total + ' lines total';
    logsCount.textContent = data.shown || 0;
    if (!lastLogsLines.length) {
      logsOutput.textContent = '(no log lines yet)';
      return;
    }
    logsOutput.textContent = lastLogsLines.map(line => escapeHtml(line)).join('\n');
  } catch {
    logsOutput.textContent = '(failed to load logs)';
  }
}

function downloadLogs() {
  if (!lastLogsLines.length) { showToast('No logs to download', 'error'); return; }
  const blob = new Blob(lastLogsLines.map(l => l + '\n'), { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'server-logs-' + new Date().toISOString().slice(0,10) + '.log';
  a.click();
  URL.revokeObjectURL(a.href);
  showToast('Logs downloaded', 'success');
}

// Health
async function checkHealth() {
  // Vision model (llama-server) via backend proxy
  try {
    const resp = await fetch('/health/model');
    const data = await resp.json();
    if (data.status === 'ok') { modelDot.classList.remove('error'); }
    else { throw new Error(); }
  } catch { modelDot.classList.add('error'); }
}

function showToast(msg, type = 'success') {
  toast.textContent = msg;
  toast.className = 'toast ' + type;
  setTimeout(() => toast.classList.add('show'), 10);
  setTimeout(() => toast.classList.remove('show'), 3000);
}

function formatBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

checkHealth();
loadConfig();
setInterval(checkHealth, 15000);
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    check_gpu_requirements()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Rotating file handler: max 10MB per file, keep 5 backups
    file_handler = logging.handlers.RotatingFileHandler(
        str(LOG_FILE),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("FASTAPI_PORT", "8000")), timeout_keep_alive=0)

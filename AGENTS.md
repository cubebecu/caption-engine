# caption-engine — Agent Instructions

## Architecture

Single-file application. Everything lives in `server/main.py` (~1500 lines): FastAPI backend, embedded Web UI (HTML string), llama-server lifecycle, job management, image validation, health monitoring.

**Two-process model:** FastAPI (port 8000) spawns `llama-server` as a subprocess (port 8080). All inference goes through llama.cpp's OpenAI-compatible API at `http://127.0.0.1:8080/v1/...`.

**Entrypoint:** `python3 -m server.main` (Docker CMD). Locally: same command from repo root.

## No Tests, No Lint

No test suite, no linter, no type checker. Any code change is manually verified by running the service and testing through the Web UI or `/caption` endpoint.

## GPU-Only

Hard requirement: NVIDIA GPU with >=4GB VRAM. The service exits at startup if GPU check fails. No CPU fallback exists or should be added.

## Critical Files

| File | Purpose |
|------|---------|
| `server/main.py` | Entire application (backend + embedded Web UI HTML) |
| `system_prompt_default.md` | Default system prompt loaded at startup |
| `system_prompt_custom.md` | Persisted custom prompt (created by user via `/config` PUT) |
| `.env` | Runtime config (gitignored, contains secrets — do not commit) |
| `Dockerfile` | gitignored, not in repo. Binaries built externally, image pushed to GHCR |

## Dotenv / Secrets

`.env` is gitignored and contains HuggingFace + GitHub tokens. Never commit `.env`. The `.env` file is NOT auto-loaded by the app — environment variables are read via `os.getenv()` with hardcoded defaults. The `.env` file is only sourced by Docker Compose automatically.

## Building from Source

`tools/!!kompilacja.txt` has llama.cpp build instructions (Polish notes). Build produces `llama-server` binary + shared libs (`libggml*.so`, `libllama*.so`, `libmtmd*.so`), all gitignored. Models in `models/` are also gitignored (~3GB). Dockerfile copies all of these at build time.

Pre-built image: `ghcr.io/cubebecu/caption-engine:latest`.

## Jobs

Persisted to disk in `jobs/` directory (one subdirectory per job). Loaded into memory at startup via `_load_jobs()`. Volume-mounted in Docker Compose for persistence.

## System Prompt Flow

1. Load `system_prompt_default.md` at module import
2. If `system_prompt_custom.md` exists, override with its contents
3. PUT `/config` writes custom prompt to disk and updates in-memory `SYSTEM_PROMPT`
4. DELETE `/config` removes custom file, reverts to default

## Third-Party

`third_party/` contains license/ attribution files only, not code. Dependencies are pinned in `requirements.txt`.

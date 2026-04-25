<div align="center">

# caption-engine

### AI-powered image captioning for technical documentation

Upload a screenshot, diagram, or CLI output — get a structured markdown description.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docker Pulls](https://img.shields.io/badge/Docker-GHCR.io-2496ED?logo=docker)](https://github.com/username/caption-engine/pkgs/container/caption-engine)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%204GB%2B-brightgreen?logo=nvidia)](https://developer.nvidia.com/cuda-gpus)

</div>

---

## Quick Start

```bash
# Pull the Docker image
docker pull ghcr.io/username/caption-engine:latest

# Start the service
docker compose up -d
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

> Upload an image, click **Generate Caption**, get markdown output.

---

## Screenshot

<!-- TODO: add Web UI screenshot -->

---

## Under the Hood

| Component | Details |
|---|---|
| **Model** | Gemma-3-4B (multimodal vision, quantized Q4_K_M) |
| **Backend** | llama.cpp + FastAPI |
| **GPU** | NVIDIA CUDA, all 32 layers offloaded |
| **VRAM** | 4 GB minimum (no CPU fallback) |

---

## Documentation

Full technical reference — API endpoints, configuration, GPU requirements, building from source:

[DOCS.md](DOCS.md)

---

## License

Licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

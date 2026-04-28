<div align="center">

# caption-engine

### Image captioning for technical documentation

Upload a screenshot, diagram, or CLI output — get a structured markdown description.  
Built for on-device inference — with an API fallback for machines without a GPU.

[![Docker Pulls](https://img.shields.io/badge/Docker-GHCR.io-2496ED?logo=docker)](https://github.com/cubebecu/caption-engine/pkgs/container/caption-engine)
![Built offline with Qwen](https://img.shields.io/badge/Built%20offline%20with-Qwen-615CED)
[![Full GPU](https://img.shields.io/badge/Full-GPU-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-gpus)
[![Lite Online](https://img.shields.io/badge/Lite-Online-D97757?logo=anthropic&logoColor=white)](https://www.anthropic.com/api)
</div>

---
![Web UI](raw/demo-2.gif)
---

## Two Modes

| | **Full (Local)** | **Lite (Online)** |
|--|-----------|----------|
| Model | Gemma-3-4B (on-device) | Claude Sonnet (Anthropic API) |
| GPU | NVIDIA 4GB+ VRAM required | None — runs on any machine |
| Offline | Yes | No (API calls) |
| Image size | ~12 GB | ~600 MB |
| Cost | Free after setup | Per-request Anthropic pricing |

---

## Quick Start — Lite (no GPU)

Fastest way to try caption-engine. No GPU required.
Requires an [Anthropic API key](https://console.anthropic.com/).

```bash
# Pull compose file
curl -O https://raw.githubusercontent.com/cubebecu/caption-engine/main/docker-compose-lite.yml

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Start the service
docker compose -f docker-compose-lite.yml up -d
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## Quick Start — Local (GPU)

For workflows where screenshots can't leave your network or where API costs add up at scale.

### Prerequisites

- Docker + Docker Compose
- NVIDIA GPU (4 GB VRAM minimum): Ada, Ampere, Hopper, Blackwell, Turing
- NVIDIA Driver 570+
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Install NVIDIA Container Toolkit

```bash
# Ubuntu
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Install Caption Engine

```bash
# Pull compose file
curl -O https://raw.githubusercontent.com/cubebecu/caption-engine/main/docker-compose.yml

# Start the service
docker compose up -d
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

> Upload an image, click **Generate Caption**, get markdown output.

---

## Under the Hood

### Local Mode

| Component | Details |
|---|---|
| Model | Gemma-3-4B (multimodal vision, quantized Q4_K_M) |
| Backend | llama.cpp + FastAPI |
| GPU | NVIDIA CUDA, all 32 layers offloaded |
| VRAM | 4 GB minimum (no CPU fallback) |

### Lite Mode

| Component | Details |
|---|---|
| Model | Auto-detected latest Claude Sonnet |
| Backend | Anthropic Messages API + FastAPI |
| Config | `LLM_BACKEND=anthropic`, `ANTHROPIC_API_KEY` required |

---

## Documentation

Full technical reference — API endpoints, configuration, GPU requirements, building from source:

[DOCS.md](DOCS.md)

---

## Author
> [!NOTE]
Built by [cubebecu](https://github.com/cubebecu) as part of an evaluation of local agentic coding setups. The entire codebase was written using a local model running on a single NVIDIA GPU

## License
> [!NOTE] Code — Apache License 2.0  
The application code in this repository is licensed under Apache 2.0.
See [LICENSE](LICENSE) for full text.

> [!NOTE] Model weights (Local mode only) — Google Gemma Terms of Use  
The Docker image bundles **Gemma-3-4B** model weights, which are **NOT**
under Apache 2.0. Gemma is governed by Google's [Gemma Terms of Use](gemma/TERMS_OF_USE.md)
and [Prohibited Use Policy](gemma/PROHIBITED_USE_POLICY.md).
By pulling the Docker image or using the bundled model in any form, you
agree to those terms. The Prohibited Use Policy contains binding
restrictions on what Gemma may be used for — read it before deploying
this in production.

**Lite mode** does not bundle any model weights and is not subject to Gemma ToU.

### Third-party components
See [NOTICE](NOTICE) and [third_party/](third_party/) for full
attribution and license texts of bundled dependencies.

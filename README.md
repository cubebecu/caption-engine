<div align="center">

# caption-engine

### Image captioning for technical documentation

Upload a screenshot, diagram, or CLI output — get a structured markdown description.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docker Pulls](https://img.shields.io/badge/Docker-GHCR.io-2496ED?logo=docker)](https://github.com/cubebecu/caption-engine/pkgs/container/caption-engine)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%204GB%2B-brightgreen?logo=nvidia)](https://developer.nvidia.com/cuda-gpus)
![Built offline](https://img.shields.io/badge/built_with-offline_LLM-blueviolet)
![Runs offline](https://img.shields.io/badge/runs-fully_offline-green)

</div>

---

![Web UI](raw/i1.png)

---

## Prerequisites

- Docker + Docker Compose
- NVIDIA GPU (4 GB VRAM minimum), supported architectures: Ada, Ampere, Hopper, Blackwell
- NVIDIA Driver 570+
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## To install NVIDIA container toolkit:
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

## To install caption engine
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
## Author
[cubebecu](https://github.com/cubebecu)

## License

Licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

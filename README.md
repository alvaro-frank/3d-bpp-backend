# 3D Bin Packing System - Backend

![CI Status](https://github.com/alvaro-frank/sentiment_analysis/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white)
![PPO](https://img.shields.io/badge/Agent-PPO-FF6F61)
![DQN](https://img.shields.io/badge/Agent-DQN-8A2BE2)

A production-grade 3D Bin Packing optimization service. This project implements a **Reinforcement Learning (DQN/PPO)** based engine for the 3D Bin Packing Problem (3D-BPP), served via a high-performance FastAPI backend. It follows **Clean Architecture** principles to ensure the core optimization logic remains decoupled from the AI inference engine and the web framework.

## 📂 Project Structure
```
├── src/
│   ├── application/        # Application Logic (Use Cases & DTOs)
│   │   ├── dtos/           # Data Transfer Objects for API-to-Domain mapping
│   │   ├── ports/          # Interfaces for Outbound Adapters (AI Predictor)
│   │   └── use_cases/      # Business logic orchestrators (Packing Workflow)
│   ├── domain/             # Business Core (Pure Entities)
│   │   └── entities.py     # Container, Box, and Position entities
│   ├── infrastructure/     # Technical Details & External Integrations
│   │   ├── adapters/       # Implementation of Ports
│   │   │   ├── ingoing/    # FastAPI Routers (HTTP Entry points)
│   │   │   └── outgoing/   # ONNX Inference Engine for the RL Model
│   │   └── schemas/        # Pydantic Schemas for Request/Response validation
│   └── main.py             # Application entry point & DI Container
├── models/                 # Pre-trained RL Models (ONNX format)
│   ├── dqn/                # Deep Q-Network models
│   └── ppo/                # Proximal Policy Optimization models
├── tests/                  # Automated Test Suite
│   ├── unit/               # Domain, Use Case, and Adapter unit tests
│   └── integration/        # End-to-end packing flow tests
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Service orchestration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🛠️ Setup & Requirements

- `Python 3.10+`
- `Docker` and `Docker Compose`

1. **Clone the repository**
```bash
git clone https://github.com/alvaro-frank/3d-bpp-backend.git
cd 3d-bpp-backend
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

To run the API using **Docker** (recommended):

```bash
docker-compose up --build
```

The API will be available at http://localhost:8000. You can access the interactive documentation at http://localhost:8000/docs.

To run **locally**:

```bash
python -m src.main
```

## 🏃 Usage & API Specification

The primary endpoint is `POST /api/pack`.

**Request Payload Example**:
```bash
{
  "container": { "width": 10, "depth": 10, "height": 10 },
  "boxes": [
    { "id": "box_1", "width": 5, "depth": 5, "height": 5 },
    { "id": "box_2", "width": 3, "depth": 3, "height": 3 },
    { "id": "box_3", "width": 2, "depth": 2, "height": 2 },
    { "id": "box_4", "width": 4, "depth": 3, "height": 1 },
    { "id": "box_5", "width": 2, "depth": 2, "height": 3 },
    { "id": "box_6", "width": 3, "depth": 3, "height": 1 },
  ]
}
```

**Response Payload Example**:
```bash
{
  "packed_boxes": [
    {
      "box_id": "box_1",
      "position": { "x": 0, "y": 0, "z": 0 },
      "rotation_type": 0,
      "rotated_dimensions": [5, 5, 5]
    },
    ...
  ]
}
```

## 🧠 Methodology

**AI & Reinforcement Learning**

The packing logic is driven by a Reinforcement Learning agent trained in a custom 3D environment.

1. **Observation Space**: A 2D Heightmap represents the current state of the container, combined with the dimensions of the next items in the queue (Lookahead).
2. **Action Space**: The agent selects a coordinate $(x, y)$ and a rotation $(0-5)$.
3. Inference: Models are exported to **ONNX** format to allow high-speed inference without the overhead of heavy deep learning frameworks (like PyTorch or TensorFlow) in production.

**Physics & Action Masking**

To prevent the AI from making illegal moves, a **Physics Mask** is generated for every step. It calculates valid $(x, y)$ positions based on the current heightmap and the box's rotated footprint, ensuring 100% valid placements.

## 🧪 Testing

The project maintains high reliability through a tiered testing strategy:

```bash
# Run all tests
pytest

# Run with coverage logs
pytest -s -v
```

- **Domain Tests**: Validate rotation math and volume calculations.
- **Application Tests**: Mock the AI to test the packing orchestration logic.
- **Infrastructure Tests**: Verify the ONNX adapter, action masking, and FastAPI routing.
- **Integration Tests**: Full end-to-end flow using real .onnx model files.

## ⚙️ CI/CD Pipeline

The project includes a GitHub Actions workflow that automates the quality gate:

- **Linting**: Ensures PEP8 compliance.
- **Automated Testing**: Executes the full pytest suite on every push.
- **Build Verification**: Ensures the Docker image builds successfully.

## 🐳 Docker Support

The application is optimized for containerization:

- **Lightweight**: Uses python:3.10-slim to minimize image size.
- **Production-Ready**: Uses uvicorn as an ASGI server.
- **Isolated Inference**: The ONNX Runtime is CPU-optimized for cloud deployment.

# Spacewalker - Traversing Representations Spaces for Fast Interactive Exploration and Annotation of Unstructured Data

__Disclaimer: Do not use the version "as is" in production. This repository uses unsafe, dummy access credentials for Postgres and MinIO to showcase its functionality.__

Please switch to the `development` branch for the latest features.
## Prerequisites
- [Docker](https://www.docker.com/get-started/)
- Optional (for devcontainer functionality): VS Code + Docker Plugin
- Recommended: CUDA-compatible GPU for Triton. If unavailable, CPU will be used.

## Getting started

- Clone / download this repository
- Download the `model_repository` folder and place it inside the root folder of this repository. The directory should look like this:
```
Spacewalker/
├── .devcontainer
├── backend
├── environments
├── model_repository
├── Triton
├── .gitignore
├── docker-compose-develop.yml
├── docker-compose.yml
├── Dockerfile
├── inference-requests.py
├── LICENSE
├── manage.py
├── package-lock.json
├── package.json
├── README.md
└── requirements.txt
```
- Ensure that Docker is running
- Run the following command:

```bash
docker compose -f docker-compose.yml up --remove-orphans --force-recreate
```

- Navigate to `http://0.0.0.0:8080` (`http://localhost:8080` on Windows)

## Services
The following services are exposed on their default ports:
- [NVIDIA Triton Inference Server](https://www.nvidia.com/en-us/ai-data-science/products/triton-inference-server/) (8000, 8001)
- [MinIO](https://min.io) (9000, 9001)
- Spacewalker (8080)

## Development:
```bash
docker compose -f docker-compose.yml -f docker-compose-develop.yml up --remove-orphans --force-recreate
```

Frontend development:
- open project in VS Code
- open in .devcontainer
- ```cd frontend```
- ```npx parcel ./src/index.html --dist-dir=/workspaces/SpaceWalker/backend/static/frontend```
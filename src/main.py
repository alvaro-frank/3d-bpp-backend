import os
import uvicorn
from fastapi import FastAPI

from src.infrastructure.adapters.ingoing.routers import pack_router, get_use_case
from src.infrastructure.adapters.outgoing.onnx_agent import OnnxAgentPredictor
from src.application.use_cases.pack_container_use_case import PackContainerUseCase

def create_app() -> FastAPI:
    """
    Application factory that initializes the FastAPI server, wires the dependencies,
    and returns the configured application instance.

    Returns:
        FastAPI: The fully configured FastAPI application ready to serve requests.
    """
    app = FastAPI(
        title="3D Bin Packing API",
        description="An AI-powered service for optimizing 3D spatial packing using Clean Architecture.",
        version="1.0.0"
    )

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "ppo", "ppo_final.onnx")
    
    if not os.path.exists(model_path):
        print(f"WARNING: Model not found at {model_path}. Packing predictions will fail.")

    agent_predictor = OnnxAgentPredictor(model_path=model_path, lookahead=10)

    pack_use_case = PackContainerUseCase(agent_predictor=agent_predictor)

    app.dependency_overrides[get_use_case] = lambda: pack_use_case

    app.include_router(pack_router)

    @app.get("/", tags=["Health"])
    def health_check():
        """Simple health check endpoint to verify the API is running."""
        return {"status": "ok", "message": "3D Bin Packing API is up and running!"}

    return app

app = create_app()

if __name__ == "__main__":
    """
    Entry point for running the server locally for development.
    """
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
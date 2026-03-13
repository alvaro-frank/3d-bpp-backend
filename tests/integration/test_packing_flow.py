import pytest
import os
from src.application.use_cases.pack_container_use_case import PackContainerUseCase
from src.infrastructure.adapters.outgoing.onnx_agent import OnnxAgentPredictor
from src.application.dtos.dtos import ContainerInputDTO, BoxInputDTO

def test_integration_use_case_with_real_onnx():
    """
    Ensures that the Use Case and the ONNX Adapter work together correctly
    using the actual model file.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "models", "dqn", "dqn_final.onnx")
    
    if not os.path.exists(model_path):
        pytest.skip("ONNX model not found, skipping integration test.")

    predictor = OnnxAgentPredictor(model_path=model_path)
    use_case = PackContainerUseCase(agent_predictor=predictor)

    container_dto = ContainerInputDTO(width=10, depth=10, height=10)
    boxes_dto = [BoxInputDTO(id="REAL-TEST", width=2, depth=2, height=2)]

    result = use_case.execute(container_dto, boxes_dto)

    assert len(result) == 1
    assert result[0].box_id == "REAL-TEST"
    assert 0 <= result[0].position.x <= 10
    assert 0 <= result[0].position.z <= 10
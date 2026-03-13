import pytest
from unittest.mock import Mock
from src.application.use_cases.pack_container_use_case import PackContainerUseCase
from src.application.dtos.dtos import ContainerInputDTO, BoxInputDTO

def test_pack_container_use_case_orchestration():
    """
    Test if the Use Case correctly orchestrates the packing process by
    calling the predictor and mapping the results to DTOs.
    """
    mock_predictor = Mock()
    
    mock_predictor.predict_action.return_value = 0
    
    use_case = PackContainerUseCase(agent_predictor=mock_predictor)
    
    container_dto = ContainerInputDTO(width=10, depth=10, height=10)
    boxes_dto = [
        BoxInputDTO(id="BOX-1", width=2, depth=2, height=2)
    ]
    
    result = use_case.execute(container_dto, boxes_dto)
    
    assert len(result) == 1
    assert result[0].box_id == "BOX-1"
    assert result[0].position.x == 0
    assert result[0].position.y == 0
    assert result[0].position.z == 0
    
    mock_predictor.predict_action.assert_called_once()

def test_use_case_skips_box_when_ai_returns_minus_one():
    """
    Test if the Use Case correctly skips a box if the predictor returns -1
    (action masking case).
    """
    mock_predictor = Mock()

    mock_predictor.predict_action.return_value = -1
    
    use_case = PackContainerUseCase(agent_predictor=mock_predictor)
    
    container_dto = ContainerInputDTO(width=10, depth=10, height=10)
    boxes_dto = [BoxInputDTO(id="BIG-BOX", width=20, depth=20, height=20)]
    
    result = use_case.execute(container_dto, boxes_dto)
    
    assert len(result) == 0
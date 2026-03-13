import pytest
import numpy as np
import os
from src.infrastructure.adapters.outgoing.onnx_agent import OnnxAgentPredictor
from src.domain.entities import Container, Box

@pytest.fixture
def predictor():
    """Fixture to initialize the predictor with the real ONNX model."""
    current_file = os.path.abspath(__file__)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    model_path = os.path.join(root_dir, "models", "dqn", "dqn_final.onnx")
    
    if not os.path.exists(model_path):
        pytest.skip("Modelo ONNX not found")
    return OnnxAgentPredictor(model_path=model_path)

def test_observation_builder_shape(predictor):
    """
    Test if the observation vector has the correct dimensions for the ONNX model.
    Expected: (width * depth) + (lookahead * 3) + 3
    For 10x10 and lookahead 10: 100 + 30 + 3 = 133
    """
    container = Container(width=10, depth=10, height=10)
    box = Box(id="b1", width=2, depth=2, height=2)
    
    obs = predictor._build_observation(container, [], [box])
    
    assert obs.shape[0] == 133
    assert isinstance(obs, np.ndarray)

def test_action_mask_prevents_out_of_bounds(predictor):
    """
    Critical Test: Ensure the mask blocks actions that would place 
    a box outside the container.
    """
    container = Container(width=10, depth=10, height=10)

    big_box = Box(id="big", width=6, depth=6, height=6)
    
    mask = predictor._get_valid_action_mask(container, big_box, [])
    
    rotations = 6
    invalid_idx = 5 * (10 * rotations) + 5 * rotations + 0
    
    assert mask[invalid_idx] == False

def test_action_mask_allows_valid_position(predictor):
    """Ensure the mask allows a perfectly valid position."""
    container = Container(width=10, depth=10, height=10)
    small_box = Box(id="small", width=1, depth=1, height=1)
    
    mask = predictor._get_valid_action_mask(container, small_box, [])
    
    assert mask[0] == True
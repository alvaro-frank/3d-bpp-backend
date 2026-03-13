from fastapi.testclient import TestClient
from src.main import app
from src.infrastructure.adapters.ingoing.routers import get_use_case
from unittest.mock import Mock
import pytest

mock_use_case = Mock()

app.dependency_overrides[get_use_case] = lambda: mock_use_case

client = TestClient(app)

def test_pack_endpoint_success():
    """Test if the API endpoint correctly handles a valid request."""
    mock_use_case.execute.return_value = []

    payload = {
        "container": {"width": 10, "depth": 10, "height": 10},
        "boxes": [{"id": "b1", "width": 1, "depth": 1, "height": 1}]
    }
    
    response = client.post("/api/pack", json=payload)
    
    assert response.status_code == 200
    assert "packed_boxes" in response.json()
    mock_use_case.execute.assert_called_once()

def test_pack_endpoint_invalid_data():
    """Test if the API returns 422 (Unprocessable Entity) for bad JSON."""
    payload = {"container": {"width": -10}}
    response = client.post("/api/pack", json=payload)
    assert response.status_code == 422
from abc import ABC, abstractmethod
from typing import List
from src.domain.entities import Container, Box, PackedBox

class IAgentPredictor(ABC):
    """
    Interface for the AI Agent Predictor.
    
    This interface decouples the core application logic from the underlying 
    Machine Learning framework (e.g., ONNX Runtime, PyTorch, etc.).
    """
    
    @abstractmethod
    def predict_action(self, container: Container, current_box: Box, packed_boxes: List[PackedBox], remaining_boxes: List[Box]) -> int:
        """
        Predicts the optimal placement action (index) for the current box.

        Args:
            container (Container): The bin being packed.
            current_box (Box): The item currently being evaluated for placement.
            packed_boxes (List[PackedBox]): Items already placed inside the container.
            remaining_boxes (List[Box]): The lookahead queue of items yet to be packed.

        Returns:
            int: The discrete action index corresponding to (x, y, rotation).
        """
        pass
import numpy as np
import onnxruntime as ort
from typing import List
from src.domain.entities import Container, Box, PackedBox, Position
from src.application.ports.ports import IAgentPredictor

class OnnxAgentPredictor(IAgentPredictor):
    """
    Outbound Adapter that implements the IAgentPredictor port using ONNX Runtime.
    
    This class is responsible for loading the trained AI model in ONNX format,
    translating the pure domain entities into mathematical tensors, and 
    invoking the model to predict the best packing placement.
    
    Attributes:
        session (ort.InferenceSession): The active ONNX runtime session loaded in memory.
        lookahead (int): The number of upcoming boxes the model expects in the observation vector.
        input_name (str): The designated name of the input node for the loaded ONNX model.
    """
    
    def __init__(self, model_path: str, lookahead: int = 10):
        """
        Initializes the ONNX Agent Predictor by creating an inference session.

        Args:
            model_path (str): The filesystem path to the compiled .onnx model file.
            lookahead (int): How many upcoming boxes the model expects in the observation. Defaults to 10.
        """
        self.session = ort.InferenceSession(model_path)
        self.lookahead = lookahead
        
        self.input_name = self.session.get_inputs()[0].name

    def predict_action(self, container: Container, current_box: Box, packed_boxes: List[PackedBox], remaining_boxes: List[Box]) -> int:
        """
        Executes the AI model inference to predict the optimal placement action for the current box.

        Args:
            container (Container): The target bin where boxes are being packed.
            current_box (Box): The item currently being evaluated for placement.
            packed_boxes (List[PackedBox]): The items already placed inside the container.
            remaining_boxes (List[Box]): The lookahead queue of items yet to be packed.

        Returns:
            int: The discrete action index corresponding to the predicted (x, y, rotation) placement.
        """
        state_array = self._build_observation(container, packed_boxes, remaining_boxes)
        state_tensor = np.expand_dims(state_array, axis=0).astype(np.float32)
        
        outputs = self.session.run(None, {self.input_name: state_tensor})
        action_values = outputs[0][0] 
        
        valid_mask = self._get_valid_action_mask(container, current_box, packed_boxes)
        
        if not np.any(valid_mask):
            return -1

        action_values[~valid_mask] = -np.inf
        
        best_action_idx = int(np.argmax(action_values))
        
        return best_action_idx
    
    def _get_valid_action_mask(self, container: Container, box: Box, packed_boxes: List[PackedBox]) -> np.ndarray:
        """
        Calculates a boolean mask representing all physically possible placements for a box.
        
        Args:
            container (Container): The target bin.
            box (Box): The item to be placed.
            packed_boxes (List[PackedBox]): Items already placed inside the container.
            
        Returns:
            np.ndarray: A 1D boolean array where True means the action is physically valid.
        """
        rotations = 6
        action_space_size = container.width * container.depth * rotations
        mask = np.zeros(action_space_size, dtype=bool)
        
        heightmap = np.zeros((container.width, container.depth), dtype=int)
        for p_box in packed_boxes:
            bx, by, bz = p_box.position.x, p_box.position.y, p_box.position.z
            bw, bd, bh = p_box.rotated_dimensions
            heightmap[bx:bx+bw, by:by+bd] = np.maximum(heightmap[bx:bx+bw, by:by+bd], bz + bh)

        for rot in range(rotations):
            temp_pbox = PackedBox(box=box, position=Position(0,0,0), rotation_type=rot)
            rot_w, rot_d, rot_h = temp_pbox.rotated_dimensions

            for x in range(container.width):
                for y in range(container.depth):
                    if x + rot_w > container.width or y + rot_d > container.depth:
                        continue
                    
                    z = np.max(heightmap[x:x+rot_w, y:y+rot_d])
                    if z + rot_h > container.height:
                        continue
                    
                    action_idx = x * (container.depth * rotations) + y * rotations + rot
                    if action_idx < action_space_size:
                        mask[action_idx] = True
                        
        return mask

    def _build_observation(self, container: Container, packed_boxes: List[PackedBox], remaining_boxes: List[Box]) -> np.ndarray:
        """
        Translates Domain Entities into the flat Numpy array (observation vector) expected by the Neural Network.
        
        The observation vector consists of three parts:
        1. A normalized 2D heightmap of the current container.
        2. A flattened list of dimensions for the upcoming boxes (up to the lookahead limit).
        3. Global statistics including remaining box ratio, volume utilization, and height ratio.

        Args:
            container (Container): The target bin where boxes are being packed.
            packed_boxes (List[PackedBox]): The items already placed inside the container.
            remaining_boxes (List[Box]): The lookahead queue of items yet to be packed.

        Returns:
            np.ndarray: A 1D float32 numpy array representing the environment's state.
        """
        heightmap = np.zeros((container.width, container.depth), dtype=np.float32)
        
        for p_box in packed_boxes:
            x = p_box.position.x
            y = p_box.position.y
            z = p_box.position.z
            bw, bd, bh = p_box.rotated_dimensions
            
            for dx in range(bw):
                for dy in range(bd):
                    if 0 <= x + dx < container.width and 0 <= y + dy < container.depth:
                        heightmap[x + dx, y + dy] = max(heightmap[x + dx, y + dy], z + bh)
                        
        heightmap_flat = heightmap.flatten() / container.height

        upcoming = []
        for box in remaining_boxes[:self.lookahead]:
            upcoming.extend([box.width, box.depth, box.height])
            
        while len(upcoming) < self.lookahead * 3:
            upcoming.extend([0, 0, 0])
            
        upcoming_array = np.array(upcoming, dtype=np.float32)

        max_boxes = len(packed_boxes) + len(remaining_boxes) + 1
        current_step = len(packed_boxes)
        
        volume_used = sum(p.box.volume for p in packed_boxes)
        max_h = np.max(heightmap) if heightmap.size > 0 else 0.0
        
        stats_array = np.array([
            (max_boxes - current_step) / max_boxes if max_boxes > 0 else 0.0,
            volume_used / container.volume if container.volume > 0 else 0.0,
            max_h / container.height if container.height > 0 else 0.0
        ], dtype=np.float32)

        return np.concatenate([heightmap_flat, upcoming_array, stats_array])
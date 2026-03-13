from typing import List
from src.domain.entities import Container, Box, PackedBox, Position
from src.application.ports.ports import IAgentPredictor
from src.application.dtos.dtos import (
    ContainerInputDTO, 
    BoxInputDTO, 
    PackedBoxOutputDTO, 
    PositionOutputDTO
)

class PackContainerUseCase:
    """
    Use Case responsible for orchestrating the 3D Bin Packing process.
    """
    
    def __init__(self, agent_predictor: IAgentPredictor):
        self.agent_predictor = agent_predictor

    def execute(self, container_dto: ContainerInputDTO, boxes_dto: List[BoxInputDTO]) -> List[PackedBoxOutputDTO]:
        """
        Executes the packing logic by mapping input DTOs to Domain Entities, asking the AI 
        agent for predictions, and returning the result as Output DTOs.

        Args:
            container_dto (ContainerInputDTO): The data transfer object for the target bin.
            boxes_dto (List[BoxInputDTO]): The list of data transfer objects representing boxes to pack.

        Returns:
            List[PackedBoxOutputDTO]: The final packing plan represented as boundary output objects.
        """
        domain_container = Container(
            width=container_dto.width, 
            depth=container_dto.depth, 
            height=container_dto.height
        )
        domain_boxes = [
            Box(id=b.id, width=b.width, depth=b.depth, height=b.height) 
            for b in boxes_dto
        ]
        
        packed_boxes: List[PackedBox] = []
        remaining_boxes = domain_boxes.copy()
        
        while remaining_boxes:
            current_box = remaining_boxes.pop(0)
            
            action_idx = self.agent_predictor.predict_action(
                container=domain_container,
                current_box=current_box,
                packed_boxes=packed_boxes,
                remaining_boxes=remaining_boxes
            )
            
            if action_idx == -1:
                print(f"INFO: The box {current_box.id} does not fit...")
                continue
            
            x, y, rotation = self._decode_action(action_idx, domain_container.width, domain_container.depth)
            z = self._calculate_z_position(x, y, current_box, rotation, packed_boxes)
            
            position = Position(x=x, y=y, z=z)
            packed_box = PackedBox(box=current_box, position=position, rotation_type=rotation)
            packed_boxes.append(packed_box)
            
        output_dtos: List[PackedBoxOutputDTO] = []
        for p_box in packed_boxes:
            pos_dto = PositionOutputDTO(x=p_box.position.x, y=p_box.position.y, z=p_box.position.z)
            out_dto = PackedBoxOutputDTO(
                box_id=p_box.box.id,
                position=pos_dto,
                rotation_type=p_box.rotation_type,
                rotated_dimensions=p_box.rotated_dimensions
            )
            output_dtos.append(out_dto)
            
        return output_dtos

    def _decode_action(self, action_idx: int, container_width: int, container_depth: int) -> tuple[int, int, int]:
        """
        Decodes a flat action index into its corresponding 3D spatial placement and rotation components.
        
        This method reverses the mathematical mapping used to flatten the discrete action space
        during the AI training phase. The mapping assumes the loops were nested as: 
        X -> Y -> Rotation(6 possible states).
        
        Args:
            action_idx (int): The flat index predicted by the AI agent.
            container_width (int): The total capacity along the X-axis of the container.
            container_depth (int): The total capacity along the Y-axis of the container.
            
        Returns:
            Tuple[int, int, int]: A tuple containing:
                - x (int): The calculated position along the X-axis.
                - y (int): The calculated position along the Y-axis.
                - rotation (int): The calculated orientation index (0 to 5).
        """
        rotations = 6
        
        rotation = action_idx % rotations
        y = (action_idx // rotations) % container_depth
        x = action_idx // (rotations * container_depth)
        
        return x, y, rotation
        
    def _calculate_z_position(self, x: int, y: int, box: Box, rotation: int, packed_boxes: List[PackedBox]) -> int:
        """
        Calculates the lowest valid Z (height) coordinate for placing a box, simulating gravity and stacking.
        
        Evaluates potential collisions on the X and Y axes with all previously packed boxes to find
        the maximum height (Z) of any overlapping items. This maximum height becomes the base Z coordinate 
        for the new box.
        
        Args:
            x (int): The intended placement position along the X-axis.
            y (int): The intended placement position along the Y-axis.
            box (Box): The entity representing the item to be packed.
            rotation (int): The orientation index (0 to 5) applied to the box.
            packed_boxes (List[PackedBox]): The list of items already placed inside the container.
            
        Returns:
            int: The minimum required Z-coordinate (height) to safely place the box without overlapping others.
        """
        temp_position = Position(x=0, y=0, z=0)
        temp_packed_box = PackedBox(box=box, position=temp_position, rotation_type=rotation)
        bw, bd, bh = temp_packed_box.rotated_dimensions
        
        max_z = 0
        
        for p_box in packed_boxes:
            bx = p_box.position.x
            by = p_box.position.y
            bz = p_box.position.z
            bw2, bd2, bh2 = p_box.rotated_dimensions
            
            overlap_x = not (x + bw <= bx or x >= bx + bw2)
            overlap_y = not (y + bd <= by or y >= by + bd2)
            
            if overlap_x and overlap_y:
                top_z = bz + bh2
                if top_z > max_z:
                    max_z = top_z
                    
        return max_z
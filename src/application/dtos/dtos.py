from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class BoxInputDTO:
    """
    Data Transfer Object representing a box to be packed, used as input for the application layer.
    
    Attributes:
        id (str): A unique identifier for the box.
        width (int): The size along the X-axis.
        depth (int): The size along the Y-axis.
        height (int): The size along the Z-axis.
    """
    id: str
    width: int
    depth: int
    height: int

@dataclass(frozen=True)
class ContainerInputDTO:
    """
    Data Transfer Object representing the target container, used as input for the application layer.
    
    Attributes:
        width (int): The total capacity along the X-axis.
        depth (int): The total capacity along the Y-axis.
        height (int): The total capacity along the Z-axis.
    """
    width: int
    depth: int
    height: int

@dataclass(frozen=True)
class PositionOutputDTO:
    """
    Data Transfer Object representing a spatial coordinate, used as output from the application layer.
    
    Attributes:
        x (int): The calculated position along the X-axis.
        y (int): The calculated position along the Y-axis.
        z (int): The calculated position along the Z-axis.
    """
    x: int
    y: int
    z: int

@dataclass(frozen=True)
class PackedBoxOutputDTO:
    """
    Data Transfer Object representing a successfully packed box, used as output from the application layer.
    
    Attributes:
        box_id (str): The unique identifier of the packed box.
        position (PositionOutputDTO): The spatial coordinates where the box was placed.
        rotation_type (int): The orientation index applied to the box.
        rotated_dimensions (Tuple[int, int, int]): The effective dimensions of the box after rotation.
    """
    box_id: str
    position: PositionOutputDTO
    rotation_type: int
    rotated_dimensions: Tuple[int, int, int]
from pydantic import BaseModel
from typing import List, Tuple

class PositionResponse(BaseModel):
    """
    Data Transfer Object representing the spatial coordinates in the HTTP response.
    """
    x: int
    y: int
    z: int

class PackedBoxResponse(BaseModel):
    """
    Data Transfer Object representing a successfully packed box in the HTTP response.
    """
    box_id: str
    position: PositionResponse
    rotation_type: int
    rotated_dimensions: Tuple[int, int, int]

class PackResponse(BaseModel):
    """
    Data Transfer Object for the final packing plan payload.
    """
    packed_boxes: List[PackedBoxResponse]
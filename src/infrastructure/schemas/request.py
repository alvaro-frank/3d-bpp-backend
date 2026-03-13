from pydantic import BaseModel, Field
from typing import List

class BoxRequest(BaseModel):
    """
    Data Transfer Object representing a box in the incoming HTTP request.
    """
    id: str = Field(..., description="Unique identifier for the box")
    width: int = Field(..., gt=0, description="Box width (X-axis)")
    depth: int = Field(..., gt=0, description="Box depth (Y-axis)")
    height: int = Field(..., gt=0, description="Box height (Z-axis)")

class ContainerRequest(BaseModel):
    """
    Data Transfer Object representing the container in the incoming HTTP request.
    """
    width: int = Field(..., gt=0, description="Container width (X-axis)")
    depth: int = Field(..., gt=0, description="Container depth (Y-axis)")
    height: int = Field(..., gt=0, description="Container height (Z-axis)")

class PackRequest(BaseModel):
    """
    Data Transfer Object for the main packing request payload.
    """
    container: ContainerRequest
    boxes: List[BoxRequest] = Field(..., min_length=1, description="List of boxes to pack")
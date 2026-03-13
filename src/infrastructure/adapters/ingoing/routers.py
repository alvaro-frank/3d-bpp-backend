from fastapi import APIRouter, Depends, HTTPException

from src.infrastructure.schemas.request import PackRequest
from src.infrastructure.schemas.response import PackResponse, PackedBoxResponse, PositionResponse
from src.application.dtos.dtos import ContainerInputDTO, BoxInputDTO
from src.application.use_cases.pack_container_use_case import PackContainerUseCase

pack_router = APIRouter(prefix="/api", tags=["Packing"])

def get_use_case():
    """
    Dependency Injection placeholder for the FastAPI router.
    The actual implementation will be overridden in the Composition Root (main.py).
    """
    raise NotImplementedError("Dependency must be overridden in main.py")

@pack_router.post("/pack", response_model=PackResponse)
def pack_items(request: PackRequest, use_case: PackContainerUseCase = Depends(get_use_case)):
    """
    Endpoint to calculate the optimal 3D bin packing plan for a given set of boxes.
    
    Args:
        request (PackRequest): The validated JSON payload (Web DTO) containing container and box dimensions.
        use_case (PackContainerUseCase): The injected application use case.
        
    Returns:
        PackResponse: The final packing plan mapped back to a Web DTO.
    """
    try:
        app_container_dto = ContainerInputDTO(
            width=request.container.width, 
            depth=request.container.depth, 
            height=request.container.height
        )
        
        app_boxes_dto = [
            BoxInputDTO(id=b.id, width=b.width, depth=b.depth, height=b.height) 
            for b in request.boxes
        ]
        
        app_output_dtos = use_case.execute(container_dto=app_container_dto, boxes_dto=app_boxes_dto)
        
        response_boxes = []
        for out_dto in app_output_dtos:
            response_boxes.append(
                PackedBoxResponse(
                    box_id=out_dto.box_id,
                    position=PositionResponse(
                        x=out_dto.position.x,
                        y=out_dto.position.y,
                        z=out_dto.position.z
                    ),
                    rotation_type=out_dto.rotation_type,
                    rotated_dimensions=out_dto.rotated_dimensions
                )
            )
            
        return PackResponse(packed_boxes=response_boxes)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error during packing prediction.")
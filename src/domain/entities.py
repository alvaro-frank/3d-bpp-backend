from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Position:
    """
    Represents a 3D spatial coordinate within a container.
    
    Attributes:
        x (int): The position along the X-axis (width).
        y (int): The position along the Y-axis (depth).
        z (int): The position along the Z-axis (height).
    """
    x: int
    y: int
    z: int
    
    def __post_init__(self):
        if self.x < 0 or self.y < 0 or self.z < 0:
            raise ValueError(f"Domain Error: Coordinates cannot be negative. Got ({self.x}, {self.y}, {self.z})")


@dataclass(frozen=True)
class Box:
    """
    Represents an item to be packed, defined by its physical dimensions.
    
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
    
    def __post_init__(self):
        if self.width <= 0 or self.depth <= 0 or self.height <= 0:
            raise ValueError(
                f"Domain Error: Box dimensions must be strictly positive. "
                f"Got W:{self.width}, D:{self.depth}, H:{self.height} for Box {self.id}"
            )
            
    @property
    def volume(self) -> int:
        """
        Calculates the total volume of the box.

        Returns:
            int: The volume (width * depth * height).
        """
        return self.width * self.depth * self.height


@dataclass(frozen=True)
class PackedBox:
    """
    Represents a box that has been assigned a position and rotation inside a container.
    
    Attributes:
        box (Box): The original box entity.
        position (Position): The spatial coordinate where the box is placed.
        rotation_type (int): The orientation index (0 to 5) applied to the box.
    """
    box: Box
    position: Position
    rotation_type: int
    
    def __post_init__(self):
        if self.rotation_type < 0 or self.rotation_type > 5:
            raise ValueError(f"Domain Error: Rotation type must be between 0 and 5. Got {self.rotation_type}")
            
    @property
    def rotated_dimensions(self) -> Tuple[int, int, int]:
        """
        Calculates the effective dimensions of the box after applying its rotation.

        Returns:
            Tuple[int, int, int]: The rotated (width, depth, height).
        """
        w, d, h = self.box.width, self.box.depth, self.box.height
        rot = self.rotation_type
        
        if rot == 0: return (w, d, h)
        if rot == 1: return (w, h, d)
        if rot == 2: return (d, w, h)
        if rot == 3: return (h, d, w)
        if rot == 4: return (d, h, w)
        if rot == 5: return (h, w, d)
        
        return (w, d, h)


@dataclass(frozen=True)
class Container:
    """
    Represents the 3D bin/container where boxes will be packed.
    
    Attributes:
        width (int): The total capacity along the X-axis.
        depth (int): The total capacity along the Y-axis.
        height (int): The total capacity along the Z-axis.
    """
    width: int
    depth: int
    height: int
    
    def __post_init__(self):
        if self.width <= 0 or self.depth <= 0 or self.height <= 0:
            raise ValueError(
                f"Domain Error: Container dimensions must be strictly positive. "
                f"Got W:{self.width}, D:{self.depth}, H:{self.height}"
            )
            
    @property
    def volume(self) -> int:
        """
        Calculates the total maximum volume of the container.

        Returns:
            int: The volume (width * depth * height).
        """
        return self.width * self.depth * self.height
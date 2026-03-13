import pytest
from src.domain.entities import Box, Container, Position, PackedBox

def test_box_volume_calculation():
    """
    Test if the Box entity correctly calculates its volume.
    """
    box = Box(id="1", width=10, depth=10, height=10)
    assert box.volume == 1000

def test_container_volume_calculation():
    """
    Test if the Container entity correctly calculates its volume.
    """
    container = Container(width=100, depth=100, height=100)
    assert container.volume == 1000000

def test_packed_box_rotation_dimensions():
    """
    Test if the PackedBox correctly returns dimensions based on rotation type.
    Rotation 0: (W, D, H)
    Rotation 1: (W, H, D)
    ... and so on.
    """
    box = Box(id="test-rot", width=1, depth=2, height=3)
    pos = Position(x=0, y=0, z=0)

    assert PackedBox(box=box, position=pos, rotation_type=0).rotated_dimensions == (1, 2, 3)

    assert PackedBox(box=box, position=pos, rotation_type=1).rotated_dimensions == (1, 3, 2)

    assert PackedBox(box=box, position=pos, rotation_type=5).rotated_dimensions == (3, 1, 2)

def test_invalid_rotation_type():
    """
    Test if an invalid rotation type defaults correctly or raises an error 
    depending on implementation.
    """
    box = Box(id="err", width=1, depth=2, height=3)
    pos = Position(0, 0, 0)
    
    with pytest.raises(ValueError) as excinfo:
        PackedBox(box=box, position=pos, rotation_type=99)
    
    assert "Rotation type must be between 0 and 5" in str(excinfo.value)
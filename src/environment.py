"""
environment.py - Grid Environment for Programmable Matter Simulation

This module implements the grid environment for the programmable matter simulation.
It provides a 2D grid where programmable matter elements can move and form shapes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Set, Optional
from enum import Enum


class CellType(Enum):
    """
    Represents the type of a cell in the grid.
    """
    EMPTY = 0
    WALL = 1
    ELEMENT = 2
    TARGET = 3


class Grid:
    """
    Represents a 2D grid environment for programmable matter elements.
    
    The grid consists of cells that can be empty, occupied by an element,
    or be a wall/boundary. Elements can move through the grid according to
    the specified topology (Von Neumann or Moore).
    """
    
    # Cell states
    EMPTY = CellType.EMPTY.value
    WALL = CellType.WALL.value
    ELEMENT = CellType.ELEMENT.value
    TARGET = CellType.TARGET.value
    
    def __init__(self, width: int = 10, height: int = 10, topology: str = "von_neumann"):
        """
        Initialize a grid with specified dimensions.
        
        Args:
            width (int): Width of the grid
            height (int): Height of the grid
            topology (str): Movement topology - "von_neumann" or "moore"
        """
        self.width = width
        self.height = height
        self.topology = topology
        
        # Initialize grid with empty cells
        self.grid = np.zeros((height, width), dtype=int)
        
        # Add boundary walls around the grid
        self._add_boundary_walls()
        
        # Track element positions
        self.element_positions = {}  # element_id -> (row, col)
        self.target_positions = {}   # target_id -> (row, col)
        
    def _add_boundary_walls(self):
        """Add walls around the boundary of the grid."""
        # Top and bottom walls
        self.grid[0, :] = self.WALL
        self.grid[-1, :] = self.WALL
        
        # Left and right walls
        self.grid[:, 0] = self.WALL
        self.grid[:, -1] = self.WALL
    
    def add_element(self, element_id: int, position: Tuple[int, int]) -> bool:
        """
        Add an element to the grid at the specified position.
        
        Args:
            element_id (int): Unique identifier for the element
            position (Tuple[int, int]): (row, col) position on the grid
            
        Returns:
            bool: True if element was added successfully, False otherwise
        """
        row, col = position
        
        # Check if the position is valid
        if not self.is_valid_position(position):
            return False
        
        # Check if the position is empty
        if self.grid[row, col] != self.EMPTY:
            return False
        
        # Add the element
        self.grid[row, col] = self.ELEMENT
        self.element_positions[element_id] = position
        return True
    
    def add_target(self, target_id: int, position: Tuple[int, int]) -> bool:
        """
        Add a target position to the grid.
        
        Args:
            target_id (int): Unique identifier for the target
            position (Tuple[int, int]): (row, col) position on the grid
            
        Returns:
            bool: True if target was added successfully, False otherwise
        """
        row, col = position
        
        # Check if the position is valid
        if not self.is_valid_position(position):
            return False
        
        # Add the target
        # Note: Targets can overlap with elements or other targets
        # We just track them separately
        self.target_positions[target_id] = position
        return True
    
    def move_element(self, element_id: int, new_position: Tuple[int, int]) -> bool:
        """
        Move an element to a new position.
        
        Args:
            element_id (int): ID of the element to move
            new_position (Tuple[int, int]): New (row, col) position
            
        Returns:
            bool: True if movement was successful, False otherwise
        """
        # Check if element exists
        if element_id not in self.element_positions:
            return False
        
        # Check if new position is valid
        if not self.is_valid_position(new_position):
            return False
        
        # Check if new position is empty
        row, col = new_position
        if self.grid[row, col] != self.EMPTY:
            return False
        
        # Remove element from old position
        old_row, old_col = self.element_positions[element_id]
        self.grid[old_row, old_col] = self.EMPTY
        
        # Add element at new position
        self.grid[row, col] = self.ELEMENT
        self.element_positions[element_id] = new_position
        
        return True
    
    def is_valid_position(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is valid (within bounds and not a wall).
        
        Args:
            position (Tuple[int, int]): (row, col) position to check
            
        Returns:
            bool: True if the position is valid, False otherwise
        """
        row, col = position
        
        # Check bounds
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        
        # Check if it's not a wall
        if self.grid[row, col] == self.WALL:
            return False
        
        return True
    
def add_walls_around_border(self):
    """
    Public method to add walls around the border of the grid.
    This is an alias for the private _add_boundary_walls method.
    """
    self._add_boundary_walls()
    
    def is_cell_empty(self, position: Tuple[int, int]) -> bool:
        """
        Check if a cell is empty.
        
        Args:
            position (Tuple[int, int]): (row, col) position to check
            
        Returns:
            bool: True if the cell is empty, False otherwise
        """
        row, col = position
        
        # Check bounds
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        
        return self.grid[row, col] == self.EMPTY
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighboring positions based on the topology.
        
        Args:
            position (Tuple[int, int]): (row, col) position
            
        Returns:
            List[Tuple[int, int]]: List of valid neighboring positions
        """
        row, col = position
        neighbors = []
        
        # Von Neumann topology (up, down, left, right)
        von_neumann_dirs = [
            (-1, 0),  # Up
            (1, 0),   # Down
            (0, -1),  # Left
            (0, 1)    # Right
        ]
        
        # Add diagonal directions for Moore topology
        moore_dirs = von_neumann_dirs + [
            (-1, -1),  # Up-Left
            (-1, 1),   # Up-Right
            (1, -1),   # Down-Left
            (1, 1)     # Down-Right
        ]
        
        # Choose directions based on the topology
        dirs = moore_dirs if self.topology == "moore" else von_neumann_dirs
        
        # Check each neighbor
        for dr, dc in dirs:
            new_row, new_col = row + dr, col + dc
            new_pos = (new_row, new_col)
            
            # Add the neighbor if it's valid
            if self.is_valid_position(new_pos):
                neighbors.append(new_pos)
        
        return neighbors
    
    def add_internal_wall(self, positions: List[Tuple[int, int]]) -> None:
        """
        Add internal walls at specified positions.
        
        Args:
            positions (List[Tuple[int, int]]): List of (row, col) positions
        """
        for row, col in positions:
            if 0 <= row < self.height and 0 <= col < self.width:
                self.grid[row, col] = self.WALL
    
    def is_target_shape_formed(self) -> bool:
        """
        Check if the target shape is formed (all elements at target positions).
        
        Returns:
            bool: True if the shape is formed, False otherwise
        """
        # Get the set of target positions
        target_pos_set = set(self.target_positions.values())
        
        # Get the set of element positions
        element_pos_set = set(self.element_positions.values())
        
        # Check if the target positions match the element positions
        return target_pos_set == element_pos_set
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        Calculate the Manhattan distance between two positions.
        
        Args:
            pos1 (Tuple[int, int]): First position (row, col)
            pos2 (Tuple[int, int]): Second position (row, col)
            
        Returns:
            int: Manhattan distance
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def display_grid(self) -> None:
        """
        Display the grid state as a string representation.
        """
        symbols = {
            self.EMPTY: '.',
            self.WALL: '#',
            self.ELEMENT: 'E',
            self.TARGET: 'T'
        }
        
        # Create a copy of the grid for display
        display_grid = self.grid.copy()
        
        # Mark target positions
        for pos in self.target_positions.values():
            row, col = pos
            # If there's an element at the target, mark it differently
            if display_grid[row, col] == self.ELEMENT:
                display_grid[row, col] = self.TARGET
        
        print(f"Grid ({self.width}x{self.height}):")
        for row in display_grid:
            print("".join(symbols[cell] for cell in row))
        print()
    
    def clear_all_elements(self) -> None:
        """
        Remove all elements from the grid.
        """
        # Clear elements from the grid
        for pos in self.element_positions.values():
            row, col = pos
            self.grid[row, col] = self.EMPTY
        
        # Clear the element positions dictionary
        self.element_positions.clear()
    
    def clear_all_targets(self) -> None:
        """
        Remove all target positions.
        """
        self.target_positions.clear()


# Simple test if run directly
if __name__ == "__main__":
    # Create a 10x10 grid
    grid = Grid(10, 10)
    
    # Add some elements
    grid.add_element(1, (2, 3))
    grid.add_element(2, (4, 5))
    
    # Add some targets
    grid.add_target(1, (5, 5))
    grid.add_target(2, (6, 6))
    
    # Display the grid
    grid.display_grid()
    
    # Move an element
    print("Moving element 1 to (3, 3):")
    grid.move_element(1, (3, 3))
    grid.display_grid()
    
    # Check neighbors
    print("Neighbors of (3, 3):", grid.get_neighbors((3, 3)))
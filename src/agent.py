from enum import Enum
from collections import defaultdict

class Direction(Enum):
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)

class ProgrammableElement:
    """
    A class representing a single programmable matter element.
    """
    
    def __init__(self, x, y, element_id=None):
        """
        Initialize the element with a position.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            element_id: Optional identifier for the element
        """
        self.x = x
        self.y = y
        self.id = element_id
        self.target_x = None
        self.target_y = None
    
    def set_position(self, x, y):
        """Set the element's position."""
        self.x = x
        self.y = y
    
    def set_target(self, x, y):
        """Set the element's target position."""
        self.target_x = x
        self.target_y = y
    
    def has_target(self):
        """Check if the element has a target position."""
        return self.target_x is not None and self.target_y is not None
    
    def is_at_target(self):
        """Check if the element is at its target position."""
        return self.has_target() and self.x == self.target_x and self.y == self.target_y
    
    def distance_to_target(self):
        """Calculate Manhattan distance to target position."""
        if not self.has_target():
            return float('inf')
        return abs(self.x - self.target_x) + abs(self.y - self.target_y)
    
    def get_move_direction(self, to_x, to_y):
        """
        Get the direction to move to reach the given position.
        
        Returns:
            Direction or None if no move is needed
        """
        if self.x == to_x and self.y == to_y:
            return None
        
        dx = to_x - self.x
        dy = to_y - self.y
        
        # Only allow one step at a time in cardinal directions
        if dx > 0 and dy == 0:
            return Direction.RIGHT
        elif dx < 0 and dy == 0:
            return Direction.LEFT
        elif dx == 0 and dy > 0:
            return Direction.DOWN
        elif dx == 0 and dy < 0:
            return Direction.UP
        
        # If not a cardinal direction, return None
        return None
    
    def __str__(self):
        """String representation of the element."""
        target_str = f"target=({self.target_x}, {self.target_y})" if self.has_target() else "no target"
        return f"Element(id={self.id}, pos=({self.x}, {self.y}), {target_str})"

class Controller:
    """
    A centralized controller that manages all programmable matter elements.
    """
    
    def __init__(self, environment):
        """
        Initialize the controller with an environment.
        
        Args:
            environment: The Grid environment
        """
        self.environment = environment
        self.elements = {}  # id -> ProgrammableElement
        self.next_id = 0
        self.target_positions = []
    
    def add_element(self, x, y):
        """
        Add a new element at the specified position.
        
        Returns:
            The element id if successful, None otherwise
        """
        if not self.environment.add_element(x, y):
            return None
        
        element_id = self.next_id
        self.next_id += 1
        
        element = ProgrammableElement(x, y, element_id)
        self.elements[element_id] = element
        
        return element_id
    
    def remove_element(self, element_id):
        """Remove an element by id."""
        if element_id not in self.elements:
            return False
        
        element = self.elements[element_id]
        if not self.environment.remove_element(element.x, element.y):
            return False
        
        del self.elements[element_id]
        return True
    
    def move_element(self, element_id, direction):
        """
        Move an element in the specified direction.
        
        Args:
            element_id: The id of the element to move
            direction: Direction enum value
        
        Returns:
            True if the move was successful, False otherwise
        """
        if element_id not in self.elements:
            return False
        
        element = self.elements[element_id]
        dx, dy = direction.value
        new_x, new_y = element.x + dx, element.y + dy
        
        if not self.environment.move_element(element.x, element.y, new_x, new_y):
            return False
        
        element.set_position(new_x, new_y)
        return True
    
    def add_target_position(self, x, y):
        """Add a target position for the formation."""
        if not self.environment.add_target(x, y):
            return False
        
        self.target_positions.append((x, y))
        return True
    
    def clear_target_positions(self):
        """Clear all target positions."""
        for x, y in self.target_positions:
            if self.environment.is_target(x, y):
                self.environment.grid[y, x] = 0  # Set to EMPTY
        
        self.target_positions = []
    
    def assign_targets(self):
        """
        Assign elements to target positions using a greedy approach.
        Each element is assigned to the closest available target.
        """
        if len(self.elements) > len(self.target_positions):
            return False
        
        # Reset all targets
        for element in self.elements.values():
            element.set_target(None, None)
        
        # Create a copy of target positions
        available_targets = list(self.target_positions)
        
        # Sort elements by some criterion (we'll use id for simplicity)
        sorted_elements = sorted(self.elements.values(), key=lambda e: e.id)
        
        for element in sorted_elements:
            if not available_targets:
                break
            
            # Find the closest target
            closest_target = min(available_targets, 
                                key=lambda t: abs(element.x - t[0]) + abs(element.y - t[1]))
            
            # Assign the target to the element
            element.set_target(closest_target[0], closest_target[1])
            
            # Remove the target from available targets
            available_targets.remove(closest_target)
        
        return True
    
    def is_formation_complete(self):
        """Check if all elements are at their target positions."""
        if len(self.elements) < len(self.target_positions):
            return False
        
        return all(element.is_at_target() for element in self.elements.values())
    
    def get_element_positions(self):
        """Get a dictionary of element positions (id -> (x, y))."""
        return {eid: (element.x, element.y) for eid, element in self.elements.items()}
    
    def get_elements_by_position(self):
        """Get a dictionary of positions -> list of element ids."""
        pos_to_elements = defaultdict(list)
        for eid, element in self.elements.items():
            pos_to_elements[(element.x, element.y)].append(eid)
        return pos_to_elements
    
    def execute_path(self, element_id, path):
        """
        Execute a path for a specific element.
        
        Args:
            element_id: The id of the element
            path: List of (x, y) positions forming a path
        
        Returns:
            List of moves executed (from_x, from_y, to_x, to_y)
        """
        if element_id not in self.elements or not path:
            return []
        
        element = self.elements[element_id]
        moves = []
        
        # First position should be the current position
        if path[0] != (element.x, element.y):
            return []
        
        for i in range(1, len(path)):
            prev_x, prev_y = path[i-1]
            next_x, next_y = path[i]
            
            direction = element.get_move_direction(next_x, next_y)
            if direction is None:
                break
            
            if self.move_element(element_id, direction):
                moves.append((prev_x, prev_y, next_x, next_y))
            else:
                break
        
        return moves
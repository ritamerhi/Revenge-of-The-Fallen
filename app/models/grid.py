import numpy as np

# Grid cell types
EMPTY = 0
WALL = 1
ELEMENT = 2
TARGET = 3

class Grid:
    
    """Represents the 2D grid environment."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Initialize grid with all empty cells
        self.grid = np.zeros((height, width), dtype=np.int8)
        # Set outer boundary as walls
        self.grid[0, :] = WALL
        self.grid[height-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, width-1] = WALL
    
    def clear_grid(self):
        """Clear the grid, preserving walls."""
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] != WALL:
                    self.grid[y, x] = EMPTY
    
    def add_element(self, element):
        """Add an element to the grid."""
        if self.is_playable_position(element.x, element.y):
            self.grid[element.y, element.x] = ELEMENT
    
    def remove_element(self, element):
        """Remove an element from the grid."""
        if self.is_playable_position(element.x, element.y):
            self.grid[element.y, element.x] = EMPTY
    
    def move_element(self, element, new_x, new_y):
        """Move an element to a new position."""
        if not self.is_playable_position(new_x, new_y) or self.is_occupied(new_x, new_y):
            return False
        
        self.grid[element.y, element.x] = EMPTY
        self.grid[new_y, new_x] = ELEMENT
        element.x = new_x
        element.y = new_y
        return True
    
    def is_valid_position(self, x, y):
        """Check if a position is within grid boundaries."""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_playable_position(self, x, y):
        """Check if a position is within the playable area (not on walls)."""
        return self.is_valid_position(x, y) and not self.is_wall(x, y)
    
    def is_wall(self, x, y):
        """Check if a position contains a wall."""
        return self.grid[y, x] == WALL
    
    def is_element(self, x, y):
        """Check if a position contains an element."""
        return self.grid[y, x] == ELEMENT
    
    def is_empty(self, x, y):
        """Check if a position is empty."""
        return self.grid[y, x] == EMPTY
    
    def is_occupied(self, x, y):
        """Check if a position is occupied (wall or element)."""
        return self.grid[y, x] in [WALL, ELEMENT]
    
    def set_target(self, x, y):
        """Mark a position as a target."""
        if self.is_playable_position(x, y):
            self.grid[y, x] = TARGET
            return True
        return False
    
    def get_neighbors(self, x, y, topology="vonNeumann"):
        """Get neighboring positions based on topology."""
        neighbors = []
        
        # Von Neumann neighborhood (4 directions: N, E, S, W)
        if topology == "vonNeumann":
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:  # Up, Right, Down, Left
                nx, ny = x + dx, y + dy
                if self.is_playable_position(nx, ny):
                    neighbors.append((nx, ny))
        
        # Moore neighborhood (8 directions including diagonals)
        elif topology == "moore":
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # Skip the center
                    nx, ny = x + dx, y + dy
                    if self.is_playable_position(nx, ny):
                        neighbors.append((nx, ny))
        
        return neighbors
    
    def visualize(self):
        """Return a string representation of the grid."""
        symbols = {EMPTY: ".", WALL: "#", ELEMENT: "O", TARGET: "X"}
        rows = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                row += symbols[self.grid[y, x]]
            rows.append(row)
        return "\n".join(rows)
        
class Element:
    """Represents a programmable matter element."""
    def __init__(self, element_id, x, y):
        self.id = element_id
        self.x = x
        self.y = y
        self.target_x = None
        self.target_y = None
    
    def set_target(self, x, y):
        """Set the target position for this element."""
        self.target_x = x
        self.target_y = y
    
    def has_target(self):
        """Check if the element has a target assigned."""
        return self.target_x is not None and self.target_y is not None
    
    def distance_to_target(self):
        """Calculate the Manhattan distance to the target."""
        if not self.has_target():
            return float('inf')
        return abs(self.x - self.target_x) + abs(self.y - self.target_y)
    
    def __repr__(self):
        """String representation for debugging."""
        return f"Element(id={self.id}, pos=({self.x},{self.y}), target=({self.target_x},{self.target_y}))"
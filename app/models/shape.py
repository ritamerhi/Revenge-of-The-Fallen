import numpy as np
import math 
class ShapeGenerator:
    """Generates target positions for different shapes."""
    @staticmethod
    def generate_shape(shape_type, num_elements, grid_width, grid_height):
        """Generate target positions for the specified shape."""
        if shape_type == "square":
            return ShapeGenerator.generate_square(num_elements, grid_width, grid_height)
        elif shape_type == "circle":
            return ShapeGenerator.generate_circle(num_elements, grid_width, grid_height)
        elif shape_type == "triangle":
            return ShapeGenerator.generate_triangle(num_elements, grid_width, grid_height)
        elif shape_type == "heart":
            return ShapeGenerator.generate_heart(num_elements, grid_width, grid_height)
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")
        
    @staticmethod
    def generate_square(num_elements, grid_width, grid_height):
        """Generate a square shape with positions matching frontend expectations."""
        # Calculate the side length of the square
        side_length = math.ceil(math.sqrt(num_elements))
        
        # Center the square in the grid
        start_x = (grid_width - side_length) // 2
        start_y = (grid_height - side_length) // 2
        
        # Generate positions in (x,y) format but in a way that matches
        # the frontend's [row, col] visual layout
        positions = []
        for y in range(start_y, start_y + side_length):
            for x in range(start_x, start_x + side_length):
                # Note: We store coordinates as (x,y) where x=col, y=row
                positions.append((x, y))
                if len(positions) >= num_elements:
                    break
            if len(positions) >= num_elements:
                break

        # Debug output to see generated positions
        print(f"SQUARE POSITIONS GENERATED (x,y format):")
        for i, (x, y) in enumerate(positions):
            print(f"  Position {i}: ({x},{y})")
            
        return positions
    
    @staticmethod
    def generate_circle(num_elements, grid_width, grid_height):
        """Generate a circle shape that scales and centers based on grid dimensions."""
        positions = []
        
        # Calculate the radius of the circle
        radius = min(grid_width, grid_height) // 3  # Adjust radius based on grid size
        center_x = grid_width // 2
        center_y = grid_height // 2
        
        # Generate points for the circle
        for i in range(num_elements):
            angle = (2 * math.pi * i) / num_elements  # Distribute points evenly
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            
            # Ensure the point is within the grid bounds
            if 0 <= x < grid_width and 0 <= y < grid_height:
                positions.append((x, y))
        
        # If we have fewer points than requested, fill the remaining positions
        while len(positions) < num_elements:
            positions.append((center_x, center_y))  # Add points at the center if needed
        
        return positions[:num_elements]
    @staticmethod
    def calculate_min_grid_size(self, num_agents):
        """
        Calculate the minimum required grid size for a triangle formation.
        Formula: r(r+1) ≤ num_agents, where r is the number of rows.
        """
        # Formula to solve r(r+1) ≤ num_agents
        r = int((-1 + math.sqrt(1 + 4 * num_agents)) / 2)
        if r < 1 and num_agents >= 2:
            r = 1
        return r + 1  # Add 1 for buffer

    @staticmethod
    def generate_triangle(num_elements, grid_width, grid_height):
        """Generate a triangle shape that scales and centers based on grid dimensions."""
        positions = []
        
        # Calculate the number of rows needed for the triangle
        rows = int(math.sqrt(8 * num_elements + 1) - 1) // 2
        if rows * (rows + 1) // 2 < num_elements:
            rows += 1
        
        # Center the triangle in the grid
        start_x = (grid_width - rows) // 2
        start_y = (grid_height - rows) // 2
        
        # Generate positions for the triangle
        count = 0
        for row in range(rows):
            for col in range(row + 1):
                if count >= num_elements:
                    break
                x = start_x + col
                y = start_y + row
                positions.append((x, y))
                count += 1
        
        return positions[:num_elements]
    
    @staticmethod
    def generate_heart(num_elements, grid_width, grid_height):
        """Generate a heart shape with unique positions for each element."""
        # Predefined heart shape positions - each position should be unique
        heart_positions = [
            (2, 3), (2, 6),                          # Top curves
            (3, 2), (3, 4), (3, 5), (3, 7),          # Upper middle row
            (4, 2), (4, 7),                          # Middle section
            (5, 3), (5, 6),                          # Bottom curves start
            (6, 4), (6, 5),                          # Bottom middle
            (7, 5)                                   # Bottom point
        ]
        
        # Center the heart in the grid
        center_offset_x = (grid_width - 10) // 2
        center_offset_y = (grid_height - 10) // 2
        
        # Apply centering offset
        centered_positions = [(x + center_offset_x, y + center_offset_y) for x, y in heart_positions]
        
        # If we need more positions than defined, we can add variations around the heart
        if num_elements > len(centered_positions):
            # Add positions around the heart's edge if needed
            extra_positions = []
            for x, y in centered_positions:
                # Add slight variations that won't conflict with original heart
                variations = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                for vx, vy in variations:
                    if (vx, vy) not in centered_positions and (vx, vy) not in extra_positions:
                        if 1 < vx < grid_width-1 and 1 < vy < grid_height-1:  # Avoid walls
                            extra_positions.append((vx, vy))
                            if len(centered_positions) + len(extra_positions) >= num_elements:
                                break
                if len(centered_positions) + len(extra_positions) >= num_elements:
                    break
                    
            # Combine original and extra positions
            centered_positions.extend(extra_positions)
        
        # Ensure each position is unique
        unique_positions = []
        seen = set()
        for pos in centered_positions:
            if pos not in seen:
                seen.add(pos)
                unique_positions.append(pos)
        
        # Take only as many positions as needed
        return unique_positions[:num_elements]
    
    @staticmethod
    def validate_positions(positions, grid):
        """Filter out positions that would be invalid in the grid."""
        valid_positions = []
        for x, y in positions:
            if grid.is_valid_position(x, y) and not grid.is_wall(x, y):
                valid_positions.append((x, y))
        return valid_positions
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
        """Generate a square shape with prioritized placement for certain agents."""
    
        def calculate_square_dimensions(num_elements):
            """Calculate the best-fitting width and height for a square-like shape."""
            side_length = int(math.sqrt(num_elements))
        
            if side_length * side_length < num_elements:
                width = side_length
                height = math.ceil(num_elements / width)
            else:
                width = height = side_length
        
            return width, height

        width, height = calculate_square_dimensions(num_elements)
    
        # Center the square in the grid
        start_x = (grid_width - width) // 2
        start_y = (grid_height - height) // 2
    
        positions = []
    
        # Track "problematic" agents (e.g., 5 and 9) to be placed first
        problematic_indices = [5, 9] if num_elements > 9 else [5] if num_elements > 5 else []
        problematic_positions = []

        for i in range(min(height * width, num_elements)):
            y = start_y + (i // width)
            x = start_x + (i % width)

            if i in problematic_indices:
                problematic_positions.append((x, y))
            else:
                positions.append((x, y))
    
        # Ensure problematic agents get priority placement
        positions = problematic_positions + positions
    
        return positions
    
    @staticmethod
    def generate_circle(num_elements, grid_width, grid_height):
        """Generate a circle shape with prioritized placement for certain agents."""
        positions = []
        remaining = num_elements
    
        base_agents = 20  # Original configuration had 20 agents
        row_pattern = [2, 4, 4, 4, 4, 2]
        rows_with_gaps = [2, 3]  # Rows 2 and 3 (third and fourth rows, 0-indexed)
    
        # If we have additional agents beyond the base configuration,
        # adjust the row pattern to add 2 agents per row
        if num_elements > base_agents:
            additional_agents = num_elements - base_agents
        
            # Each set of 12 additional agents adds 2 agents per row to the 6 rows
            sets_of_12 = additional_agents // 12
            remaining_extra = additional_agents % 12
        
            # Modify row pattern to add 2 agents per row for each complete set of 12
            modified_row_pattern = row_pattern.copy()
            for i in range(len(modified_row_pattern)):
                modified_row_pattern[i] += 2 * sets_of_12
        
            # Distribute any remaining extra agents (less than 12) evenly
            distribution_order = [2, 3, 1, 4, 0, 5]  # Priority of rows to add extra agents
        
            for i in range(remaining_extra // 2):  # Add 2 agents at a time
                if i < len(distribution_order):
                    row_idx = distribution_order[i]
                    modified_row_pattern[row_idx] += 2
        
            # Use the modified pattern
            row_pattern = modified_row_pattern
    
        # Place agents according to the pattern
        for row_idx, agents_in_row in enumerate(row_pattern):
            # If we've placed all agents, stop
            if remaining <= 0:
                break
            
            # If we've reached the bottom of the grid, stop
            if row_idx >= grid_height:
                break
            
            # Calculate how many agents to place in this row
            agents_to_place = min(agents_in_row, remaining)
        
            # For rows that need a gap in the middle
            if row_idx in rows_with_gaps:
                agents_per_side = agents_in_row // 2
                gap_size = 2
                left_start = (grid_width - (agents_per_side * 2 + gap_size)) // 2
            
                # Left side agents
                for i in range(agents_per_side):
                    if remaining <= 0:
                        break
                    positions.append((row_idx, left_start + i))
                    remaining -= 1
            
                # Right side agents (after the gap)
                right_start = left_start + agents_per_side + gap_size
                for i in range(agents_per_side):
                    if remaining <= 0:
                        break
                    positions.append((row_idx, right_start + i))
                    remaining -= 1
            else:
                # For other rows, center the agents
                start_col = (grid_width - agents_in_row) // 2
                for i in range(agents_to_place):
                    if remaining <= 0:
                        break
                    positions.append((row_idx, start_col + i))
                    remaining -= 1
    
        # If we still have agents left to place, add them in rows below the pattern
        if remaining > 0:
            current_row = len(row_pattern)
        
            while remaining > 0 and current_row < grid_height:
                agents_to_place = min(grid_width, remaining)
                start_col = (grid_width - agents_to_place) // 2
            
                for i in range(agents_to_place):
                    positions.append((current_row, start_col + i))
                    remaining -= 1
                
                current_row += 1
    
        # Ensure we don't exceed the requested number of agents
        positions = positions[:num_elements]
    
        # Track positions of certain "problematic" agents (5 and 9)
        problematic_indices = [5, 9] if num_elements > 9 else [5] if num_elements > 5 else []
        problematic_positions = []
        normal_positions = []
    
        # Separate problematic agents from normal agents
        for i, pos in enumerate(positions):
            if i in problematic_indices:
                problematic_positions.append(pos)
            else:
                normal_positions.append(pos)
    
        # Now add the problematic positions to the beginning of the list
        positions = problematic_positions + normal_positions
    
        return positions

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
    def generate_triangle(self, num_agents):
        """
        Create a triangle formation where:
        - Top row has exactly 2 agents
        - Each row increases by 2 agents as we go down
        - Bottom row has the maximum number of agents
        """
        # Find how many full rows we can make (using quadratic formula)
        r = int((-1 + math.sqrt(1 + 4 * num_agents)) / 2)
        
        # If we can't even fill the first row with 2 agents, adjust
        if r < 1 and num_agents >= 2:
            r = 1
        
        # Calculate how many agents we'll use in complete rows
        agents_in_complete_rows = r * (r + 1)
        
        # Remaining agents for the last partial row (if any)
        remaining_agents = num_agents - agents_in_complete_rows
        
        # Determine agents per row (starting from the top with 2 agents)
        agents_per_row = []
        for i in range(r):
            agents_per_row.append(2 * (i + 1))  # 2, 4, 6, 8, ...
        
        # Add the last partial row if needed
        if remaining_agents > 0:
            agents_per_row.append(remaining_agents)
        
        # Now create positions for each agent
        positions = []
        for row, num_in_row in enumerate(agents_per_row):
            # Center the agents in this row
            start_col = (self.grid_size - num_in_row) // 2
            
            # Add agents for this row
            for col in range(num_in_row):
                positions.append((row, start_col + col))
        
        return positions

    
    @staticmethod
    def generate_heart(num_elements, grid_width, grid_height):
        """Generate a heart shape."""
        # Predefined heart shape positions
        heart_positions = [
            (2, 3), (2, 6),
            (3, 2), (3, 4), (3, 5), (3, 7),
            (4, 2), (4, 7),
            (5, 3), (5, 6),
            (6, 4), (6, 5),
            (7, 5)
        ]
        
        # Center the heart
        center_offset_x = (grid_width - 10) // 2
        center_offset_y = (grid_height - 10) // 2
        
        centered_positions = [(x + center_offset_x, y + center_offset_y) for x, y in heart_positions]
        
        # Take only as many positions as needed
        return centered_positions[:num_elements]
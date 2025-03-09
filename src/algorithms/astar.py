import heapq

def manhattan_distance(x1, y1, x2, y2):
    """Calculate the Manhattan distance between two points."""
    return abs(x1 - x2) + abs(y1 - y2)

def astar_pathfind(grid, start_x, start_y, goal_x, goal_y):
    """
    Find a path from start to goal using A* search with Manhattan distance heuristic.
    
    Args:
        grid: The Grid environment
        start_x, start_y: Starting position
        goal_x, goal_y: Goal position
    
    Returns:
        List of (x, y) positions forming a path from start to goal,
        or None if no path exists
    """
    # Check if start and goal are valid positions
    if not grid.is_valid_position(start_x, start_y) or not grid.is_valid_position(goal_x, goal_y):
        return None
    
    # Check if start or goal are walls
    if grid.is_wall(start_x, start_y) or grid.is_wall(goal_x, goal_y):
        return None
    
    # If start is the goal, return a single-element path
    if start_x == goal_x and start_y == goal_y:
        return [(start_x, start_y)]
    
    # Define possible moves (up, right, down, left)
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    
    # Priority queue for A*
    open_set = []
    heapq.heappush(open_set, (0, (start_x, start_y)))  # (f_score, position)
    
    # Set to keep track of positions in the open set
    open_set_positions = {(start_x, start_y)}
    
    # Dictionary to store the g_score (cost from start) for each position
    g_score = {(start_x, start_y): 0}
    
    # Dictionary to store the f_score (g_score + heuristic) for each position
    f_score = {(start_x, start_y): manhattan_distance(start_x, start_y, goal_x, goal_y)}
    
    # Dictionary to store the parent of each position for path reconstruction
    parent = {}
    
    while open_set:
        # Get the position with the lowest f_score
        current_f, (current_x, current_y) = heapq.heappop(open_set)
        open_set_positions.remove((current_x, current_y))
        
        # Check if we reached the goal
        if current_x == goal_x and current_y == goal_y:
            # Reconstruct path
            path = [(current_x, current_y)]
            while (path[-1] != (start_x, start_y)):
                path.append(parent[path[-1]])
            path.reverse()
            return path
        
        # Check all four directions
        for dx, dy in directions:
            next_x, next_y = current_x + dx, current_y + dy
            
            # Check if the next position is valid
            if grid.is_valid_position(next_x, next_y):
                # Check if the next position is not a wall and not occupied by another element
                if not grid.is_wall(next_x, next_y) and not grid.is_element(next_x, next_y):
                    # Calculate tentative g_score
                    tentative_g_score = g_score[(current_x, current_y)] + 1
                    
                    # If this path is better than any previous one
                    if (next_x, next_y) not in g_score or tentative_g_score < g_score[(next_x, next_y)]:
                        # Record the path
                        parent[(next_x, next_y)] = (current_x, current_y)
                        g_score[(next_x, next_y)] = tentative_g_score
                        
                        # Calculate f_score
                        next_f_score = tentative_g_score + manhattan_distance(next_x, next_y, goal_x, goal_y)
                        f_score[(next_x, next_y)] = next_f_score
                        
                        # Add to the open set if not already there
                        if (next_x, next_y) not in open_set_positions:
                            heapq.heappush(open_set, (next_f_score, (next_x, next_y)))
                            open_set_positions.add((next_x, next_y))
    
    # If no path is found
    return None

def astar_multi_element(controller, grid):
    """
    Find paths for all elements to their targets using A*.
    
    Args:
        controller: The Controller managing the elements
        grid: The Grid environment
    
    Returns:
        Dictionary mapping element IDs to paths
    """
    # Assign target positions if not already assigned
    controller.assign_targets()
    
    # Dictionary to store paths for each element
    element_paths = {}
    
    # Process elements one by one, starting with those closest to their targets
    sorted_elements = sorted(
        controller.elements.values(),
        key=lambda e: e.distance_to_target() if e.has_target() else float('inf')
    )
    
    for element in sorted_elements:
        if not element.has_target():
            continue
        
        # Temporarily remove the element from the grid for pathfinding
        grid.grid[element.y, element.x] = 0  # Set to EMPTY
        
        # Find a path for this element
        path = astar_pathfind(grid, element.x, element.y, element.target_x, element.target_y)
        
        # Put the element back
        grid.grid[element.y, element.x] = 2  # Set to ELEMENT
        
        # Store the path if one was found
        if path:
            element_paths[element.id] = path
    
    return element_paths
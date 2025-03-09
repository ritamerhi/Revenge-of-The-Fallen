from collections import deque

def bfs_pathfind(grid, start_x, start_y, goal_x, goal_y):
    """
    Find a path from start to goal using Breadth-First Search.
    
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
    
    # Queue for BFS
    queue = deque([(start_x, start_y)])
    
    # Set to keep track of visited positions
    visited = set([(start_x, start_y)])
    
    # Dictionary to store the parent of each position for path reconstruction
    parent = {}
    
    while queue:
        current_x, current_y = queue.popleft()
        
        # Check all four directions
        for dx, dy in directions:
            next_x, next_y = current_x + dx, current_y + dy
            
            # Check if the next position is valid and not visited
            if (next_x, next_y) not in visited and grid.is_valid_position(next_x, next_y):
                # Check if the next position is not a wall and not occupied by another element
                if not grid.is_wall(next_x, next_y) and not grid.is_element(next_x, next_y):
                    # Add to the queue and mark as visited
                    queue.append((next_x, next_y))
                    visited.add((next_x, next_y))
                    parent[(next_x, next_y)] = (current_x, current_y)
                    
                    # Check if we reached the goal
                    if next_x == goal_x and next_y == goal_y:
                        # Reconstruct path
                        path = [(next_x, next_y)]
                        while (path[-1] != (start_x, start_y)):
                            path.append(parent[path[-1]])
                        path.reverse()
                        return path
    
    # If no path is found
    return None

def bfs_multi_element(controller, grid):
    """
    Find paths for all elements to their targets using BFS.
    
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
    
    # Process elements one by one
    for element_id, element in controller.elements.items():
        if not element.has_target():
            continue
        
        # Temporarily remove the element from the grid for pathfinding
        grid.grid[element.y, element.x] = 0  # Set to EMPTY
        
        # Find a path for this element
        path = bfs_pathfind(grid, element.x, element.y, element.target_x, element.target_y)
        
        # Put the element back
        grid.grid[element.y, element.x] = 2  # Set to ELEMENT
        
        # Store the path if one was found
        if path:
            element_paths[element_id] = path
    
    return element_paths
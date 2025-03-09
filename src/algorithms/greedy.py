import heapq

def manhattan_distance(x1, y1, x2, y2):
    """Calculate the Manhattan distance between two points."""
    return abs(x1 - x2) + abs(y1 - y2)

def greedy_pathfind(grid, start_x, start_y, goal_x, goal_y):
    """
    Find a path from start to goal using greedy best-first search with Manhattan distance.
    
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
    
def greedy_multi_element(controller, grid):
    """
    Find paths for all elements to their targets using greedy best-first search.
    
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
        path = greedy_pathfind(grid, element.x, element.y, element.target_x, element.target_y)
        
        # Put the element back
        grid.grid[element.y, element.x] = 2  # Set to ELEMENT
        
        # Store the path if one was found
        if path:
            element_paths[element.id] = path
    
    return element_paths
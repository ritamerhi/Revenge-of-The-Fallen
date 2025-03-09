import argparse
import time
import numpy as np
from enum import Enum

from environment import Grid, CellType
from visualization import GridVisualizer
from agent import Controller, Direction
from algorithms.bfs import bfs_pathfind, bfs_multi_element
from algorithms.astar import astar_pathfind, astar_multi_element
from algorithms.greedy import greedy_pathfind, greedy_multi_element

class Algorithm(Enum):
    BFS = "bfs"
    ASTAR = "astar"
    GREEDY = "greedy"

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Programmable Matter Search Agent')
    
    parser.add_argument('--grid-size', type=int, default=10,
                        help='Size of the grid (default: 10)')
    
    parser.add_argument('--algorithm', type=str, default='astar',
                        choices=['bfs', 'astar', 'greedy'],
                        help='Search algorithm to use (default: astar)')
    
    parser.add_argument('--shape', type=str, default='square',
                        choices=['square', 'circle', 'triangle'],
                        help='Target shape to form (default: square)')
    
    parser.add_argument('--num-elements', type=int, default=4,
                        help='Number of programmable matter elements (default: 4)')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the grid and movements')
    
    parser.add_argument('--save-animation', type=str, default=None,
                        help='Save animation to the specified file (.gif or .mp4)')
    
    parser.add_argument('--random-seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def generate_shape_targets(shape, grid_size, shape_size=3, center_x=None, center_y=None):
    """
    Generate target positions for the specified shape.
    
    Args:
        shape: Shape type ('square', 'circle', or 'triangle')
        grid_size: Size of the grid
        shape_size: Size parameter for the shape
        center_x, center_y: Center coordinates (defaults to grid center if None)
    
    Returns:
        List of (x, y) target positions
    """
    if center_x is None:
        center_x = grid_size // 2
    if center_y is None:
        center_y = grid_size // 2
    
    targets = []
    
    if shape == 'square':
        # Generate a square with specified size
        half_size = shape_size // 2
        for dx in range(-half_size, half_size + 1):
            for dy in range(-half_size, half_size + 1):
                x, y = center_x + dx, center_y + dy
                if 0 < x < grid_size - 1 and 0 < y < grid_size - 1:
                    targets.append((x, y))
    
    elif shape == 'circle':
        # Generate an approximate circle with specified radius
        radius = shape_size // 2
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                # Use distance formula to check if point is in circle
                if dx**2 + dy**2 <= radius**2:
                    x, y = center_x + dx, center_y + dy
                    if 0 < x < grid_size - 1 and 0 < y < grid_size - 1:
                        targets.append((x, y))
    
    elif shape == 'triangle':
        # Generate a triangle with specified height
        for dy in range(shape_size):
            # Width at each row is proportional to height
            width = (dy + 1) * 2 - 1
            half_width = width // 2
            for dx in range(-half_width, half_width + 1):
                x, y = center_x + dx, center_y + dy
                if 0 < x < grid_size - 1 and 0 < y < grid_size - 1:
                    targets.append((x, y))
    
    return targets

def place_random_elements(grid, controller, num_elements, seed=None):
    """
    Place elements at random positions on the grid.
    
    Args:
        grid: The Grid environment
        controller: The Controller managing the elements
        num_elements: Number of elements to place
        seed: Random seed for reproducibility
    
    Returns:
        Number of elements successfully placed
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get all empty cells
    empty_cells = []
    for y in range(grid.height):
        for x in range(grid.width):
            if grid.is_cell_empty((x, y)):
                empty_cells.append((x, y))
    
    # Shuffle the empty cells
    np.random.shuffle(empty_cells)
    
    # Place elements
    count = 0
    for x, y in empty_cells:
        if count >= num_elements:
            break
        
        if controller.add_element(x, y) is not None:
            count += 1
    
    return count

def execute_algorithm(algorithm, controller, grid):
    """
    Execute the specified search algorithm.
    
    Args:
        algorithm: Algorithm enum value
        controller: The Controller managing the elements
        grid: The Grid environment
    
    Returns:
        Dictionary mapping element IDs to paths
    """
    if algorithm == Algorithm.BFS:
        return bfs_multi_element(controller, grid)
    elif algorithm == Algorithm.ASTAR:
        return astar_multi_element(controller, grid)
    elif algorithm == Algorithm.GREEDY:
        return greedy_multi_element(controller, grid)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

def execute_paths(controller, element_paths, visualizer=None, save_animation=None):
    """
    Execute the paths for all elements.
    
    Args:
        controller: The Controller managing the elements
        element_paths: Dictionary mapping element IDs to paths
        visualizer: Optional GridVisualizer for visualization
        save_animation: Optional filename to save animation
    
    Returns:
        List of all moves executed (from_x, from_y, to_x, to_y)
    """
    all_moves = []
    
    # Sort elements by distance to target (closest first)
    sorted_elements = sorted(
        controller.elements.values(),
        key=lambda e: len(element_paths.get(e.id, [])) if e.id in element_paths else float('inf')
    )
    
    for element in sorted_elements:
        if element.id not in element_paths:
            continue
        
        path = element_paths[element.id]
        moves = controller.execute_path(element.id, path)
        all_moves.extend(moves)
    
    # Create and save animation if requested
    if visualizer and all_moves and save_animation:
        ani = visualizer.save_animation(all_moves, save_animation)
    
    return all_moves

def main():
    """Main function."""
    args = parse_arguments()
    
    # Set random seed if specified
    if args.random_seed is not None:
        np.random.seed(args.random_seed)
    
    # Create grid
    grid = Grid(args.grid_size, args.grid_size)
    grid.add_walls_around_border()  # This now works!
    
    # Create controller
    controller = Controller(grid)
    
    # Create visualizer if needed
    visualizer = GridVisualizer(grid) if args.visualize else None
    
    # Generate target positions for the specified shape
    targets = generate_shape_targets(args.shape, args.grid_size)
    
    # Add target positions to the grid
    for target_id, (x, y) in enumerate(targets):
        grid.add_target(target_id, (x, y))
    
    # Place random elements on the grid
    num_placed = place_random_elements(grid, controller, 
                                      min(args.num_elements, len(targets)),
                                      args.random_seed)
    
    print(f"Placed {num_placed} elements out of {args.num_elements} requested")
    print(f"Target shape: {args.shape} with {len(targets)} positions")
    
    # Print initial grid state
    print("\nInitial Grid:")
    grid.display_grid()
    
    # Visualize initial state if requested
    if visualizer:
        visualizer.update_visualization()
    
    # Execute the specified algorithm
    algorithm = Algorithm(args.algorithm)
    print(f"\nExecuting {algorithm.value} algorithm...")
    
    start_time = time.time()
    element_paths = execute_algorithm(algorithm, controller, grid)
    end_time = time.time()
    
    print(f"Algorithm completed in {end_time - start_time:.4f} seconds")
    print(f"Found paths for {len(element_paths)} out of {len(controller.elements)} elements")
    
    # Execute the paths
    print("\nExecuting paths...")
    all_moves = execute_paths(controller, element_paths, visualizer, args.save_animation)
    
    print(f"Executed {len(all_moves)} moves")
    
    # Print final grid state
    print("\nFinal Grid:")
    grid.display_grid()
    
    # Check if formation is complete
    if grid.is_target_shape_formed():
        print("\nFormation complete! All elements are at their target positions.")
    else:
        print("\nFormation incomplete. Some elements could not reach their targets.")
    
    # Show final state if visualization is enabled
    if visualizer:
        visualizer.show()

if __name__ == "__main__":
    main()
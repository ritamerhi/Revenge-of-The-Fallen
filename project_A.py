import heapq
import numpy as np
from queue import Queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- L-System for Target Shape Generation ---
def l_system(axiom, rules, iterations):
    """Generate L-System pattern."""
    for _ in range(iterations):
        axiom = ''.join(rules.get(c, c) for c in axiom)
    return axiom

# --- Cellular Automata for Local Motion Control ---
def initialize_grid(n, m, num_blocks):
    """Initialize grid with blocks on the last 2 lines."""
    grid = np.zeros((n, m), dtype=int)  # 0 = empty, 1 = block
    blocks = []
    
    # Place 10 blocks in the 9th row, 10 blocks in the 10th row
    for i in range(10):
        grid[8, i] = 1  # 9th row (index 8)
        blocks.append((8, i))
    for i in range(10):
        grid[9, i] = 1  # 10th row (index 9)
        blocks.append((9, i))
        
    return grid, blocks

def print_grid(grid):
    """Print the grid."""
    for row in grid:
        print(' '.join(str(cell) for cell in row))
    print()

def move_block_to_target(block_pos, target_pos):
    """Move block to the target position."""
    x, y = block_pos
    tx, ty = target_pos
    dx, dy = np.sign(tx - x), np.sign(ty - y)

    return (x + dx, y + dy)

# --- A* Search Algorithm for Navigation ---
def a_star_search(grid, start, target):
    """Find the optimal path using A* search."""
    # Priority queue for the frontier
    pq = []
    heapq.heappush(pq, (0, start, []))  # (priority, node, path)

    # Cost dictionary to store the minimum cost of reaching each node
    costs = {start: 0}
    visited = set()

    while pq:
        priority, current, path = heapq.heappop(pq)
        
        # Check if target is reached
        if current == target:
            return path

        visited.add(current)

        # Explore neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] != 1:
                neighbor = (nx, ny)
                
                # Calculate the new cost to reach the neighbor
                new_cost = costs[current] + 1
                
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    # Heuristic: Manhattan distance to the target
                    heuristic = abs(nx - target[0]) + abs(ny - target[1])
                    priority = new_cost + heuristic
                    heapq.heappush(pq, (priority, neighbor, path + [neighbor]))

    return []  # Return an empty list if no path found


# --- Visualization ---
def visualize_grid(grid, title="Grid"):
    """Visualize the grid using matplotlib with enhanced clarity."""
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="viridis", vmin=0, vmax=2)
    plt.title(title)
    plt.grid(True, which='major', color='white', linestyle='-', alpha=0.3)
    plt.xticks(range(grid.shape[1]))
    plt.yticks(range(grid.shape[0]))
    plt.colorbar(ticks=[0, 1, 2], label="Cell State (0: Empty, 1: Block, 2: Target)")
    plt.show()

def animate_grid(grid_history):
    """Animate the grid history with enhanced clarity."""
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(grid_history[0], cmap="viridis", vmin=0, vmax=2)
    plt.colorbar(im, ticks=[0, 1, 2], label="Cell State (0: Empty, 1: Block, 2: Target)")
    ax.grid(True, which='major', color='white', linestyle='-', alpha=0.3)
    ax.set_xticks(range(grid_history[0].shape[1]))
    ax.set_yticks(range(grid_history[0].shape[0]))

    def update(frame):
        im.set_data(grid_history[frame])
        ax.set_title(f"Step {frame + 1}")
        return im

    ani = animation.FuncAnimation(fig, update, frames=len(grid_history), interval=500, repeat=False)
    plt.show()
    
# --- Main Program ---
# L-System Configuration
axiom = "F"
iterations = 2
rules = {
    "F": "F+F-F-F+F",  # Square pattern rule
    "+": "+",  # Turning right rule
    "-": "-"   # Turning left rule
}

# Function to generate shapes
def generate_shape(shape_type, num_blocks=20):
    """Generate positions for circle, triangle, or square shapes."""
    positions = []
    n, m = 10, 10  # Grid size
    center_x, center_y = n // 2, m // 2

    if shape_type == "circle":
        positions = generate_circle_shape_with_hole(center_x, center_y, num_blocks)
        return positions

    elif shape_type == "triangle":
        # Calculate the optimal number of rows and blocks per row
        rows, row_distributions = determine_triangle_structure(num_blocks)

        # Generate positions based on the distribution
        for row_idx, num_blocks_in_row in row_distributions:
            # Center the blocks in this row
            start_col = center_y - (num_blocks_in_row // 2)

            # Add positions for this row
            for col in range(num_blocks_in_row):
                positions.append((row_idx, start_col + col))

        return positions
    
    else:  # square
        # Use A* to find optimal square dimensions
        side_length = int(np.ceil(np.sqrt(num_blocks)))
        start_row = (n - side_length) // 2
        start_col = (m - side_length) // 2
        
        for i in range(side_length):
            for j in range(side_length):
                if len(positions) < num_blocks:
                    positions.append((start_row + i, start_col + j))
        
        return positions[:num_blocks]

def determine_triangle_structure(num_blocks):
    """
    Calculate the optimal number of rows and block distribution
    for a perfect triangle where the top has the fewest blocks, and each row below
    increases until the base. All blocks must be taken into account.
    """
    rows = 0
    total_blocks = 0
    row_distributions = []

    # Determine the maximum possible rows while taking all blocks into account
    while total_blocks + (2 * rows + 1) <= num_blocks:
        current_row_blocks = 2 * rows + 1
        row_distributions.append((rows, current_row_blocks))
        total_blocks += current_row_blocks
        rows += 1

    # Redistribute any remaining blocks to maintain a triangular structure
    remaining_blocks = num_blocks - total_blocks
    for i in range(remaining_blocks):
        row_distributions[i % len(row_distributions)] = (
            row_distributions[i % len(row_distributions)][0],
            row_distributions[i % len(row_distributions)][1] + 1
        )

    return rows, row_distributions



def a_star_circle_radius(num_blocks, center_x, center_y):
    """Use A* search to find optimal circle radius."""
    # Start with small radius and expand until we can fit all blocks
    radius = 1
    while True:
        block_positions = []
        for i in range(10):
            for j in range(10):
                if ((i - center_x)**2 + (j - center_y)**2)**0.5 <= radius:
                    block_positions.append((i, j))
        if len(block_positions) >= num_blocks:
            return radius
        radius += 0.5

def find_optimal_radius_for_empty_center(grid_size, num_blocks, empty_radius):
    center = grid_size // 2
    radius = 1

    while True:
        block_positions = []
        empty_positions = []

        # Search for positions inside the growing circle
        for i in range(grid_size):
            for j in range(grid_size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)

                # Exclude positions within empty center radius
                if dist <= radius and dist > empty_radius:
                    block_positions.append((i, j))
                elif dist <= empty_radius:
                    empty_positions.append((i, j))

        # Check if we have enough block positions
        if len(block_positions) >= num_blocks:
            break

        radius += 0.5

    return radius, block_positions, empty_positions


def generate_circle_shape_with_hole(center_x, center_y, num_blocks, grid_size=10, empty_radius=1.5):
    """Generate circular positions around the center with an empty hole."""
    
    # Get optimal radius and block positions
    optimal_radius, block_positions, _ = find_optimal_radius_for_empty_center(
        grid_size, num_blocks, empty_radius
    )

    # Filter positions around the desired center within computed radius
    positions = [
        (i, j) for (i, j) in block_positions
        if ((i - center_x)**2 + (j - center_y)**2)**0.5 <= optimal_radius
    ]

    # Return the exact number of needed positions
    return positions[:num_blocks]

# Function to map each agent to a unique target position
def assign_targets(blocks, target_positions):
    """Assign target positions for each block."""
    return {blocks[i]: target_positions[i] for i in range(len(blocks))}

# Visual Feedback on Agent Movements
def visualize_block_paths(block_paths):
    """Display movement paths for blocks towards target shape."""
    print("Block Movement Paths:")
    for block, path in block_paths.items():
        print(f"Block {block} path: {path}")

# User input for shape selection
shape_type = input("Select shape (circle, triangle, square): ").lower()

# Generate the target pattern based on the user's selection
target_positions = generate_shape(shape_type)
print(f"Generated {shape_type} shape positions: {target_positions}")

# Get grid dimensions
n, m = 10, 10
num_blocks = len(target_positions)

# Calculate the center of the grid
center_x, center_y = n // 2, m // 2

# Adjust positions based on the shape's size and the grid's center
max_x = max([pos[0] for pos in target_positions])
max_y = max([pos[1] for pos in target_positions])

# Calculate the offset
offset_x = center_x - (max_x // 2)  # Center the shape horizontally
offset_y = center_y - (max_y // 2)  # Center the shape vertically

# Apply the offset to each position in the target shape
target_positions_centered = [(x + offset_x, y + offset_y) for (x, y) in target_positions]
target_positions_centered = target_positions_centered[:num_blocks]
print(f"Centered {shape_type} shape positions: {target_positions_centered}")

# Initialize the grid for Cellular Automata
grid, blocks = initialize_grid(n, m, num_blocks)

# Display Initial Grid
print("Initial Grid:")
print_grid(grid)
visualize_grid(grid, title="Initial Grid")

# Move the blocks towards the target shape (15 steps)
grid_history = [grid.copy()]
for step in range(15):
    for i in range(num_blocks):  # Ensure all blocks are moved toward their target
        target_pos = target_positions_centered[i % len(target_positions_centered)]
        grid[blocks[i][0], blocks[i][1]] = 0  # Remove block from current position
        blocks[i] = move_block_to_target(blocks[i], target_pos)
        grid[blocks[i][0], blocks[i][1]] = 1  # Place block in new position
    
    grid_history.append(grid.copy())

    print(f"Step {step + 1}:")
    print_grid(grid)
    print(f"Blocks after step {step + 1}: {blocks}")


# Animate the grid history
animate_grid(grid_history)

# Use A* Search to find the path from the agent to the target
path = a_star_search(grid, blocks[0], target_positions_centered[0])
print("A* Path from First Block to Target:", path)
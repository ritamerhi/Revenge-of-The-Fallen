# Grid configuration
GRID_SIZE = 10

# Define movement directions based on topology
MOVEMENT_DIRECTIONS = {
    "vonNeumann": [(0, 1), (1, 0), (0, -1), (-1, 0)],  # Right, Down, Left, Up
    "moore": [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]  # 8 directions
}

# In-memory storage for current simulation state
simulation_state = {
    "agents": [],
    "target_positions": [],
    "grid": None,
    "agent_moved": [],
    "current_agent_idx": None,
    "path": None,
    "movement_step": 0,
    "algorithm": "astar",
    "topology": "vonNeumann",
    "movement": "sequential",
    "collision_enabled": True,
    "shape": "square"
}

# Function to generate a circle formation
def generate_circle_shape(num_agents):
    positions = []
    remaining = num_agents
    
    # Base configuration for the circle shape
    base_agents = 20  # Original configuration had 20 agents
    
    # Define how many agents should be in each row for the base configuration
    row_pattern = [2, 4, 4, 4, 4, 2]
    
    # Define which rows should have gaps in the middle
    rows_with_gaps = [2, 3]  # Rows 2 and 3 (third and fourth rows, 0-indexed)
    
    # If we have additional agents beyond the base configuration,
    # adjust the row pattern to add 2 agents per row
    if num_agents > base_agents:
        # Calculate additional agents beyond the base
        additional_agents = num_agents - base_agents
        
        # Each set of 12 additional agents adds 2 agents per row to the 6 rows
        sets_of_12 = additional_agents // 12
        remaining_extra = additional_agents % 12
        
        # Modify row pattern to add 2 agents per row for each complete set of 12
        modified_row_pattern = row_pattern.copy()
        for i in range(len(modified_row_pattern)):
            modified_row_pattern[i] += 2 * sets_of_12
        
        # Distribute any remaining extra agents (less than 12) evenly
        # starting from the middle rows
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
        if row_idx >= GRID_SIZE:
            break
            
        # Calculate how many agents to place in this row
        agents_to_place = min(agents_in_row, remaining)
        
        # For rows that need a gap in the middle
        if row_idx in rows_with_gaps:
            # Calculate the number of agents per side
            agents_per_side = agents_in_row // 2
            
            # Calculate the size of the gap (always maintain 2 empty cells in the middle)
            gap_size = 2
            
            # Calculate the starting column for the left side
            # This centers the formation with the gap in the middle
            left_start = (GRID_SIZE - (agents_per_side * 2 + gap_size)) // 2
            
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
            start_col = (GRID_SIZE - agents_in_row) // 2
            for i in range(agents_to_place):
                if remaining <= 0:
                    break
                positions.append((row_idx, start_col + i))
                remaining -= 1
    
    # If we still have agents left to place, add them in rows below the pattern
    if remaining > 0:
        current_row = len(row_pattern)
        
        while remaining > 0 and current_row < GRID_SIZE:
            # Place up to 10 agents per row (grid width)
            agents_to_place = min(GRID_SIZE, remaining)
            start_col = (GRID_SIZE - agents_to_place) // 2
            
            for i in range(agents_to_place):
                positions.append((current_row, start_col + i))
                remaining -= 1
                
            current_row += 1
    
    # Ensure we don't exceed the requested number of agents
    positions = positions[:num_agents]
    
    return positions

# Function to generate a triangle formation
def generate_triangle_shape(num_agents):
    """
    Create a triangle where:
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
        start_col = (GRID_SIZE - num_in_row) // 2
        
        # Add agents for this row
        for col in range(num_in_row):
            positions.append((row, start_col + col))
    
    return positions

# Function to determine square dimensions based on number of agents
def calculate_square_dimensions(num_agents):
    # Find the closest square root
    side_length = int(math.sqrt(num_agents))
    
    # If num_agents is not a perfect square, adjust
    if side_length * side_length < num_agents:
        width = side_length
        height = math.ceil(num_agents / width)
    else:
        width = height = side_length
        
    return width, height

# Function to generate a square formation
def generate_square_shape(num_agents):
    width, height = calculate_square_dimensions(num_agents)
    
    # Center the square in the grid
    start_row = (GRID_SIZE - height) // 2
    start_col = (GRID_SIZE - width) // 2
    
    positions = []
    for i in range(min(height * width, num_agents)):
        row = start_row + (i // width)
        col = start_col + (i % width)
        positions.append((row, col))
    
    return positions

# Function to generate a heart shape
def generate_heart_shape(num_agents):
    # Define a basic heart shape pattern (use scaling based on number of agents)
    heart_positions = [
        (2, 3), (2, 6),
        (3, 2), (3, 4), (3, 5), (3, 7),
        (4, 2), (4, 7),
        (5, 3), (5, 6),
        (6, 4), (6, 5),
        (7, 5)
    ]
    
    # If we need more or fewer agents, scale the heart
    if num_agents < len(heart_positions):
        # Use only the first n positions
        return heart_positions[:num_agents]
    elif num_agents > len(heart_positions):
        # For simplicity, add extra positions around the heart
        extra_needed = num_agents - len(heart_positions)
        extra_positions = []
        
        # Add positions around the existing heart shape
        # This is a simple approach - could be improved for better distribution
        for row in range(2, 8):
            for col in range(2, 8):
                if (row, col) not in heart_positions:
                    extra_positions.append((row, col))
                    if len(extra_positions) >= extra_needed:
                        break
            if len(extra_positions) >= extra_needed:
                break
        
        return heart_positions + extra_positions[:extra_needed]

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import heapq
import math
import time
import os

app = Flask(__name__, static_folder='static')

# Grid configuration
GRID_SIZE = 10

# Define movement directions based on topology
MOVEMENT_DIRECTIONS = {
    "vonNeumann": [(0, 1), (1, 0), (0, -1), (-1, 0)],  # Right, Down, Left, Up
    "moore": [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]  # 8 directions
}

# In-memory storage for current simulation state
simulation_state = {
    "agents": [],
    "target_positions": [],
    "grid": None,
    "agent_moved": [],
    "current_agent_idx": None,
    "path": None,
    "movement_step": 0,
    "algorithm": "astar",
    "topology": "vonNeumann",
    "movement": "sequential",
    "collision_enabled": True,
    "shape": "square"
}

# Function to generate a circle formation
def generate_circle_shape(num_agents):
    positions = []
    remaining = num_agents
    
    # Base configuration for the circle shape
    base_agents = 20  # Original configuration had 20 agents
    
    # Define how many agents should be in each row for the base configuration
    row_pattern = [2, 4, 4, 4, 4, 2]
    
    # Define which rows should have gaps in the middle
    rows_with_gaps = [2, 3]  # Rows 2 and 3 (third and fourth rows, 0-indexed)
    
    # If we have additional agents beyond the base configuration,
    # adjust the row pattern to add 2 agents per row
    if num_agents > base_agents:
        # Calculate additional agents beyond the base
        additional_agents = num_agents - base_agents
        
        # Each set of 12 additional agents adds 2 agents per row to the 6 rows
        sets_of_12 = additional_agents // 12
        remaining_extra = additional_agents % 12
        
        # Modify row pattern to add 2 agents per row for each complete set of 12
        modified_row_pattern = row_pattern.copy()
        for i in range(len(modified_row_pattern)):
            modified_row_pattern[i] += 2 * sets_of_12
        
        # Distribute any remaining extra agents (less than 12) evenly
        # starting from the middle rows
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
        if row_idx >= GRID_SIZE:
            break
            
        # Calculate how many agents to place in this row
        agents_to_place = min(agents_in_row, remaining)
        
        # For rows that need a gap in the middle
        if row_idx in rows_with_gaps:
            # Calculate the number of agents per side
            agents_per_side = agents_in_row // 2
            
            # Calculate the size of the gap (always maintain 2 empty cells in the middle)
            gap_size = 2
            
            # Calculate the starting column for the left side
            # This centers the formation with the gap in the middle
            left_start = (GRID_SIZE - (agents_per_side * 2 + gap_size)) // 2
            
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
            start_col = (GRID_SIZE - agents_in_row) // 2
            for i in range(agents_to_place):
                if remaining <= 0:
                    break
                positions.append((row_idx, start_col + i))
                remaining -= 1
    
    # If we still have agents left to place, add them in rows below the pattern
    if remaining > 0:
        current_row = len(row_pattern)
        
        while remaining > 0 and current_row < GRID_SIZE:
            # Place up to 10 agents per row (grid width)
            agents_to_place = min(GRID_SIZE, remaining)
            start_col = (GRID_SIZE - agents_to_place) // 2
            
            for i in range(agents_to_place):
                positions.append((current_row, start_col + i))
                remaining -= 1
                
            current_row += 1
    
    # Ensure we don't exceed the requested number of agents
    positions = positions[:num_agents]
    
    return positions

# Function to generate a triangle formation
def generate_triangle_shape(num_agents):
    """
    Create a triangle where:
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
        start_col = (GRID_SIZE - num_in_row) // 2
        
        # Add agents for this row
        for col in range(num_in_row):
            positions.append((row, start_col + col))
    
    return positions

# Function to determine square dimensions based on number of agents
def calculate_square_dimensions(num_agents):
    # Find the closest square root
    side_length = int(math.sqrt(num_agents))
    
    # If num_agents is not a perfect square, adjust
    if side_length * side_length < num_agents:
        width = side_length
        height = math.ceil(num_agents / width)
    else:
        width = height = side_length
        
    return width, height

# Function to generate a square formation
def generate_square_shape(num_agents):
    width, height = calculate_square_dimensions(num_agents)
    
    # Center the square in the grid
    start_row = (GRID_SIZE - height) // 2
    start_col = (GRID_SIZE - width) // 2
    
    positions = []
    for i in range(min(height * width, num_agents)):
        row = start_row + (i // width)
        col = start_col + (i % width)
        positions.append((row, col))
    
    return positions

# Function to generate a heart shape
def generate_heart_shape(num_agents):
    # Define a basic heart shape pattern (use scaling based on number of agents)
    heart_positions = [
        (2, 3), (2, 6),
        (3, 2), (3, 4), (3, 5), (3, 7),
        (4, 2), (4, 7),
        (5, 3), (5, 6),
        (6, 4), (6, 5),
        (7, 5)
    ]
    
    # If we need more or fewer agents, scale the heart
    if num_agents < len(heart_positions):
        # Use only the first n positions
        return heart_positions[:num_agents]
    elif num_agents > len(heart_positions):
        # For simplicity, add extra positions around the heart
        extra_needed = num_agents - len(heart_positions)
        extra_positions = []
        
        # Add positions around the existing heart shape
        # This is a simple approach - could be improved for better distribution
        for row in range(2, 8):
            for col in range(2, 8):
                if (row, col) not in heart_positions:
                    extra_positions.append((row, col))
                    if len(extra_positions) >= extra_needed:
                        break
            if len(extra_positions) >= extra_needed:
                break
        
        return heart_positions + extra_positions[:extra_needed]
    
    return heart_positions

# Helper function to determine if an agent is blocked
def is_agent_blocked(agent_idx, agents, grid, moves):
    x, y = agents[agent_idx]
    
    # Check if the agent can move in any direction
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] == 0:
            return False  # Agent can move in at least one direction
    
    return True  # Agent is blocked in all directions

# Determine which agents are free (not blocked)
def prioritize_agents(agents, grid, agent_moved, moves):
    free_agents = []
    blocked_agents = []
    
    # Check each agent
    for i in range(len(agents)):
        if not agent_moved[i]:
            if is_agent_blocked(i, agents, grid, moves):
                blocked_agents.append(i)
            else:
                free_agents.append(i)
    
    # Return free agents first, then blocked ones
    return free_agents + blocked_agents

# A* implementation for pathfinding with occupancy grid
def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(grid, start, goal, moves):
    # Priority queue for A* algorithm
    open_set = []
    # Entry format: (f_score, count, position, path)
    count = 0
    heapq.heappush(open_set, (heuristic(start, goal), count, start, [start]))
    
    # Keep track of visited nodes
    closed_set = set()
    
    # g_score[n] = cost from start to n
    g_score = {start: 0}
    
    while open_set:
        # Get node with lowest f_score
        f, _, current, path = heapq.heappop(open_set)
        
        # If we reached the goal, return the path
        if current == goal:
            return path
        
        # Skip if we've already processed this node
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        # Check all neighbors
        for dx, dy in moves:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            
            # Skip invalid or occupied cells
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE) or grid[nx, ny] == 1:
                continue
            
            # Calculate new path cost
            tentative_g_score = g_score[current] + 1
            
            # Check if this path is better than any previous one
            if neighbor in g_score and tentative_g_score >= g_score[neighbor]:
                continue
            
            # This path is better, record it
            g_score[neighbor] = tentative_g_score
            f_score = tentative_g_score + heuristic(neighbor, goal)
            
            count += 1
            heapq.heappush(open_set, (f_score, count, neighbor, path + [neighbor]))
    
    return []  # No path found

# Breadth-First Search implementation
def breadth_first_search(grid, start, goal, moves):
    # Initialize queue with starting position and path
    queue = [(start, [start])]
    # Keep track of visited nodes
    visited = {start}
    
    while queue:
        # Dequeue a node and its path
        current, path = queue.pop(0)
        
        # If goal reached, return the path
        if current == goal:
            return path
        
        # Explore neighbors
        for dx, dy in moves:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            
            # Skip invalid or occupied cells
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE) or grid[nx, ny] == 1:
                continue
                
            # If not visited yet, add to queue
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return []  # No path found

# Greedy Best-First Search implementation
def greedy_search(grid, start, goal, moves):
    # Priority queue for greedy algorithm
    open_set = []
    # Entry format: (heuristic, count, position, path)
    count = 0
    heapq.heappush(open_set, (heuristic(start, goal), count, start, [start]))
    
    # Keep track of visited nodes
    visited = {start}
    
    while open_set:
        # Get node with lowest heuristic
        _, _, current, path = heapq.heappop(open_set)
        
        # If goal reached, return the path
        if current == goal:
            return path
        
        # Explore neighbors
        for dx, dy in moves:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            
            # Skip invalid or occupied cells
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE) or grid[nx, ny] == 1:
                continue
                
            # If not visited yet, add to queue
            if neighbor not in visited:
                visited.add(neighbor)
                count += 1
                heapq.heappush(open_set, (heuristic(neighbor, goal), count, neighbor, path + [neighbor]))
    
    return []  # No path found

def find_path_with_algorithm(grid, start, goal, algorithm, moves):
    if algorithm == "astar":
        return a_star_search(grid, start, goal, moves)
    elif algorithm == "bfs":
        return breadth_first_search(grid, start, goal, moves)
    elif algorithm == "greedy":
        return greedy_search(grid, start, goal, moves)
    else:
        # Default to A*
        return a_star_search(grid, start, goal, moves)

# Function to initialize or reset agents
def initialize_agents(num_agents, shape):
    # Initialize agents at the bottom row and above if needed
    agents = []
    bottom_row_count = min(GRID_SIZE, num_agents)
    for i in range(bottom_row_count):
        agents.append((GRID_SIZE - 1, i))  # Bottom row

    # Add more agents above as needed
    remaining = max(0, num_agents - bottom_row_count)
    row = GRID_SIZE - 2  # Start one row above the bottom
    while remaining > 0 and row >= 0:  # Check row bounds
        cols_to_fill = min(remaining, GRID_SIZE)
        for i in range(cols_to_fill):
            agents.append((row, i))
            remaining -= 1
        row -= 1  # Move up one row
    
    # Generate target positions based on selected shape
    if shape == "circle":
        target_positions = generate_circle_shape(num_agents)
    elif shape == "triangle":
        target_positions = generate_triangle_shape(num_agents)
    elif shape == "heart":
        target_positions = generate_heart_shape(num_agents)
    else:  # "square" or default
        target_positions = generate_square_shape(num_agents)
    
    # Create occupancy grid for current state
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for pos in agents:
        x, y = pos
        grid[x, y] = 1
    
    # Reset movement tracking
    agent_moved = [False] * num_agents
    
    return agents, target_positions, grid, agent_moved

# Route to serve the frontend
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# API to initialize the simulation
@app.route('/api/initialize', methods=['POST'])
def initialize_simulation():
    data = request.json
    num_agents = data.get('agents', 8)
    shape = data.get('shape', 'square')
    algorithm = data.get('algorithm', 'astar')
    topology = data.get('topology', 'vonNeumann')
    movement = data.get('movement', 'sequential')
    collision_enabled = data.get('collision_enabled', True)
    
    # Initialize agents, target positions, grid, and movement tracking
    agents, target_positions, grid, agent_moved = initialize_agents(num_agents, shape)
    
    # Update simulation state
    simulation_state.update({
        "agents": agents,
        "target_positions": target_positions,
        "grid": grid.tolist(),  # Convert numpy array to list for JSON serialization
        "agent_moved": agent_moved,
        "current_agent_idx": None,
        "path": None,
        "movement_step": 0,
        "algorithm": algorithm,
        "topology": topology,
        "movement": movement,
        "collision_enabled": collision_enabled,
        "shape": shape
    })
    
    # Return initial state
    return jsonify({
        "status": "success",
        "agents": agents,
        "target_positions": target_positions,
        "grid": simulation_state["grid"],
        "agent_moved": agent_moved,
        "shape": shape,
        "num_agents": num_agents
    })

# API to get the current simulation state
@app.route('/api/state', methods=['GET'])
def get_simulation_state():
    return jsonify({
        "status": "success",
        "agents": simulation_state["agents"],
        "target_positions": simulation_state["target_positions"],
        "grid": simulation_state["grid"],
        "agent_moved": simulation_state["agent_moved"],
        "current_agent_idx": simulation_state["current_agent_idx"],
        "shape": simulation_state["shape"]
    })

# API to update settings
@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.json
    
    # Update simulation settings
    if 'algorithm' in data:
        simulation_state["algorithm"] = data['algorithm']
    if 'topology' in data:
        simulation_state["topology"] = data['topology']
    if 'movement' in data:
        simulation_state["movement"] = data['movement']
    if 'collision_enabled' in data:
        simulation_state["collision_enabled"] = data['collision_enabled']
    
    return jsonify({
        "status": "success",
        "settings": {
            "algorithm": simulation_state["algorithm"],
            "topology": simulation_state["topology"],
            "movement": simulation_state["movement"],
            "collision_enabled": simulation_state["collision_enabled"]
        }
    })

# API to select an agent to move
@app.route('/api/select_agent', methods=['GET'])
def select_agent():
    # Get available agents with priority (unblocked first)
    moves = MOVEMENT_DIRECTIONS["moore" if simulation_state["topology"] == "moore" else "vonNeumann"]
    
    available_agents = prioritize_agents(
        simulation_state["agents"],
        np.array(simulation_state["grid"]), 
        simulation_state["agent_moved"],
        moves
    )
    
    if not available_agents:
        # All agents have reached their targets
        return jsonify({
            "status": "complete",
            "message": "All agents have reached their target positions"
        })
    
    # Select the first available agent
    selected_agent = available_agents[0]
    simulation_state["current_agent_idx"] = selected_agent
    
    # Check if the agent is blocked
    is_blocked = is_agent_blocked(
        selected_agent, 
        simulation_state["agents"], 
        np.array(simulation_state["grid"]),
        moves
    )
    
    return jsonify({
        "status": "success",
        "agent_idx": selected_agent,
        "is_blocked": is_blocked
    })

# API to calculate path for the current agent
@app.route('/api/calculate_path', methods=['GET'])
def calculate_path():
    agent_idx = simulation_state["current_agent_idx"]
    
    if agent_idx is None:
        return jsonify({
            "status": "error",
            "message": "No agent selected"
        })
    
    # Create a temporary grid showing current occupancy (excluding the agent to move)
    temp_grid = np.zeros((GRID_SIZE, GRID_SIZE))
    
    # Mark all agent positions except the selected one
    for i, pos in enumerate(simulation_state["agents"]):
        if i != agent_idx:
            x, y = pos
            temp_grid[x, y] = 1
    
    # Get appropriate movement directions based on topology
    moves = MOVEMENT_DIRECTIONS["moore" if simulation_state["topology"] == "moore" else "vonNeumann"]
    
    # Find path using selected algorithm
    start_pos = simulation_state["agents"][agent_idx]
    target_pos = simulation_state["target_positions"][agent_idx]
    
    path = find_path_with_algorithm(
        temp_grid, 
        start_pos, 
        target_pos, 
        simulation_state["algorithm"],
        moves
    )
    
    # If agent is blocked and no path found, try moving it out of the way first
    if not path or len(path) <= 1:
        # Find any possible move to free up the agent
        x, y = start_pos
        escape_path = []
        
        # Try each direction
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and temp_grid[nx, ny] == 0:
                # Found a free cell to move to
                escape_path = [start_pos, (nx, ny)]
                break
        
        # If we found an escape path, use it
        if escape_path and len(escape_path) > 1:
            path = escape_path
        else:
            # No escape path found, we'll need to wait
            path = [start_pos]
    
    # Store the path in simulation state
    simulation_state["path"] = path
    simulation_state["movement_step"] = 0
    
    # Check if the agent is at target position
    at_target = start_pos == target_pos
    
    return jsonify({
        "status": "success",
        "agent_idx": agent_idx,
        "path": path,
        "path_length": len(path),
        "at_target": at_target
    })
@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {e}")
    return jsonify({"status": "error", "message": str(e)}), 500@app.route('/api/transform', methods=['POST'])

@app.route('/api/transform', methods=['POST'])
def transform():
    data = request.json  # Get JSON data from request
    
    # Extract key parameters
    algorithm = data.get('algorithm', 'astar')
    shape = data.get('shape', 'square')
    num_elements = data.get('num_elements', 16)
    topology = data.get('topology', 'vonNeumann')
    movement = data.get('movement', 'sequential')
    control_mode = data.get('control_mode', 'centralized')
    collision = data.get('collision', True)
    
    start_time = time.time()
    
    # Initialize agents with the target shape
    agents, target_positions, grid, agent_moved = initialize_agents(num_elements, shape)
    
    # Convert grid to numpy array
    grid_array = np.array(grid)
    
    # Determine movement directions based on topology
    moves = MOVEMENT_DIRECTIONS["moore" if topology == "moore" else "vonNeumann"]
    
    # Store all moves for the transformation
    transformation_moves = []
    nodes_explored = 0
    
    # Simulation-wide tracking
    current_agents = agents.copy()
    current_grid = grid_array.copy()
    
    # Track which agents have reached their target
    completed_agents = [False] * num_elements
    
    # Sequential or parallel movement based on selected mode
    if movement == 'sequential':
        # Process each agent sequentially
        for agent_idx in range(num_elements):
            start_pos = current_agents[agent_idx]
            target_pos = target_positions[agent_idx]
            
            # Create a temporary grid excluding the current agent
            temp_grid = current_grid.copy()
            temp_grid[start_pos[0], start_pos[1]] = 0
            
            # Find path for this agent
            path = find_path_with_algorithm(
                temp_grid, 
                start_pos, 
                target_pos, 
                algorithm,
                moves
            )
            
            # Process the path
            for step_idx in range(1, len(path)):
                prev_pos = path[step_idx - 1]
                curr_pos = path[step_idx]
                
                # Record the move
                transformation_moves.append({
                    "agentId": agent_idx,
                    "from": {"x": prev_pos[0], "y": prev_pos[1]},
                    "to": {"x": curr_pos[0], "y": curr_pos[1]}
                })
                
                # Update grid and agent position
                current_grid[prev_pos[0], prev_pos[1]] = 0
                current_grid[curr_pos[0], curr_pos[1]] = 1
                current_agents[agent_idx] = curr_pos
    
    elif movement == 'parallel':
        # More complex parallel movement logic would go here
        # For now, we'll do a simplified parallel approach
        remaining_agents = list(range(num_elements))
        
        while remaining_agents:
            # Tracks moves in this iteration
            iteration_moves = []
            
            # Try to move each remaining agent
            for agent_idx in remaining_agents.copy():
                start_pos = current_agents[agent_idx]
                target_pos = target_positions[agent_idx]
                
                # Create a temporary grid excluding the current agent
                temp_grid = current_grid.copy()
                temp_grid[start_pos[0], start_pos[1]] = 0
                
                # Find path for this agent
                path = find_path_with_algorithm(
                    temp_grid, 
                    start_pos, 
                    target_pos, 
                    algorithm,
                    moves
                )
                
                # If path exists and has more than one step
                if len(path) > 1:
                    next_pos = path[1]
                    
                    # Check if the next position is free
                    if current_grid[next_pos[0], next_pos[1]] == 0:
                        # Record the move
                        iteration_moves.append({
                            "agentId": agent_idx,
                            "from": {"x": start_pos[0], "y": start_pos[1]},
                            "to": {"x": next_pos[0], "y": next_pos[1]}
                        })
                        
                        # Update grid and agent position
                        current_grid[start_pos[0], start_pos[1]] = 0
                        current_grid[next_pos[0], next_pos[1]] = 1
                        current_agents[agent_idx] = next_pos
                        
                        # Check if agent reached target
                        if next_pos == target_pos:
                            remaining_agents.remove(agent_idx)
                    
                # If no move possible or already at target
                elif start_pos == target_pos:
                    remaining_agents.remove(agent_idx)
            
            # Add moves from this iteration
            transformation_moves.extend(iteration_moves)
            
            # Prevent infinite loop
            if not iteration_moves:
                break
    
    # Calculate transformation metrics
    end_time = time.time()
    transformation_time = end_time - start_time
    
    # Prepare response
    response_data = {
        "status": "success",
        "message": "Transformation complete",
        "moves": transformation_moves,
        "time": transformation_time,
        "nodes": len(transformation_moves) * 2,  # Approximation
        "algorithm": algorithm,
        "shape": shape,
        "topology": topology,
        "movement": movement,
        "control_mode": control_mode,
        "collision": collision
    }
    
    return jsonify(response_data)
# API to perform the next movement step
@app.route('/api/move_step', methods=['GET'])
def move_step():
    if simulation_state["current_agent_idx"] is None or simulation_state["path"] is None:
        return jsonify({
            "status": "error",
            "message": "No agent or path selected"
        })
    
    agent_idx = simulation_state["current_agent_idx"]
    path = simulation_state["path"]
    step_idx = simulation_state["movement_step"] + 1
    
    # If the path is not valid or complete
    if not path or len(path) <= 1 or step_idx >= len(path):
        # Check if the agent reached its intended target position
        if path and path[-1] == simulation_state["target_positions"][agent_idx]:
            # Agent has reached its final destination
            simulation_state["agent_moved"][agent_idx] = True
            status_msg = f"Agent {agent_idx} has reached its target position!"
            reached_target = True
        else:
            # Agent has only moved to an intermediate position
            status_msg = f"Agent {agent_idx} has moved to an intermediate position."
            reached_target = False
        
        # Clear the movement state
        simulation_state["current_agent_idx"] = None
        simulation_state["path"] = None
        simulation_state["movement_step"] = 0
        
        return jsonify({
            "status": "step_complete",
            "message": status_msg,
            "agent_idx": agent_idx,
            "reached_target": reached_target,
            "agent_moved": simulation_state["agent_moved"],
            "agents": simulation_state["agents"]
        })
    
    # Process the movement step
    # Remove agent from old position
    old_x, old_y = simulation_state["agents"][agent_idx]
    grid = np.array(simulation_state["grid"])
    grid[old_x, old_y] = 0
    
    # Update position of current agent
    simulation_state["agents"][agent_idx] = path[step_idx]
    
    # Mark new position as occupied
    new_x, new_y = path[step_idx]
    grid[new_x, new_y] = 1
    
    # Update grid in simulation state
    simulation_state["grid"] = grid.tolist()
    
    # Update step counter
    simulation_state["movement_step"] = step_idx
    
    # Check if we're on the last step
    is_last_step = step_idx == len(path) - 1
    
    return jsonify({
        "status": "success",
        "agent_idx": agent_idx,
        "step": step_idx,
        "total_steps": len(path) - 1,
        "agent_position": path[step_idx],
        "agents": simulation_state["agents"],
        "grid": simulation_state["grid"],
        "is_last_step": is_last_step
    })

# API to check if the simulation is complete
@app.route('/api/check_completion', methods=['GET'])
def check_completion():
    all_moved = all(simulation_state["agent_moved"])
    moved_count = sum(simulation_state["agent_moved"])
    total_agents = len(simulation_state["agents"])
    
    return jsonify({
        "status": "success",
        "complete": all_moved,
        "moved_count": moved_count,
        "total_agents": total_agents,
        "completion_percentage": int((moved_count / total_agents) * 100) if total_agents > 0 else 0
    })
@app.before_request
def log_request():
    print(f"Received {request.method} request for {request.path}")
# Serve static files (CSS, JS, images, etc.)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Create static directory if it doesn't exist (keeping this is fine)
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Remove all these file copying operations
    # with open('static/index.html', 'w') as f:
    #     f.write(open('index.html', 'r').read())
    
    # with open('static/styles.css', 'w') as f:
    #     f.write(open('styles.css', 'r').read())
    
    # with open('static/script.js', 'w') as f:
    #     f.write(open('script.js', 'r').read())
    
    app.run(debug=True)
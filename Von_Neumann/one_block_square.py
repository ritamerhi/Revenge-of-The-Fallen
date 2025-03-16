import numpy as np
import streamlit as st
import time
import matplotlib.pyplot as plt
import math
import heapq  # For priority queue in A*

# Fixed grid size of 10
GRID_SIZE = 10

# User input for number of agents
NUM_AGENTS = st.number_input("Number of agents", min_value=1, value=16, step=1)

# Validate number of agents against grid size
max_agents = GRID_SIZE * GRID_SIZE
if NUM_AGENTS > max_agents:
    st.error(f"Grid size of {GRID_SIZE}x{GRID_SIZE} can only accommodate up to {max_agents} agents.")
    st.warning(f"Setting number of agents to {max_agents}.")
    NUM_AGENTS = max_agents

# Function to determine square dimensions based on number of agents
def calculate_square_dimensions(num_agents):
    # Find the closest square root
    side_length = int(math.sqrt(num_agents))
    
    # If num_agents is not a perfect square, adjust
    if side_length * side_length < num_agents:
        # Determine if we should make a rectangle or an incomplete square
        width = side_length
        height = math.ceil(num_agents / width)
    else:
        width = height = side_length
        
    return width, height

# Calculate square dimensions
width, height = calculate_square_dimensions(NUM_AGENTS)

# Validate grid size for square formation
min_grid_size = max(width, height)
if GRID_SIZE < min_grid_size:
    st.error(f"Grid size must be at least {min_grid_size} to accommodate the square formation.")
    st.stop()

# Define movement directions (Von Neumann topology)
MOVES = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

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

# Function to display the grid
def display_grid(agent_positions, moving_agent_idx=None, target_positions=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Draw target positions
    if target_positions:
        for i, (tx, ty) in enumerate(target_positions):
            ax.add_patch(plt.Rectangle((ty, GRID_SIZE - tx - 1), 1, 1, color='lightblue', alpha=0.5))
            ax.text(ty + 0.5, GRID_SIZE - tx - 0.5, str(i), ha='center', va='center', color='blue')
    
    # Draw agents
    for i, (x, y) in enumerate(agent_positions):
        if i < len(agent_positions):  # Only draw valid agents
            # Determine color: green for moving, blue for reached target, red for others
            if i == moving_agent_idx:
                color = 'green'
            elif st.session_state.get('agent_moved', [False] * len(agent_positions))[i]:
                color = 'blue'
            else:
                color = 'red'
                
            ax.add_patch(plt.Rectangle((y, GRID_SIZE - x - 1), 1, 1, color=color))
            ax.text(y + 0.5, GRID_SIZE - x - 0.5, str(i), ha='center', va='center', color='white')
    
    return fig

# Helper function to determine if an agent is blocked
def is_agent_blocked(agent_idx, agents, grid):
    x, y = agents[agent_idx]
    
    # Check if the agent can move in any direction
    for dx, dy in MOVES:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] == 0:
            return False  # Agent can move in at least one direction
    
    return True  # Agent is blocked in all directions

# Determine which agents are free (not blocked)
def prioritize_agents(agents, grid, agent_moved):
    free_agents = []
    blocked_agents = []
    
    # Create a copy of the grid for checking
    temp_grid = grid.copy()
    
    # Check each agent
    for i in range(len(agents)):
        if not agent_moved[i]:
            if is_agent_blocked(i, agents, temp_grid):
                blocked_agents.append(i)
            else:
                free_agents.append(i)
    
    # Return free agents first, then blocked ones
    return free_agents + blocked_agents

# A* implementation for pathfinding with occupancy grid
def heuristic(a, b):
    # Manhattan distance for Von Neumann topology
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(grid, start, goal):
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
        
        # Check all neighbors (Von Neumann topology: up, down, left, right)
        for dx, dy in MOVES:
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

# Create containers for UI elements (only once)
if 'ui_containers' not in st.session_state:
    st.session_state.ui_containers = {
        'title': st.empty(),
        'frame': st.empty(),
        'status': st.empty(),
        'progress': st.empty(),
        'message': st.empty(),
        'square_info': st.empty(),
        'target_viz': st.empty(),
        'stats': st.empty(),
    }

# Use the containers to update content
containers = st.session_state.ui_containers

# Set the title once
containers['title'].title("Von Neumann Method - Square Formation")

# Function to initialize or reset agents
def initialize_agents(num_agents):
    # Initialize agents at the bottom row and above if needed
    agents = []
    bottom_row_count = min(10, num_agents)
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
    
    # Generate target positions
    target_positions = generate_square_shape(num_agents)
    
    # Create occupancy grid for current state
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for pos in agents:
        x, y = pos
        grid[x, y] = 1
    
    # Reset movement tracking
    agent_moved = [False] * num_agents
    
    return agents, target_positions, grid, agent_moved

# Set agents button
if st.button("Set Agents"):
    # Initialize or reset the agents
    agents, target_positions, grid, agent_moved = initialize_agents(NUM_AGENTS)
    
    # Update session state
    st.session_state.agents = agents
    st.session_state.target_positions = target_positions
    st.session_state.grid = grid
    st.session_state.agent_moved = agent_moved
    st.session_state.current_agent_count = NUM_AGENTS
    st.session_state.initialized = True
    
    # Force recreation of visualization
    if 'target_viz_created' in st.session_state:
        del st.session_state.target_viz_created
    
    # Clear any in-progress movement
    if 'current_moving_agent' in st.session_state:
        del st.session_state.current_moving_agent
    
    st.rerun()

# Initialize grid and state variables if not already done
if 'initialized' not in st.session_state:
    # Initialize with the current number of agents
    agents, target_positions, grid, agent_moved = initialize_agents(NUM_AGENTS)
    
    # Store in session state
    st.session_state.initialized = True
    st.session_state.agents = agents
    st.session_state.target_positions = target_positions
    st.session_state.grid = grid
    st.session_state.agent_moved = agent_moved
    st.session_state.current_agent_count = NUM_AGENTS

# Create target visualization once
if 'target_viz_created' not in st.session_state:
    target_fig, target_ax = plt.subplots(figsize=(4, 4))
    target_ax.set_xlim(0, GRID_SIZE)
    target_ax.set_ylim(0, GRID_SIZE)
    target_ax.axis('off')
    
    for i, (tx, ty) in enumerate(st.session_state.target_positions):
        target_ax.add_patch(plt.Rectangle((ty, GRID_SIZE - tx - 1), 1, 1, color='blue'))
        target_ax.text(ty + 0.5, GRID_SIZE - tx - 0.5, str(i), ha='center', va='center', color='white')
    
    containers['target_viz'].write("### Target Square Formation")
    containers['target_viz'].pyplot(target_fig)
    plt.close(target_fig)
    
    # Show square info
    current_width, current_height = calculate_square_dimensions(st.session_state.current_agent_count)
    containers['square_info'].write("### Square Structure:")
    containers['square_info'].write(f"Width: {current_width}, Height: {current_height}")
    containers['square_info'].write(f"Total blocks: {st.session_state.current_agent_count}")
    
    st.session_state.target_viz_created = True

# Initial display of the grid
fig = display_grid(st.session_state.agents, None, st.session_state.target_positions)
containers['frame'].pyplot(fig)
plt.close(fig)

# Display stats
moved_count = sum(st.session_state.get('agent_moved', [False] * st.session_state.current_agent_count))
containers['stats'].write(f"Progress: {moved_count}/{st.session_state.current_agent_count} agents have reached their targets.")

# Automated agent movement - prioritize unblocked agents
agent_moved = st.session_state.get('agent_moved', [False] * st.session_state.current_agent_count)

# Get available agents with priority (unblocked first)
available_agents = prioritize_agents(st.session_state.agents, st.session_state.grid, agent_moved)

if not available_agents:
    containers['status'].write("### All agents have reached their target positions")
    if containers['message'].button("Reset Simulation"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
else:
    # Automatically select the first available agent (prioritized)
    selected_agent = available_agents[0]
    
    # Display the agent status (blocked or free)
    is_blocked = is_agent_blocked(selected_agent, st.session_state.agents, st.session_state.grid)
    status = "BLOCKED" if is_blocked else "FREE"
    containers['status'].write(f"### Currently moving: Agent {selected_agent} ({status})")
    
    # Update display graph with current state
    fig = display_grid(st.session_state.agents, None, st.session_state.target_positions)
    containers['frame'].pyplot(fig)
    plt.close(fig)
    
    # Automatically start movement if no agent is currently moving
    if 'current_moving_agent' not in st.session_state or st.session_state.current_moving_agent is None:
        # Initiate movement of the selected agent
        st.session_state.current_moving_agent = selected_agent
        st.session_state.movement_step = 0
        st.session_state.path = None
        
        # Moved count
        moved_count = sum(st.session_state.get('agent_moved', [False] * st.session_state.current_agent_count))
        containers['stats'].write(f"Progress: {moved_count}/{st.session_state.current_agent_count} agents have reached their targets.")
        
        st.rerun()

# Process agent movement - with special handling for blocked agents
if 'current_moving_agent' in st.session_state and st.session_state.current_moving_agent is not None:
    agent_idx = st.session_state.current_moving_agent
    
    # If we haven't calculated the path yet, do it now
    if 'path' not in st.session_state or st.session_state.path is None:
        # Create a temporary grid showing current occupancy (excluding the agent to move)
        temp_grid = np.zeros((GRID_SIZE, GRID_SIZE))
        
        # Mark all agent positions except the selected one
        for i, pos in enumerate(st.session_state.agents):
            if i != agent_idx:
                x, y = pos
                temp_grid[x, y] = 1
                
        # Find path using A*
        start_pos = st.session_state.agents[agent_idx]
        target_pos = st.session_state.target_positions[agent_idx]
        
        # Find path
        path = a_star_search(temp_grid, start_pos, target_pos)
        
        # If agent is blocked and no path found, try moving it out of the way first
        if not path or len(path) <= 1:
            # Find any possible move to free up the agent
            x, y = start_pos
            escape_path = []
            
            # Try each direction
            for dx, dy in MOVES:
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
                
        st.session_state.path = path
        st.session_state.movement_step = 0
    
    # If we have a valid path
    if st.session_state.path and len(st.session_state.path) > 1:
        path = st.session_state.path
        step_idx = st.session_state.movement_step + 1
        
        if step_idx < len(path):
            # Progress indicators
            containers['progress'].progress(step_idx / (len(path) - 1))
            containers['message'].write(f"Moving Agent {agent_idx} - Step {step_idx}/{len(path)-1}")
            
            # Remove agent from old position
            old_x, old_y = st.session_state.agents[agent_idx]
            st.session_state.grid[old_x, old_y] = 0
            
            # Update position of current agent
            st.session_state.agents[agent_idx] = path[step_idx]
            
            # Mark new position as occupied
            new_x, new_y = path[step_idx]
            st.session_state.grid[new_x, new_y] = 1
            
            # Display the grid with updated positions
            fig = display_grid(st.session_state.agents, agent_idx, st.session_state.target_positions)
            containers['frame'].pyplot(fig)
            plt.close(fig)
            
            # Update step counter
            st.session_state.movement_step = step_idx
            
            # Schedule the next step
            time.sleep(0.3)
            st.rerun()
        else:
            # Check if the agent reached its intended target position
            if path[-1] == st.session_state.target_positions[agent_idx]:
                # Agent has reached its final destination
                agent_moved = st.session_state.get('agent_moved', [False] * st.session_state.current_agent_count)
                agent_moved[agent_idx] = True
                st.session_state.agent_moved = agent_moved
                
                status_msg = f"Agent {agent_idx} has reached its target position!"
                containers['message'].success(status_msg)
            else:
                # Agent has only moved to an intermediate position
                # It will be picked up again in the next round
                status_msg = f"Agent {agent_idx} has moved to an intermediate position."
                containers['message'].info(status_msg)
            
            # Update the display
            fig = display_grid(st.session_state.agents, None, st.session_state.target_positions)
            containers['frame'].pyplot(fig)
            plt.close(fig)
            
            # Clear the movement state
            st.session_state.current_moving_agent = None
            st.session_state.path = None
            st.session_state.movement_step = 0
            
            # Clear progress bar
            containers['progress'].empty()
            
            time.sleep(1)
            st.rerun()
    else:
        # No valid path found and no escape route
        agent_moved = st.session_state.get('agent_moved', [False] * st.session_state.current_agent_count)
        
        if is_agent_blocked(agent_idx, st.session_state.agents, st.session_state.grid):
            # Skip the agent for now if it's completely blocked
            containers['message'].warning(f"Agent {agent_idx} is completely blocked. Will try again later.")
        else:
            # If agent is at its target position already, mark it as moved
            current_pos = st.session_state.agents[agent_idx]
            target_pos = st.session_state.target_positions[agent_idx]
            
            if current_pos == target_pos:
                agent_moved[agent_idx] = True
                st.session_state.agent_moved = agent_moved
                containers['message'].success(f"Agent {agent_idx} is already at its target position!")
            else:
                containers['message'].error(f"No valid path found for Agent {agent_idx}!")
        
        # Update display
        fig = display_grid(st.session_state.agents, None, st.session_state.target_positions)
        containers['frame'].pyplot(fig)
        plt.close(fig)
        
        st.session_state.current_moving_agent = None
        time.sleep(1)
        st.rerun()
import numpy as np
import streamlit as st
import time
import matplotlib.pyplot as plt
import math
import heapq  # For priority queue in A*
import random  # For randomizing movements to break deadlocks

# Fixed grid size of 10
GRID_SIZE = 10

# User input for number of agents
NUM_AGENTS = st.number_input("Number of agents", min_value=1, value=36, step=1)

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

# Define movement directions (Moore topology - 8 directional)
MOVES = [(0, 1),(1, 0),(0, -1),(-1, 0),(1, 1),(-1,1),(1, -1), (-1, -1)]  # Diagonal

# Function to generate a square formation
def generate_square_shape(num_agents):
    width, height = calculate_square_dimensions(num_agents)
    
    # Center the square in the grid
    start_row = (GRID_SIZE - height) // 2
    start_col = (GRID_SIZE - width) // 2
    
    positions = []
    
    # NEW: Track positions of certain "problematic" agents (5 and 9)
    problematic_indices = [5, 9] if num_agents > 9 else [5] if num_agents > 5 else []
    problematic_positions = []
    
    # First, generate all positions
    for i in range(min(height * width, num_agents)):
        row = start_row + (i // width)
        col = start_col + (i % width)
        
        # Store the position
        if i in problematic_indices:
            problematic_positions.append((row, col))
        else:
            positions.append((row, col))
    
    # Now add the problematic positions to the beginning of the list
    # This ensures they get more favorable positions in the square
    # (typically corners or edges which are easier to reach)
    positions = problematic_positions + positions
    
    return positions

# Function to display the grid
def display_grid(agent_positions, moving_agent_indices=None, target_positions=None):
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
            # Determine color: 
            # - green for moving
            # - blue for reached target
            # - orange for severely stuck agents
            # - red for others
            
            # Check if this agent is severely stuck
            is_severely_stuck = False
            if hasattr(st.session_state, 'agent_stuck_counters'):
                if i < len(st.session_state.agent_stuck_counters) and st.session_state.agent_stuck_counters[i] > 5:
                    is_severely_stuck = True
            
            if moving_agent_indices and i in moving_agent_indices:
                color = 'green'
            elif st.session_state.get('agent_moved', [False] * len(agent_positions))[i]:
                color = 'blue'
            elif is_severely_stuck:
                color = 'orange'  # Highlight severely stuck agents
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

# A* implementation for pathfinding with occupancy grid
def heuristic(a, b):
    # Mix of Manhattan and Euclidean for better diagonal movement
    return (abs(a[0] - b[0]) + abs(a[1] - b[1]) + 
           0.5 * math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2))


def a_star_search(grid, start, goal):
    """
    Improved A* search with more direct path prioritization
    """
    open_set = []
    count = 0
    heapq.heappush(open_set, (heuristic(start, goal), count, start, [start]))
    
    closed_set = set()
    g_score = {start: 0}
    
    # New parameter to prioritize more direct paths
    direct_path_bias = 0.5
    
    while open_set:
        f, _, current, path = heapq.heappop(open_set)
        
        if current == goal:
            return path
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        for dx, dy in MOVES:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            
            # Skip invalid or occupied cells
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE) or grid[nx, ny] == 1:
                continue
            
            # Calculate move cost with bias for directness
            move_cost = 1.0 if dx == 0 or dy == 0 else 1.414
            tentative_g_score = g_score[current] + move_cost
            
            # Bias towards more direct paths
            direction_bias = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
            f_score = (tentative_g_score + 
                       heuristic(neighbor, goal) * (1 + direct_path_bias) - 
                       direction_bias * direct_path_bias)
            
            if neighbor in g_score and tentative_g_score >= g_score[neighbor]:
                continue
            
            g_score[neighbor] = tentative_g_score
            
            count += 1
            heapq.heappush(open_set, (f_score, count, neighbor, path + [neighbor]))
    
    return []  # No path found


# Updated Intermediate Goal Finding Function

def find_intermediate_goal(grid, start, goal, max_depth=5):
    """
    Find an intermediate goal with recursion protection
    
    Args:
    - grid: Occupancy grid
    - start: Starting position
    - goal: Target position
    - max_depth: Maximum recursion depth to prevent stack overflow
    """
    # Base case to prevent infinite recursion
    if max_depth <= 0:
        return []
    
    # If we can't find a direct path to goal, find an intermediate location
    # Create a list of possible intermediate positions
    intermediate_positions = []
    
    # Try to find empty spaces in concentric circles around the goal
    max_distance = max(GRID_SIZE, GRID_SIZE)
    for distance in range(1, max_distance):
        for dx in range(-distance, distance + 1):
            for dy in range(-distance, distance + 1):
                # Only consider positions on the perimeter of the circle
                if abs(dx) == distance or abs(dy) == distance:
                    nx, ny = goal[0] + dx, goal[1] + dy
                    
                    # Check if position is valid and empty
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] == 0:
                        # Calculate distances
                        dist_from_start = abs(nx - start[0]) + abs(ny - start[1])
                        dist_from_goal = abs(nx - goal[0]) + abs(ny - goal[1])
                        
                        # Only consider positions that are closer to the goal than the start
                        if dist_from_goal < dist_from_start:
                            intermediate_positions.append((nx, ny, dist_from_goal))
        
        # If we found some intermediate positions in this ring, stop searching
        if intermediate_positions:
            break
    
    # Sort by distance to goal
    intermediate_positions.sort(key=lambda x: x[2])
    
    # Try to find paths to the intermediate positions
    for nx, ny, _ in intermediate_positions:
        # Try A* search first
        path = a_star_search(grid, start, (nx, ny))
        
        if path and len(path) > 1:
            return path
    
    # If no path was found to any intermediate position, try any valid neighboring cell
    for dx, dy in MOVES:
        nx, ny = start[0] + dx, start[1] + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny] == 0:
            return [start, (nx, ny)]
    
    # Last resort: recursive call with reduced depth
    if max_depth > 1:
        # Modify grid slightly to break potential deadlock
        temp_grid = grid.copy()
        x, y = start
        # Try to clear a nearby cell to create a path
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                temp_grid[nx, ny] = 0
                result = find_intermediate_goal(temp_grid, start, goal, max_depth - 1)
                if result:
                    return result
    
    return []  # No path found

# Replace the existing plan_coordinated_moves function with this enhanced version

def plan_coordinated_moves(agents, target_positions, agent_moved, stuck_counter):
    """
    Enhanced version with prioritization for any stuck agents, not just a specific one.
    """
    all_paths = {}
    blocked_agents = set()
    
    # Track which agents have been stuck for a long time
    # We'll detect this based on their coordinates not changing over multiple iterations
    # This would need to be tracked in session_state, but for now let's use stuck_counter as a proxy
    priority_agents = set()
    
    # Add agent 5 and 9 as priority agents when we're having issues with them
    if 5 < len(agents) and not agent_moved[5] and stuck_counter > 3:
        priority_agents.add(5)
    if 9 < len(agents) and not agent_moved[9] and stuck_counter > 3:
        priority_agents.add(9)
    
    # First pass: Handle priority agents with special care
    for priority_agent in priority_agents:
        start = agents[priority_agent]
        goal = target_positions[priority_agent]
        
        # Create two temporary grids - one for just checking path existence
        # and another for actual path planning that might exclude some agents
        check_grid = np.zeros((GRID_SIZE, GRID_SIZE))
        planning_grid = np.zeros((GRID_SIZE, GRID_SIZE))
        
        # Fill check grid with all other agents
        for j, pos in enumerate(agents):
            if j != priority_agent:
                x, y = pos
                check_grid[x, y] = 1
        
        # First try normal A* search
        path = a_star_search(check_grid, start, goal)
        
        # If no path exists, identify potential blockers and remove them from planning grid
        if not path or len(path) <= 1:
            print(f"Priority Agent {priority_agent} has no direct path to its target.")
            
            # Find potential blockers in the path direction
            dx = goal[0] - start[0]
            dy = goal[1] - start[1]
            
            # Normalize direction
            dir_length = max(1, abs(dx) + abs(dy))
            dx_norm = dx / dir_length
            dy_norm = dy / dir_length
            
            # Look for agents in the target direction
            potential_blockers = []
            for j, pos in enumerate(agents):
                if j != priority_agent and not agent_moved[j]:
                    # Check if this agent is in the general direction of the target
                    agent_dx = pos[0] - start[0]
                    agent_dy = pos[1] - start[1]
                    
                    # Calculate dot product to see if in same general direction
                    dot_product = (dx_norm * agent_dx) + (dy_norm * agent_dy)
                    
                    # Calculate distance to determine if it's close enough to be relevant
                    dist = abs(agent_dx) + abs(agent_dy)
                    
                    if dot_product > 0 and dist <= 5:  # In target direction and close enough
                        potential_blockers.append((j, dist))
            
            # Sort blockers by distance
            potential_blockers.sort(key=lambda x: x[1])
            
            # Create a planning grid where we exclude the top 3 blockers (if any)
            for j, pos in enumerate(agents):
                if j != priority_agent:
                    is_excluded_blocker = False
                    if len(potential_blockers) > 0:
                        # Check if this agent is one of the top blockers we want to exclude
                        for blocker_idx, _ in potential_blockers[:min(3, len(potential_blockers))]:
                            if j == blocker_idx:
                                is_excluded_blocker = True
                                print(f"Temporarily ignoring agent {j} as potential blocker for Priority Agent {priority_agent}")
                                break
                    
                    if not is_excluded_blocker:
                        x, y = pos
                        planning_grid[x, y] = 1
            
            # Now try to find a path with some blockers removed
            path = a_star_search(planning_grid, start, goal)
            
            # If still no path, try intermediate goals
            if not path or len(path) <= 1:
                path = find_intermediate_goal(planning_grid, start, goal, max_depth=7)  # Higher max_depth for priority agents
        
        if path and len(path) > 1:
            print(f"Priority Agent {priority_agent} path: {path}")
            all_paths[priority_agent] = path  # Store path for priority agent
        else:
            # If still no path, add to blocked agents list but will be handled with higher priority later
            blocked_agents.add(priority_agent)
    
    # Second pass: Handle all remaining agents
    for i in range(len(agents)):
        if not agent_moved[i] and i not in priority_agents:
            # Create a temporary grid showing current occupancy (excluding this agent)
            temp_grid = np.zeros((GRID_SIZE, GRID_SIZE))
            for j, pos in enumerate(agents):
                if j != i:
                    x, y = pos
                    temp_grid[x, y] = 1
            
            # Use improved path-finding methods
            path = a_star_search(temp_grid, agents[i], target_positions[i])
            
            # If no direct path, try advanced intermediate goal
            if not path or len(path) <= 1:
                if stuck_counter > 2:
                    path = find_intermediate_goal(temp_grid, agents[i], target_positions[i])
                
                # Mark this agent as blocked if it still has no path
                if not path or len(path) <= 1:
                    blocked_agents.add(i)
            
            # Store valid paths
            if path and len(path) > 1:
                all_paths[i] = path
    
    # Step 2: Identify agents that can move simultaneously
    planned_moves = {}
    reserved_positions = set()  # Track positions that will be occupied in the next step
    
    # First, process agents that are at their targets but not marked as moved
    for i in range(len(agents)):
        if not agent_moved[i] and agents[i] == target_positions[i]:
            planned_moves[i] = [agents[i]]  # Mark as "moved" with no actual movement
    
    # Special handling for priority agents if they're blocked
    for priority_agent in priority_agents:
        if priority_agent in blocked_agents:
            # For a blocked priority agent, check EVERY possible move direction
            x, y = agents[priority_agent]
            
            # Try all possible moves, find the one that gets closest to target
            best_move = None
            best_distance = float('inf')
            
            for dx, dy in MOVES:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    # Check if position is free
                    is_occupied = False
                    for j, pos in enumerate(agents):
                        if j != priority_agent and pos == (nx, ny):
                            is_occupied = True
                            break
                    
                    if not is_occupied:
                        # Calculate distance to target
                        dist_to_target = abs(nx - target_positions[priority_agent][0]) + abs(ny - target_positions[priority_agent][1])
                        if dist_to_target < best_distance:
                            best_distance = dist_to_target
                            best_move = (nx, ny)
            
            if best_move:
                planned_moves[priority_agent] = [agents[priority_agent], best_move]
                reserved_positions.add(best_move)
                # Remove from blocked_agents since we found a move
                blocked_agents.remove(priority_agent)
    
    # Special handling for other blocked agents
    if blocked_agents and stuck_counter > 3:
        for blocked_id in blocked_agents:
            # For each blocked agent, find agents that might be in the way
            blocked_agent_pos = agents[blocked_id]
            blocked_agent_target = target_positions[blocked_id]
            
            # Calculate ideal direction vector toward target
            dx = blocked_agent_target[0] - blocked_agent_pos[0]
            dy = blocked_agent_target[1] - blocked_agent_pos[1]
            
            # Normalize direction
            dir_length = max(1, abs(dx) + abs(dy))
            dx = dx / dir_length
            dy = dy / dir_length
            
            # Look for agents that are in the general direction of the target
            potential_blockers = []
            for j, pos in enumerate(agents):
                if j != blocked_id and not agent_moved[j]:
                    # Check if this agent is in the general direction of the target
                    agent_dx = pos[0] - blocked_agent_pos[0]
                    agent_dy = pos[1] - blocked_agent_pos[1]
                    
                    # Check if in similar direction and close enough
                    dist_to_blocker = abs(agent_dx) + abs(agent_dy)
                    if dist_to_blocker <= 3:  # Within 3 cells distance
                        # Calculate dot product to see if in same general direction
                        dot_product = (dx * agent_dx) + (dy * agent_dy)
                        if dot_product > 0:  # In the same general direction
                            potential_blockers.append((j, dist_to_blocker))
            
            # Sort potential blockers by distance (closest first)
            potential_blockers.sort(key=lambda x: x[1])
            
            # Prioritize moving those potential blocking agents if they have paths
            for blocker_id, _ in potential_blockers:
                if blocker_id in all_paths and blocker_id not in planned_moves:
                    # This is a potential blocker with a valid path
                    path = all_paths[blocker_id]
                    next_pos = path[1]
                    
                    # Check if the next position is already reserved
                    if next_pos in reserved_positions:
                        continue
                    
                    # Check if the next position conflicts with another agent
                    is_conflict = False
                    for j, pos in enumerate(agents):
                        if j != blocker_id and pos == next_pos and j not in planned_moves:
                            is_conflict = True
                            break
                    
                    if not is_conflict:
                        planned_moves[blocker_id] = path[:2]  # Just the current position and next step
                        reserved_positions.add(next_pos)
    
    # Prioritize remaining priority agents' movements
    for priority_agent in priority_agents:
        if priority_agent in all_paths and priority_agent not in planned_moves:
            path = all_paths[priority_agent]
            next_pos = path[1]
            
            # Check if the position is already reserved
            if next_pos not in reserved_positions:
                # Check for conflicts
                is_conflict = False
                for j, pos in enumerate(agents):
                    if j != priority_agent and pos == next_pos and j not in planned_moves:
                        is_conflict = True
                        break
                
                # If there's a conflict, prioritize moving it by moving the conflicting agent first
                if is_conflict:
                    # Identify the conflicting agent
                    conflicting_agent = None
                    for j, pos in enumerate(agents):
                        if j != priority_agent and pos == next_pos:
                            conflicting_agent = j
                            break
                    
                    # Try to move the conflicting agent out of the way
                    if conflicting_agent is not None and not agent_moved[conflicting_agent]:
                        x, y = agents[conflicting_agent]
                        
                        # Find a free adjacent cell for the conflicting agent
                        for dx, dy in MOVES:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                                # Check if this position is free and not reserved
                                is_free = True
                                if (nx, ny) in reserved_positions:
                                    is_free = False
                                else:
                                    for k, pos in enumerate(agents):
                                        if k != conflicting_agent and pos == (nx, ny):
                                            is_free = False
                                            break
                                
                                if is_free and (nx, ny) != agents[priority_agent]:
                                    # Move the conflicting agent
                                    planned_moves[conflicting_agent] = [agents[conflicting_agent], (nx, ny)]
                                    reserved_positions.add((nx, ny))
                                    
                                    # Now priority agent can move
                                    planned_moves[priority_agent] = path[:2]
                                    reserved_positions.add(next_pos)
                                    break
                else:
                    # No conflict, priority agent can move
                    planned_moves[priority_agent] = path[:2]
                    reserved_positions.add(next_pos)
    
    # Process remaining agents with valid paths by priority
    # If we're stuck for a long time, prioritize random movement to break deadlocks
    if stuck_counter > 5:
        # Shuffle agents to break potential deadlocks
        agent_ids = list(all_paths.keys())
        random.shuffle(agent_ids)
        agents_with_paths = [(i, len(all_paths[i])) for i in agent_ids if i in all_paths]
    else:
        # Normal case: prioritize by path length
        agents_with_paths = [(i, len(path)) for i, path in all_paths.items()]
        agents_with_paths.sort(key=lambda x: x[1])  # Sort by path length
    
    for agent_id, _ in agents_with_paths:
        # Skip if already planned
        if agent_id in planned_moves:
            continue
        
        path = all_paths[agent_id]
        if len(path) > 1:
            next_pos = path[1]
            
            # Check if the next position is already reserved
            if next_pos in reserved_positions:
                continue
            
            # Check if the next position is another agent's current position
            # that isn't planning to move in this round
            is_conflict = False
            for j, pos in enumerate(agents):
                if j != agent_id and pos == next_pos and j not in planned_moves:
                    is_conflict = True
                    break
            
            if not is_conflict:
                planned_moves[agent_id] = path[:2]  # Just the current position and next step
                reserved_positions.add(next_pos)
    
    return planned_moves

# Replace or add this improved swap_candidates function
def find_swap_candidates(agents, target_positions, agent_moved, stuck_counter):
    swap_candidates = []
    
    # Identify potentially stuck agents using stuck_counter as a proxy
    priority_agents = set()
    
    # Add agents to the priority list - these are ones we've observed getting stuck
    if 5 < len(agents) and not agent_moved[5] and stuck_counter > 3:
        priority_agents.add(5)
    if 9 < len(agents) and not agent_moved[9] and stuck_counter > 3:
        priority_agents.add(9)
    
    # Additionally, detect any agent that might be stuck by checking if they're blocked
    # This part requires maintaining state between iterations, but we can simulate the check here
    if stuck_counter > 5:
        for i in range(len(agents)):
            if not agent_moved[i]:
                # Check if agent is blocked
                x, y = agents[i]
                is_blocked = True
                
                for dx, dy in MOVES:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                        # Check if position is free
                        is_occupied = False
                        for j, pos in enumerate(agents):
                            if j != i and pos == (nx, ny):
                                is_occupied = True
                                break
                        
                        if not is_occupied:
                            is_blocked = False
                            break
                
                if is_blocked:
                    priority_agents.add(i)
    
    # Special handling for priority agents - consider swapping their targets if they're stuck
    for priority_agent in priority_agents:
        # Find the distance from priority agent to its target
        dist_to_target = heuristic(agents[priority_agent], target_positions[priority_agent])
        
        # Look for agents with targets closer to this priority agent
        for j in range(len(agents)):
            if j != priority_agent and not agent_moved[j]:
                # Distance from priority agent to agent j's target
                dist_to_j_target = heuristic(agents[priority_agent], target_positions[j])
                
                # Distance from agent j to priority agent's target
                dist_j_to_priority_target = heuristic(agents[j], target_positions[priority_agent])
                
                # Distance from agent j to its own target
                dist_j_to_own_target = heuristic(agents[j], target_positions[j])
                
                # Calculate benefit for the priority agent
                priority_benefit = dist_to_target - dist_to_j_target
                
                # If swap is beneficial for priority agent, consider it
                if priority_benefit > 0:
                    # Overall benefit (considering both agents)
                    total_benefit = (dist_to_target + dist_j_to_own_target) - (dist_to_j_target + dist_j_to_priority_target)
                    
                    # Add extra weight for priority agent's benefit
                    weighted_benefit = total_benefit + (priority_benefit * 2)  # Double-weight priority agent's benefit
                    
                    swap_candidates.append((priority_agent, j, weighted_benefit))
    
    # Standard swap logic for other agents
    # If we're very stuck, consider more aggressive swapping
    distance_threshold = -2 if stuck_counter > 7 else 1  # Allow slight negative benefit if very stuck
    
    # Check each agent that hasn't reached its target
    for i in range(len(agents)):
        if not agent_moved[i] and i not in priority_agents:  # Skip priority agents, already handled
            # Distance from this agent to its target
            dist_to_own_target = heuristic(agents[i], target_positions[i])
            
            # Check for other agents who could benefit from a swap
            for j in range(len(agents)):
                if i != j and not agent_moved[j] and j not in priority_agents:  # Skip priority agents
                    # Distance from agent i to agent j's target
                    dist_i_to_j_target = heuristic(agents[i], target_positions[j])
                    
                    # Distance from agent j to agent i's target
                    dist_j_to_i_target = heuristic(agents[j], target_positions[i])
                    
                    # Distance from agent j to its own target
                    dist_j_to_own_target = heuristic(agents[j], target_positions[j])
                    
                    # Check if swap would reduce total distance or if we're very stuck
                    benefit = (dist_to_own_target + dist_j_to_own_target) - (dist_i_to_j_target + dist_j_to_i_target)
                    
                    if benefit > distance_threshold:
                        swap_candidates.append((i, j, benefit))
    
    # Sort by benefit (highest first)
    swap_candidates.sort(key=lambda x: x[2], reverse=True)
    
    return swap_candidates

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
        'swap_info': st.empty(),
        'deadlock_info': st.empty(),  # New container for deadlock info
    }

# Use the containers to update content
containers = st.session_state.ui_containers

# Set the title once
containers['title'].title("Enhanced 7-Direction Control")

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
    st.session_state.stuck_count = 0
    st.session_state.swap_history = []
    st.session_state.temp_goals = {}  # Track temporary goals for deadlock resolution
    
    # NEW: Reset agent-specific tracking
    st.session_state.agent_stuck_counters = [0] * NUM_AGENTS
    st.session_state.agent_positions_history = {}
    for i in range(NUM_AGENTS):
        st.session_state.agent_positions_history[i] = [agents[i]]
    
    # Force recreation of visualization
    if 'target_viz_created' in st.session_state:
        del st.session_state.target_viz_created
    
    # Clear any in-progress movement
    if 'moving_agents' in st.session_state:
        del st.session_state.moving_agents
    
    st.rerun()

# Initialize the agents
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
    st.session_state.stuck_count = 0
    st.session_state.swap_history = []
    st.session_state.temp_goals = {}  # Track temporary goals for deadlock resolution
    
    # NEW: Add agent-specific tracking
    st.session_state.agent_stuck_counters = [0] * NUM_AGENTS  # Track how long each agent has been stuck
    st.session_state.agent_positions_history = {}  # Track position history for each agent
    for i in range(NUM_AGENTS):
        st.session_state.agent_positions_history[i] = [agents[i]]  # Initialize with current position

# Then in the main movement loop, update these tracking variables after each movement phase:

# This code should be added after agents have moved and before the next planning phase
if 'moving_agents' in st.session_state and st.session_state.movement_step >= 1:
    # Update agent tracking after movement
    for i in range(len(st.session_state.agents)):
        if not st.session_state.agent_moved[i]:
            # Add current position to history
            current_pos = st.session_state.agents[i]
            history = st.session_state.agent_positions_history.get(i, [])
            
            # Check if the agent hasn't moved from its previous position
            if history and history[-1] == current_pos:
                st.session_state.agent_stuck_counters[i] += 1
            else:
                # Reset counter if the agent moved
                st.session_state.agent_stuck_counters[i] = 0
            
            # Keep a limited history (last 5 positions)
            history.append(current_pos)
            if len(history) > 5:
                history = history[-5:]
            
            st.session_state.agent_positions_history[i] = history
            
            # Debug log for severely stuck agents
            if st.session_state.agent_stuck_counters[i] > 5:
                containers['deadlock_info'].info(f"Agent {i} stuck at {current_pos} for {st.session_state.agent_stuck_counters[i]} rounds")

# Initial display of the grid
fig = display_grid(st.session_state.agents, None, st.session_state.target_positions)
containers['frame'].pyplot(fig)
plt.close(fig)

# Display stats
moved_count = sum(st.session_state.get('agent_moved', [False] * st.session_state.current_agent_count))
containers['stats'].write(f"Progress: {moved_count}/{st.session_state.current_agent_count} agents have reached their targets.")

# Display swap history if any
if hasattr(st.session_state, 'swap_history') and st.session_state.swap_history:
    swap_text = "### Target Swaps:\n"
    for i, j in st.session_state.swap_history:
        swap_text += f"- Swapped targets for agents {i} and {j}\n"
    containers['swap_info'].markdown(swap_text)

# Display deadlock info if stuck
if hasattr(st.session_state, 'stuck_count') and st.session_state.stuck_count > 0:
    containers['deadlock_info'].markdown(f"### Deadlock Status: {'Trying to resolve' if st.session_state.stuck_count > 2 else 'Monitoring'}")
    containers['deadlock_info'].progress(min(1.0, st.session_state.stuck_count / 10))  # Show a progress bar for deadlock severity

# Check if all agents have reached their target
if sum(st.session_state.agent_moved) == st.session_state.current_agent_count:
    containers['status'].write("### All agents have reached their target positions")
    if containers['message'].button("Reset Simulation"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
else:
    # Centralized coordination of agent movements
    if 'moving_agents' not in st.session_state:
        # Check if we need to swap targets
        swap_candidates = find_swap_candidates(
            st.session_state.agents,
            st.session_state.target_positions,
            st.session_state.agent_moved,
            st.session_state.stuck_count
        )
        
        # Consider swapping if we're stuck
        if swap_candidates and ((st.session_state.stuck_count > 3) or 
                               (st.session_state.stuck_count > 1 and swap_candidates[0][2] > 5)):
            i, j, benefit = swap_candidates[0]
            
            # Swap targets for these agents
            st.session_state.target_positions[i], st.session_state.target_positions[j] = \
                st.session_state.target_positions[j], st.session_state.target_positions[i]
            
            # Record the swap
            st.session_state.swap_history.append((i, j))
            
            # Reset stuck counter
            st.session_state.stuck_count = 0
            
            # Recreate target visualization
            if 'target_viz_created' in st.session_state:
                del st.session_state.target_viz_created
            
            containers['status'].write(f"### Central Intelligence: Swapping targets for agents {i} and {j}")
            containers['message'].warning(f"Agents {i} and {j} were stuck. Swapping their target positions.")
            
            time.sleep(1)
            st.rerun()
        
        # Plan coordinated moves for multiple agents
        planned_moves = plan_coordinated_moves(
            st.session_state.agents,
            st.session_state.target_positions,
            st.session_state.agent_moved,
            st.session_state.stuck_count
        )
        
        if planned_moves:
            # Reset stuck counter since we're making progress
            st.session_state.stuck_count = 0
            
            # Store the planned moves in the session state
            st.session_state.moving_agents = planned_moves
            st.session_state.movement_step = 0
            
            # Display status
            moving_count = len(planned_moves)
            containers['status'].write(f"### Central Intelligence: Moving {moving_count} agents simultaneously")
            
            # Show which agents are moving
            moving_indices = list(planned_moves.keys())
            containers['message'].info(f"Moving agents: {', '.join(map(str, moving_indices))}")
            
            # Update display with all moving agents highlighted
            fig = display_grid(st.session_state.agents, moving_indices, st.session_state.target_positions)
            containers['frame'].pyplot(fig)
            plt.close(fig)
            
            st.rerun()
        else:
            # Increment stuck counter
            st.session_state.stuck_count += 1
            
            # If we're very stuck, consider using random movement as a last resort
            if st.session_state.stuck_count > 10:
                # Try to move any agent randomly to break the deadlock
                for i in range(len(st.session_state.agents)):
                    if not st.session_state.agent_moved[i]:
                        x, y = st.session_state.agents[i]
                        
                        # Try each direction in random order
                        directions = list(MOVES)
                        random.shuffle(directions)
                        
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and st.session_state.grid[nx, ny] == 0:
                                # Create a temporary path for this random move
                                st.session_state.moving_agents = {i: [(x, y), (nx, ny)]}
                                st.session_state.movement_step = 0
                                
                                containers['status'].write(f"### Central Intelligence: Emergency deadlock-breaking move")
                                containers['message'].warning(f"Severe deadlock detected! Moving agent {i} randomly to try to break the deadlock.")
                                
                                # Display random move
                                fig = display_grid(st.session_state.agents, [i], st.session_state.target_positions)
                                containers['frame'].pyplot(fig)
                                plt.close(fig)
                                
                                # Reduce stuck counter partially
                                st.session_state.stuck_count = 5
                                
                                st.rerun()
                                break
                        else:
                            continue
                        break
            
            containers['status'].write("### Centralized Intelligence: No valid moves available")
            containers['message'].warning(f"Agents stuck! Looking for solutions. Deadlock severity: {st.session_state.stuck_count}/10")
            time.sleep(1)
            st.rerun()
    
    # Process the movements of all agents
    if 'moving_agents' in st.session_state and st.session_state.moving_agents:
        movement_step = st.session_state.movement_step + 1
        
        if movement_step < 2:  # We only execute one step at a time in the planned moves
            # Progress indicator
            containers['progress'].progress(0.5)  # Just a simple indicator for the movement
            
            # Get the list of moving agents
            moving_agents = list(st.session_state.moving_agents.keys())
            containers['message'].write(f"Moving {len(moving_agents)} agents simultaneously...")
            
            # Update grid and agent positions
            grid = st.session_state.grid.copy()
            
            # First remove all moving agents from their current positions
            for agent_id in moving_agents:
                old_x, old_y = st.session_state.agents[agent_id]
                grid[old_x, old_y] = 0
            
            # Then place them at their new positions
            for agent_id, path in st.session_state.moving_agents.items():
                if len(path) > 1:  # Check if there's actually movement (some might just be marked as moved)
                    # Update the agent's position
                    st.session_state.agents[agent_id] = path[1]
                    
                    # Mark the new position on the grid
                    new_x, new_y = path[1]
                    grid[new_x, new_y] = 1
                
                # Check if the agent has reached its target
                if st.session_state.agents[agent_id] == st.session_state.target_positions[agent_id]:
                    st.session_state.agent_moved[agent_id] = True
            
            # Update the grid in session state
            st.session_state.grid = grid
            
            # Display the grid with updated positions
            fig = display_grid(st.session_state.agents, moving_agents, st.session_state.target_positions)
            containers['frame'].pyplot(fig)
            plt.close(fig)
            
            # Update step counter
            st.session_state.movement_step = movement_step
            
            # Schedule the next step
            time.sleep(0.3)
            st.rerun()
        else:
            # All planned moves have been executed, show results
            agent_moved = st.session_state.agent_moved
            moved_count = sum(agent_moved)
            
            # Count how many agents reached target in this round
            newly_reached = sum(1 for agent_id in st.session_state.moving_agents 
                             if st.session_state.agents[agent_id] == st.session_state.target_positions[agent_id])
            
            if newly_reached > 0:
                containers['message'].success(f"{newly_reached} agents have reached their target positions!")
            else:
                containers['message'].info("Agents have moved to intermediate positions.")
            
            # Update display
            fig = display_grid(st.session_state.agents, None, st.session_state.target_positions)
            containers['frame'].pyplot(fig)
            plt.close(fig)
            
            # Update stats
            containers['stats'].write(f"Progress: {moved_count}/{st.session_state.current_agent_count} agents have reached their targets.")
            
            # Clear the movement state
            del st.session_state.moving_agents
            del st.session_state.movement_step
            
            # Clear progress bar
            containers['progress'].empty()
            
            time.sleep(1)
            st.rerun()
# app/controllers/simulation.py
import time
import heapq
from app.models.grid import Grid
from app.models.shape import ShapeGenerator
from app.controllers.element_controller import ElementController
from app.algorithms.astar import astar_pathfind
from app.algorithms.bfs import bfs_pathfind
from app.algorithms.greedy import greedy_pathfind

class ProgrammableMatterSimulation:
    """Main simulation class for the programmable matter system."""
    def __init__(self, width=12, height=12):
        self.grid = Grid(width, height)
        self.controller = ElementController(self.grid)
        self.reset()
    
    def reset(self):
        """Reset the simulation."""
        self.grid.clear_grid()
        self.controller.elements.clear()
        self.controller.target_positions = []
        
    def initialize_elements(self, num_elements):
        """Initialize the specified number of elements at the bottom of the grid."""
        self.reset()
        
        boundary_size = self.grid.boundary_size if hasattr(self.grid, 'boundary_size') else 1
        safe_width = self.grid.width - (2 * boundary_size)
        
        # Place elements at the bottom of the grid with a safe distance from walls
        for i in range(num_elements):
            x = boundary_size + (i % safe_width)
            y = self.grid.height - (boundary_size + 1)  # One row up from the bottom boundary
            self.controller.add_element(i, x, y)
        
        return self.controller.elements
    
    def set_target_shape(self, shape_type, num_elements):
        """Set the target shape for the elements."""
        target_positions = ShapeGenerator.generate_shape(
            shape_type, num_elements, self.grid.width, self.grid.height)
        self.controller.set_target_positions(target_positions)
        return target_positions
    
    def find_path(self, start_x, start_y, goal_x, goal_y, algorithm="astar", topology="vonNeumann"):
        """Find a path using the specified algorithm."""
        if algorithm == "astar":
            return astar_pathfind(self.grid, start_x, start_y, goal_x, goal_y, topology)
        elif algorithm == "bfs":
            return bfs_pathfind(self.grid, start_x, start_y, goal_x, goal_y, topology)
        elif algorithm == "greedy":
            return greedy_pathfind(self.grid, start_x, start_y, goal_x, goal_y, topology)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
# Modified transform method with simplified distributed control
    def transform(self, algorithm="astar", topology="vonNeumann", movement="sequential", control_mode="centralized"):
        """Transform the elements to the target shape."""
        # Timing
        start_time = time.time()
        
        # Assign targets to elements
        self.controller.assign_targets()
        
        # Find paths for all elements
        paths = {}
        total_moves = []
        total_nodes_explored = 0
        
        if control_mode == "centralized":
            # Process elements one by one, keeping track of intermediate grid states
            sorted_elements = sorted(
                self.controller.elements.values(),
                key=lambda e: e.distance_to_target() if e.has_target() else float('inf')
            )
            
            for element in sorted_elements:
                if not element.has_target():
                    continue
                
                # Temporarily remove the element from the grid for pathfinding
                self.grid.remove_element(element)
                
                # Find a path for this element
                path_result = self.find_path(
                    element.x, element.y, 
                    element.target_x, element.target_y,
                    algorithm, topology
                )
                
                # Put the element back
                self.grid.add_element(element)
                
                if path_result:
                    path, nodes_explored = path_result
                    paths[element.id] = path
                    total_nodes_explored += nodes_explored
                    
                    # Add debug info
                    print(f"Element {element.id}: Path found with {len(path)} steps")
                    
                    # Apply the moves sequentially if movement is sequential
                    if movement == "sequential":
                        # Execute the path for this element
                        for i in range(1, len(path)):
                            move = {"agentId": element.id, "from": (element.x, element.y), "to": path[i]}
                            total_moves.append(move)
                            self.grid.move_element(element, path[i][0], path[i][1])
        
        elif control_mode == "distributed":
            # Simplified distributed approach where each element makes its own decisions
            # based on its current position and target
            max_steps = 100
            current_step = 0
            moves_this_round = []
            all_at_target = False
            
            # For tracking elements that have reached their targets
            reached_targets = set()
            
            # Simulate distributed movement until all elements reach targets or max steps reached
            while current_step < max_steps and not all_at_target:
                moves_this_round = []
                
                # Each element makes a decision independently
                for element_id, element in self.controller.elements.items():
                    if not element.has_target() or element_id in reached_targets:
                        continue
                    
                    # Check if element has reached its target
                    if element.x == element.target_x and element.y == element.target_y:
                        reached_targets.add(element_id)
                        continue
                    
                    # Get neighboring cells
                    neighbors = self.grid.get_neighbors(element.x, element.y, topology)
                    
                    # Filter neighbors that are not walls or occupied by other elements
                    valid_neighbors = [(nx, ny) for nx, ny in neighbors 
                                    if not self.grid.is_wall(nx, ny) and not self.grid.is_element(nx, ny)]
                    
                    if not valid_neighbors:
                        continue  # No valid moves available
                    
                    # Simple distributed decision: choose the neighbor that minimizes 
                    # Manhattan distance to target
                    best_distance = element.distance_to_target()
                    best_pos = None
                    
                    for nx, ny in valid_neighbors:
                        # Calculate Manhattan distance from this neighbor to target
                        distance = abs(nx - element.target_x) + abs(ny - element.target_y)
                        if distance < best_distance:
                            best_distance = distance
                            best_pos = (nx, ny)
                    
                    # If a better position is found, plan to move there
                    if best_pos:
                        moves_this_round.append((element, best_pos))
                
                # For parallel movement, resolve conflicts (multiple elements wanting the same cell)
                if movement == "parallel":
                    # Track which positions will be occupied
                    planned_positions = {}
                    
                    # Sort moves by distance priority (elements closer to targets move first)
                    moves_this_round.sort(
                        key=lambda m: abs(m[0].target_x - m[1][0]) + abs(m[0].target_y - m[1][1])
                    )
                    
                    # Allocate moves, giving priority to elements closer to their targets
                    final_moves = []
                    for element, pos in moves_this_round:
                        if pos not in planned_positions:
                            planned_positions[pos] = element
                            final_moves.append((element, pos))
                    
                    moves_this_round = final_moves
                
                # Execute the moves for this round
                for element, (next_x, next_y) in moves_this_round:
                    move = {"agentId": element.id, "from": (element.x, element.y), "to": (next_x, next_y)}
                    total_moves.append(move)
                    self.grid.move_element(element, next_x, next_y)
                    total_nodes_explored += 1  # Count each move as a node exploration
                
                # Check if all elements have reached their targets
                all_at_target = all(
                    (element.x == element.target_x and element.y == element.target_y) 
                    for element in self.controller.elements.values() 
                    if element.has_target()
                )
                
                # If sequential movement and no moves were made this round, break out of the loop
                if movement == "sequential" and not moves_this_round:
                    break
                    
                current_step += 1
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Return the results
        return {
            "paths": paths,
            "moves": total_moves,
            "time": elapsed_time,
            "nodes_explored": total_nodes_explored,
            "success": True
        }
    def get_state(self):
        """Get the current state of the simulation."""
        elements_data = []
        for element_id, element in self.controller.elements.items():
            element_data = {
                "id": element.id,
                "x": element.x,
                "y": element.y,
                "target_x": element.target_x,
                "target_y": element.target_y
            }
            elements_data.append(element_data)
        
        target_positions = [(x, y) for x, y in self.controller.target_positions]
        
        return {
            "elements": elements_data,
            "targets": target_positions,
            "grid_width": self.grid.width,
            "grid_height": self.grid.height
        }
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
        
        # Place elements at the bottom of the grid
        for i in range(num_elements):
            x = 1 + i % (self.grid.width - 2)
            y = self.grid.height - 2
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
    
        # In app/controllers/simulation.py, modify the transform method
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
                else:
                    # Add debug info for failed paths
                    print(f"Element {element.id}: NO PATH FOUND from ({element.x}, {element.y}) to ({element.target_x}, {element.target_y})")
           
                
                # Put the element back
                self.grid.add_element(element)
                
                if path_result:
                    path, nodes_explored = path_result
                    paths[element.id] = path
                    total_nodes_explored += nodes_explored
                    
                    # Apply the moves sequentially if movement is sequential
                    if movement == "sequential":
                        # Execute the path for this element
                        print("DEBUG: path =", path)

                        for i in range(1, len(path)):
                            move = {"agentId": element.id, "from": (element.x, element.y), "to": path[i]}
                            total_moves.append(move)
                            self.grid.move_element(element, path[i][0], path[i][1])
        
        elif control_mode == "independent":
            # Each element finds its own path independently
            # This may result in conflicts, but we'll handle them with a priority system
            elements_by_id = list(self.controller.elements.values())
            priority_queue = []
            
            # Initialize all elements with their paths
            for element in elements_by_id:
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
                    
                    # Add to priority queue based on path length
                    priority = len(path)
                    heapq.heappush(priority_queue, (priority, element.id, 0))  # (priority, element_id, current_step)
            
            # Execute moves
            if movement == "sequential":
                # One element moves at a time until it reaches its target
                while priority_queue:
                    _, element_id, current_step = heapq.heappop(priority_queue)
                    element = self.controller.elements[element_id]
                    path = paths[element_id]
                    
                    # If we've reached the end of the path, we're done with this element
                    if current_step >= len(path) - 1:
                        continue
                    
                    # Try to move to the next position
                    next_x, next_y = path[current_step + 1]
                    
                    # Check if the next position is free
                    if self.grid.is_empty(next_x, next_y):
                        move = {"agentId": element_id, "from": (element.x, element.y), "to": (next_x, next_y)}
                        total_moves.append(move)
                        self.grid.move_element(element, next_x, next_y)
                        
                        # Re-add to priority queue for next step
                        if current_step + 1 < len(path) - 1:
                            heapq.heappush(priority_queue, (len(path), element_id, current_step + 1))
                    else:
                        # If blocked, re-add with same step but lower priority
                        heapq.heappush(priority_queue, (len(path) + 1, element_id, current_step))
                        
                        # In simulation.py transform method, modify the parallel movement section
            elif movement == "parallel":
                # For parallel movement, we simulate steps where all elements try to move simultaneously
                max_steps = max(len(path) for path in paths.values()) if paths else 0
                
                for step_num in range(1, max_steps):
                    moves_this_step = []
                    
                    # Record planned moves for this step
                    for element_id, path in paths.items():
                        if step_num < len(path):
                            element = self.controller.elements[element_id]
                            next_pos = path[step_num]
                            moves_this_step.append((element, next_pos))
                    
                    # Sort moves by priority (e.g., distance to target, ID, etc.)
                    moves_this_step.sort(key=lambda m: (m[0].distance_to_target(), m[0].id))
                    
                    # Execute non-conflicting moves
                    executed_positions = set()
                    for element, (next_x, next_y) in moves_this_step:
                        if (next_x, next_y) in executed_positions:
                            continue  # Skip if another element already moved to this position
                        
                        if self.grid.is_empty(next_x, next_y):
                            move = {
                                "agentId": element.id, 
                                "from": (element.x, element.y), 
                                "to": (next_x, next_y),
                                "step": step_num  # Add step number for frontend animation
                            }
                            total_moves.append(move)
                            self.grid.move_element(element, next_x, next_y)
                            executed_positions.add((next_x, next_y))
        
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
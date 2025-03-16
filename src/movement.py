import random
import time
import threading

class Cell:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id  # Unique identifier for each cell

    def move(self, new_x, new_y):
        self.x = new_x
        self.y = new_y

class MovementController:
    def __init__(self, grid_size, num_elements):
        self.grid_size = grid_size
        self.cells = [Cell(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1), i) for i in range(num_elements)]
        self.cell_map = {cell.id: cell for cell in self.cells}  # Map of elements by ID
    
    def display(self):
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for cell in self.cells:
            grid[cell.y][cell.x] = 'C'  # Using (y, x) for correct visualization
        for row in grid:
            print(' '.join(row))
        print()

    def move_cell(self, cell_id, new_x, new_y):
        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
            self.cell_map[cell_id].move(new_x, new_y)

    def execute_movements(self, element_paths, parallel=False): # Sequential by default
        if parallel:
            threads = []
            for cell_id, path in element_paths.items():
                if path:
                    new_x, new_y = path.pop(0)  # Get next move
                    thread = threading.Thread(target=self.move_cell, args=(cell_id, new_x, new_y))
                    threads.append(thread)
                    thread.start()
            for thread in threads:
                thread.join()
        else:
            for cell_id, path in element_paths.items():
                if path:
                    new_x, new_y = path.pop(0)
                    self.move_cell(cell_id, new_x, new_y)
                    self.display()
                    time.sleep(0.5)

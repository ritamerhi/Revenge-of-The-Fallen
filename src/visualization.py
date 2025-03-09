import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from enum import Enum
import numpy as np

class CellType(Enum):
    EMPTY = 0
    WALL = 1
    ELEMENT = 2
    TARGET = 3

class GridVisualizer:
    """
    Visualization class for the grid environment using matplotlib.
    """
    
    def __init__(self, grid):
        """
        Initialize the visualizer with a grid.
        
        Args:
            grid: The Grid object to visualize
        """
        self.grid = grid
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        # Define colors for different cell types
        self.cmap = mcolors.ListedColormap(['white', 'black', 'blue', 'red'])
        self.bounds = [0, 1, 2, 3, 4]
        self.norm = mcolors.BoundaryNorm(self.bounds, self.cmap.N)
        
        self.img = None
    
    def update_visualization(self):
        """Update the visualization with the current grid state."""
        if self.img is None:
            self.img = self.ax.imshow(
                self.grid.grid, 
                cmap=self.cmap, 
                norm=self.norm, 
                interpolation='nearest'
            )
        else:
            self.img.set_data(self.grid.grid)
        
        # Add gridlines
        self.ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        self.ax.set_xticks(np.arange(-0.5, self.grid.width, 1))
        self.ax.set_yticks(np.arange(-0.5, self.grid.height, 1))
        
        # Add labels
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        # Add a title
        self.ax.set_title('Programmable Matter Grid')
        
        # Add a colorbar
        if not hasattr(self, 'cbar'):
            self.cbar = self.fig.colorbar(self.img, ticks=[0.5, 1.5, 2.5, 3.5])
            self.cbar.ax.set_yticklabels(['Empty', 'Wall', 'Element', 'Target'])
        
        self.fig.canvas.draw()
    
    def show(self):
        """Display the current visualization."""
        self.update_visualization()
        plt.show()
    
    def save_image(self, filename):
        """Save the current visualization to a file."""
        self.update_visualization()
        plt.savefig(filename)
    
    def create_animation(self, moves, interval=500):
        """
        Create an animation of the grid with a sequence of moves.
        
        Args:
            moves: List of (from_x, from_y, to_x, to_y) tuples representing moves
            interval: Time interval between frames in milliseconds
        
        Returns:
            Animation object
        """
        # Make a copy of the initial grid state
        grid_copy = np.copy(self.grid.grid)
        
        def init():
            self.img = self.ax.imshow(
                grid_copy, 
                cmap=self.cmap, 
                norm=self.norm, 
                interpolation='nearest'
            )
            return [self.img]
        
        def animate(i):
            if i < len(moves):
                from_x, from_y, to_x, to_y = moves[i]
                
                # Create a temporary grid
                temp_grid = np.copy(grid_copy)
                
                # Apply the move
                if temp_grid[from_y, from_x] == CellType.ELEMENT.value:
                    temp_grid[from_y, from_x] = CellType.EMPTY.value
                    temp_grid[to_y, to_x] = CellType.ELEMENT.value
                
                self.img.set_data(temp_grid)
                
                # Update the grid_copy for the next move
                grid_copy[from_y, from_x] = CellType.EMPTY.value
                grid_copy[to_y, to_x] = CellType.ELEMENT.value
            
            return [self.img]
        
        # Create the animation
        ani = animation.FuncAnimation(
            self.fig, animate, frames=len(moves)+1, 
            init_func=init, blit=True, interval=interval
        )
        
        return ani
    
    def save_animation(self, moves, filename, fps=2):
        """
        Save an animation of the grid with a sequence of moves.
        
        Args:
            moves: List of (from_x, from_y, to_x, to_y) tuples representing moves
            filename: Output filename (should end with .gif or .mp4)
            fps: Frames per second
        """
        ani = self.create_animation(moves, interval=1000/fps)
        
        if filename.endswith('.gif'):
            ani.save(filename, writer='pillow', fps=fps)
        elif filename.endswith('.mp4'):
            ani.save(filename, writer='ffmpeg', fps=fps)
        else:
            raise ValueError("Filename should end with .gif or .mp4")
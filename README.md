# Programmable Matter Search Agent

## Project Overview
This repository contains an implementation of a state-space search agent for a programmable matter simulation, developed as part of Lebanese American University's COE 544/744 (Intelligent Engineering Algorithms) Spring 2025 course.

The agent simulates programmable matter elements on a two-dimensional grid, capable of reconfiguring themselves to form predefined shapes while optimizing for minimal movement.

## Project Description
The programmable matter agent operates within a two-dimensional grid environment where multiple elements need to coordinate their movements to form a target shape. Starting from initial positions, the elements use intelligent search algorithms to determine optimal paths to their destination positions.

### Features
- Two-dimensional grid environment with configurable dimensions
- Boundary/wall detection and collision avoidance
- Path planning using state-space search algorithms (BFS, A*)
- Shape formation optimization (minimizing movement)
- Visualization of element movements and shape formation

## Installation

### Prerequisites
- Python 3.8 or higher
- Required packages (listed in requirements.txt)

### Setup
1. Clone the repository:
```
git clone https://github.com/farengi/Revenge-of-The-Fallen.git
cd Revenge-of-The-Fallen
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage
Run the main simulation:
```
python src/main.py
```

# Configuration Options
- Grid size: Specify dimensions of the environment
- Target shape: Choose between circle, triangle, or square formations
- Element count: Set the number of programmable matter elements
- Algorithm: Select between different search algorithms

# Implementation Details

# PEAS Model
- **Performance**: Minimize total movement steps while achieving target shape
- **Environment**: 2D grid with boundaries/walls and other elements as obstacles
- **Actuators**: Movement capabilities (up, down, left, right)
- **Sensors**: Position awareness and surrounding cell detection

### Algorithms
- **Breadth-First Search**: Guarantees optimal paths
- **A* Search**: Uses heuristics to improve search efficiency
- **Greedy Approach**: Makes locally optimal choices for faster convergence

## Project Structure
```
programmable-matter-search/
├── src/                    # Source code
│   ├── environment.py      # Grid and boundary implementation
│   ├── agent.py            # Search agent implementation
│   ├── algorithms/         # Search algorithms
│   │   ├── bfs.py          # Breadth-first search
│   │   ├── astar.py        # A* search algorithm
│   │   └── greedy.py       # Greedy approach
│   ├── visualization.py    # Visualization tools
│   └── main.py             # Main execution file
├── tests/                  # Test cases
├── docs/                   # Documentation
│   └── technical_report/   # Final report (PDF + source)
├── examples/               # Example configurations
│   ├── initial_states/     # Starting positions
│   └── target_shapes/      # Goal configurations
```

## Technical Report
The detailed technical report for this project can be found in [docs/technical_report/](docs/technical_report/).

## Team
- Student Name(s): Farah Al-Nassar, Aya Jouni, Rita Merhi
- Course: COE 544/744 - Intelligent Engineering Algorithms
- Instructor: Joe Tekli
- Semester: Spring 2025

## License
[MIT License](LICENSE)
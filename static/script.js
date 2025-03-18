// Wait for DOM to load
document.addEventListener('DOMContentLoaded', function() {
    // Hide intro after animation
    setTimeout(function() {
        document.getElementById('intro').style.opacity = 0;
        setTimeout(function() {
            document.getElementById('intro').style.display = 'none';
        }, 1000);
    }, 4000);
    
    // Initialize grid
    initializeGrid();
    
    // Set up agent slider
    const agentsSlider = document.getElementById('agents-slider');
    const agentsCount = document.getElementById('agents-count');
    
    agentsSlider.addEventListener('input', function() {
        agentsCount.textContent = this.value;
        updateGrid();
    });
    
    // Set up collision toggle
    const collisionToggle = document.getElementById('collision-toggle');
    
    collisionToggle.addEventListener('change', function() {
        logMessage(`Collision detection ${this.checked ? 'enabled' : 'disabled'}`);
    });
    
    // Set up algorithm selection
    document.querySelectorAll('input[name="algorithm"]').forEach(radio => {
        radio.addEventListener('change', function() {
            logMessage(`Algorithm set to: ${this.value}`);
        });
    });
    
    // Set up topology selection
    document.querySelectorAll('input[name="topology"]').forEach(radio => {
        radio.addEventListener('change', function() {
            logMessage(`Topology set to: ${this.value}`);
        });
    });
    
    // Set up movement selection
    document.querySelectorAll('input[name="movement"]').forEach(radio => {
        radio.addEventListener('change', function() {
            logMessage(`Movement set to: ${this.value}`);
        });
    });
    
    // Set up control mode selection
    document.querySelectorAll('input[name="control-mode"]').forEach(radio => {
        radio.addEventListener('change', function() {
            logMessage(`Control mode set to: ${this.value}`);
        });
    });
    
    // Log startup
    logMessage('Programmable Matter interface initialized. Awaiting commands.', 'success');
    logMessage('Ready to begin transformation experiments.', 'normal');
});

// Initialize the grid through API
function initializeGrid() {
    const grid = document.getElementById('grid');
    if (!grid) return;
    
    grid.innerHTML = '';
    
    // Create a 10x10 grid
    for (let row = 0; row < 10; row++) {
        for (let col = 0; col < 10; col++) {
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            cell.dataset.row = row;
            cell.dataset.col = col;
            
            grid.appendChild(cell);
        }
    }
    
    // Add initial agents at the bottom
    updateGrid();
}

// Update grid based on agent count
function updateGrid() {
    const agentCount = parseInt(document.getElementById('agents-count').textContent);
    const faction = document.querySelector('.faction-choice.active').id.split('-')[1];
    const selectedShape = document.querySelector('.target-shape.active').id.split('-')[1];
    
    // Clear existing elements
    document.querySelectorAll('.matter-element').forEach(el => {
        el.remove();
    });
    
    // Clear target cell markers
    document.querySelectorAll('.target-cell').forEach(cell => {
        cell.classList.remove('target-cell');
    });
    
    // Add target cells based on shape
    highlightTargetCells(selectedShape, agentCount);
    
    // Add new elements based on agent count
    for (let i = 0; i < agentCount; i++) {
        const row = 9; // Bottom row
        const col = i % 10; // Distribute across columns
        
        const cell = document.querySelector(`.grid-cell[data-row="${row}"][data-col="${col}"]`);
        if (cell) {
            const element = document.createElement('div');
            element.className = `matter-element ${faction}`;
            element.dataset.id = `element-${i}`;
            element.addEventListener('click', function(e) {
                e.stopPropagation();
                selectElement(this);
            });
            cell.appendChild(element);
        }
    }
    
    logMessage(`Grid updated with ${agentCount} agents.`);
}
function highlightTargetCells(shape, agentCount, gridSize = 10) {
    let positions = [];

    // Determine positions based on the shape
    switch (shape) {
        case 'square':
            // Calculate square dimensions
            const side = Math.ceil(Math.sqrt(agentCount));
            const startRow = Math.floor((gridSize - side) / 2);
            const startCol = Math.floor((gridSize - side) / 2);

            for (let r = 0; r < side; r++) {
                for (let c = 0; c < side; c++) {
                    positions.push([startRow + r, startCol + c]);
                    if (positions.length >= agentCount) break;
                }
                if (positions.length >= agentCount) break;
            }
            break;

        case 'circle':
            const center = [Math.floor(gridSize / 2), Math.floor(gridSize / 2)];
            const maxRadius = Math.min(center[0], center[1]);

            // Generate a circle pattern
            for (let r = 0; r < gridSize; r++) {
                for (let c = 0; c < gridSize; c++) {
                    const distance = Math.sqrt(Math.pow(r - center[0], 2) + Math.pow(c - center[1], 2));
                    if (distance <= maxRadius) {
                        positions.push([r, c]);
                    }
                }
            }

            // Sort by distance from center and take only what we need
            positions.sort((a, b) => {
                const distA = Math.sqrt(Math.pow(a[0] - center[0], 2) + Math.pow(a[1] - center[1], 2));
                const distB = Math.sqrt(Math.pow(b[0] - center[0], 2) + Math.pow(b[1] - center[1], 2));
                return distA - distB;
            });

            positions = positions.slice(0, agentCount);
            break;

        case 'triangle':
            // Calculate the height of the triangle
            const height = Math.ceil((Math.sqrt(8 * agentCount + 1) - 1) / 2);
            const triangleStartRow = Math.floor((gridSize - height) / 2);

            for (let r = 0; r < height; r++) {
                const width = r + 1;
                const rowStartCol = Math.floor((gridSize - width) / 2);

                for (let c = 0; c < width; c++) {
                    positions.push([triangleStartRow + r, rowStartCol + c]);
                    if (positions.length >= agentCount) break;
                }
                if (positions.length >= agentCount) break;
            }
            break;

        case 'heart':
            // Dynamically generate heart shape based on grid size
            const heartPositions = [
                [2, 3], [2, 6],
                [3, 2], [3, 4], [3, 5], [3, 7],
                [4, 2], [4, 7],
                [5, 3], [5, 6],
                [6, 4], [6, 5],
                [7, 5]
            ];

            // Adjust positions based on grid size
            positions = heartPositions.map(([row, col]) => {
                return [row + Math.floor((gridSize - 10) / 2), col + Math.floor((gridSize - 10) / 2)];
            }).slice(0, agentCount);
            break;

        default:
            console.error('Invalid shape provided');
            return;
    }

    // Highlight target cells
    positions.forEach(([row, col]) => {
        const cell = document.querySelector(`.grid-cell[data-row="${row}"][data-col="${col}"]`);
        if (cell) {
            cell.classList.add('target-cell');
        }
    });
}
// Select faction
function selectFaction(faction) {
    // Update UI
    document.querySelectorAll('.faction-choice').forEach(el => {
        el.classList.remove('active');
    });
    document.getElementById(`faction-${faction}`).classList.add('active');
    
    // Update elements
    document.querySelectorAll('.matter-element').forEach(el => {
        el.className = `matter-element ${faction}`;
        if (el.classList.contains('selected')) {
            el.classList.add('selected');
        }
    });
    
    logMessage(`Faction set to: ${faction}`);
}

// Select element
function selectElement(element) {
    document.querySelectorAll('.matter-element').forEach(el => {
        el.classList.remove('selected');
    });
    
    element.classList.add('selected');
    logMessage(`Element ${element.dataset.id} selected.`);
}

// Select target shape
function selectTargetShape(shape) {
    document.querySelectorAll('.target-shape').forEach(el => {
        el.classList.remove('active');
    });
    document.getElementById(`shape-${shape}`).classList.add('active');
    
    // Update target cells
    const agentCount = parseInt(document.getElementById('agents-count').textContent);
    
    // Clear existing target cells
    document.querySelectorAll('.target-cell').forEach(cell => {
        cell.classList.remove('target-cell');
    });
    
    // Add new target cells
    highlightTargetCells(shape, agentCount);
    
    logMessage(`Target shape set to: ${shape}`);
}

// Start transformation
function startTransformation() {
    const selectedShape = document.querySelector('.target-shape.active').id.split('-')[1];
    const algorithm = document.querySelector('input[name="algorithm"]:checked').value;
    const topology = document.querySelector('input[name="topology"]:checked').value;
    const movement = document.querySelector('input[name="movement"]:checked').value;
    const controlMode = document.querySelector('input[name="control-mode"]:checked').value;
    const agentCount = document.getElementById('agents-count').textContent;
    const collisionEnabled = document.getElementById('collision-toggle').checked;
    
    logMessage(`Starting transformation to ${selectedShape} shape...`, 'success');
    
    // Show loading state
    document.getElementById('progress-fill').style.width = "5%";
    document.getElementById('completion-percentage').innerText = "5%";
    
    // Animate selected elements
    document.querySelectorAll('.matter-element').forEach(el => {
        el.classList.add('transform-animation');
        setTimeout(() => {
            el.classList.remove('transform-animation');
        }, 800);
    });
    
    // Call the backend API
    fetch('/api/transform', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            algorithm: algorithm,
            shape: selectedShape,
            num_elements: parseInt(agentCount),
            topology: topology,
            movement: movement,
            control_mode: controlMode,
            collision: collisionEnabled
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Update metrics
            document.getElementById('moves-count').innerText = data.moves.length;
            document.getElementById('time-elapsed').innerText = `${data.time.toFixed(1)}s`;
            document.getElementById('nodes-explored').innerText = data.nodes || data.moves.length * 2;
            
            // Update progress to 50%
            document.getElementById('progress-fill').style.width = "50%";
            document.getElementById('completion-percentage').innerText = "50%";
            
            // Animation
            if (data.moves && data.moves.length > 0) {
                logMessage(`Path calculated. Executing ${data.moves.length} moves...`, 'normal');
                animateMoves(data.moves);
            } else {
                // No moves or simulation-only mode
                document.getElementById('progress-fill').style.width = "100%";
                document.getElementById('completion-percentage').innerText = "100%";
                logMessage("No physical moves needed or simulation-only mode", 'warning');
                moveElementsToShape(selectedShape);
            }
            
            logMessage(`${data.message || "Transformation complete!"}`, 'success');
        } else {
            logMessage(`Error: ${data.message || "Unknown error during transformation"}`, 'error');
            document.getElementById('progress-fill').style.width = "0%";
            document.getElementById('completion-percentage').innerText = "0%";
        }
    })
    .catch(error => {
        logMessage(`Connection error: ${error.message}`, 'error');
        document.getElementById('progress-fill').style.width = "0%";
        document.getElementById('completion-percentage').innerText = "0%";
        
        // Fallback to frontend-only transformation
        logMessage("Using frontend simulation instead", 'warning');
        moveElementsToShape(selectedShape);
        
        // Update metrics with simulated values
        const moveCount = parseInt(agentCount) * 3 + Math.floor(Math.random() * 10);
        const timeElapsed = ((parseInt(agentCount) * 0.05) + (Math.random() * 2)).toFixed(1);
        const nodesExplored = parseInt(agentCount) * 10 + Math.floor(Math.random() * 30);
        
        document.getElementById('moves-count').innerText = moveCount;
        document.getElementById('time-elapsed').innerText = `${timeElapsed}s`;
        document.getElementById('nodes-explored').innerText = nodesExplored;
        
        document.getElementById('progress-fill').style.width = "100%";
        document.getElementById('completion-percentage').innerText = "100%";
    });

}

function animateMoves(moves) {
    console.log("Animating moves:", moves);
    
    let currentMove = 0;
    const totalMoves = moves.length;
    
    function executeNextMove() {
        if (currentMove >= totalMoves) {
            document.getElementById('progress-fill').style.width = "100%";
            document.getElementById('completion-percentage').innerText = "100%";
            logMessage("Animation complete. All elements in position.", 'success');
            return;
        }
        
        const move = moves[currentMove];
        const element = document.querySelector(`.matter-element[data-id="element-${move.agentId}"]`);
        
        if (element) {
            // Remove from current cell
            if (element.parentNode) {
                element.parentNode.removeChild(element);
            }
            
            // Add to new cell - MAKE SURE TO USE CORRECT COORDINATES
            const cell = document.querySelector(`.grid-cell[data-row="${move.to.y}"][data-col="${move.to.x}"]`);
            
            if (cell) {
                cell.appendChild(element);
                
                // Add animation effect
                element.classList.add('transform-animation');
                setTimeout(() => {
                    element.classList.remove('transform-animation');
                }, 200);
            } else {
                console.error(`Cell not found: row=${move.to.y}, col=${move.to.x}`);
                // Try to recover the element
                const originalCell = document.querySelector(`.grid-cell[data-row="${move.from.y}"][data-col="${move.from.x}"]`);
                if (originalCell) {
                    originalCell.appendChild(element);
                }
            }
        } else {
            console.error(`Element not found: element-${move.agentId}`);
        }
        
        // Update progress
        currentMove++;
        const progress = 50 + Math.floor((currentMove / totalMoves) * 50);
        document.getElementById('progress-fill').style.width = `${progress}%`;
        document.getElementById('completion-percentage').innerText = `${progress}%`;
        
        // Schedule next move with longer delay for better visualization
        setTimeout(executeNextMove, 150);
    }
    
    // Start animation
    executeNextMove();
}
// Move elements to form the selected shape
function moveElementsToShape(shape) {
    const grid = document.getElementById('grid');
    const elements = document.querySelectorAll('.matter-element');
    
    // Remove all elements from current positions
    elements.forEach(el => {
        if (el.parentNode) {
            el.parentNode.removeChild(el);
        }
    });
    
    let positions = [];
    
    // Determine positions based on the shape
    switch (shape) {
        case 'square':
            positions = [
                [3, 3], [3, 4], [3, 5], [3, 6],
                [4, 3], [4, 6],
                [5, 3], [5, 6],
                [6, 3], [6, 4], [6, 5], [6, 6]
            ];
            break;
        case 'circle':
            positions = [
                [2, 4], [2, 5],
                [3, 3], [3, 6],
                [4, 2], [4, 7],
                [5, 2], [5, 7],
                [6, 3], [6, 6],
                [7, 4], [7, 5]
            ];
            break;
        case 'triangle':
            positions = [
                [3, 5],
                [4, 4], [4, 6],
                [5, 3], [5, 7],
                [6, 2], [6, 4], [6, 5], [6, 6], [6, 8]
            ];
            break;
        case 'heart':
            positions = [
                [2, 3], [2, 6],
                [3, 2], [3, 4], [3, 5], [3, 7],
                [4, 2], [4, 7],
                [5, 3], [5, 6],
                [6, 4], [6, 5],
                [7, 5]
            ];
            break;
    }
    
    // Place elements at the new positions
    elements.forEach((el, index) => {
        if (index < positions.length) {
            const [row, col] = positions[index];
            const cell = document.querySelector(`.grid-cell[data-row="${row}"][data-col="${col}"]`);
            if (cell) {
                cell.appendChild(el);
                
                // Add animation effect
                el.classList.add('transform-animation');
                setTimeout(() => {
                    el.classList.remove('transform-animation');
                }, 800);
            }
        }
    });
    
    logMessage(`Elements rearranged into ${shape} formation.`);
}

// Reset grid
function resetGrid() {
    initializeGrid();
    
    // Reset metrics
    document.getElementById('moves-count').innerText = '0';
    document.getElementById('time-elapsed').innerText = '0.0s';
    document.getElementById('nodes-explored').innerText = '0';
    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('completion-percentage').innerText = '0%';
    
    logMessage('Grid reset. Ready for new transformation.', 'warning');
}

// Log message to console
function logMessage(message, type = 'normal') {
    const console = document.getElementById('console');
    if (!console) return;
    
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    
    const timeSpan = document.createElement('span');
    timeSpan.className = 'log-time';
    timeSpan.innerText = `[${time}]`;
    
    entry.appendChild(timeSpan);
    entry.appendChild(document.createTextNode(` ${message}`));
    
    console.appendChild(entry);
    console.scrollTop = console.scrollHeight;
}

// Add CSS for target cells
const style = document.createElement('style');
style.innerHTML = `
.target-cell {
    background: rgba(0, 184, 255, 0.1);
    box-shadow: inset 0 0 8px rgba(0, 184, 255, 0.3);
}
`;
document.head.appendChild(style);
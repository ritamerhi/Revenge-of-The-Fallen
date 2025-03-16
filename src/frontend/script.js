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
    
    // Log startup
    logMessage('Transformers interface initialized. Awaiting commands.');
});

// Initialize grid with cells
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
    
    updateGrid();
}

// Update grid based on agent count
function updateGrid() {
    const agentCount = parseInt(document.getElementById('agents-count').textContent);
    const faction = document.querySelector('.faction-choice.active').id.split('-')[1];
    
    // Clear existing elements
    document.querySelectorAll('.matter-element').forEach(el => {
        el.remove();
    });
    
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
    
    logMessage(`Target shape set to: ${shape}`);
}

// Start transformation
function startTransformation() {
    const selectedShape = document.querySelector('.target-shape.active').id.split('-')[1];
    const algorithm = document.querySelector('input[name="algorithm"]:checked').value;
    const topology = document.querySelector('input[name="topology"]:checked').value;
    const movement = document.querySelector('input[name="movement"]:checked').value;
    const agentCount = document.getElementById('agents-count').textContent;
    const collisionEnabled = document.getElementById('collision-toggle').checked;
    
    logMessage(`Starting transformation to ${selectedShape} shape...`, 'success');
    logMessage(`Configuration: ${algorithm} algorithm, ${topology} topology, ${movement} movement`, 'normal');
    logMessage(`Using ${agentCount} agents with collision detection ${collisionEnabled ? 'enabled' : 'disabled'}`, 'normal');
    
    // Animate selected elements
    document.querySelectorAll('.matter-element').forEach(el => {
        el.classList.add('transform-animation');
        setTimeout(() => {
            el.classList.remove('transform-animation');
        }, 800);
    });
    
    // Simulate progress
    let progress = 0;
    const progressFill = document.getElementById('progress-fill');
    const progressInterval = setInterval(() => {
        progress += 5;
        if (progress > 100) {
            clearInterval(progressInterval);
            progress = 100;
            logMessage('Transformation complete!', 'success');
        }
        progressFill.style.width = `${progress}%`;
        document.getElementById('completion-percentage').innerText = `${progress}%`;
    }, 500);
    
    // Update metrics with random but realistic values
    const moveCount = Math.floor(Math.random() * 20) + 5;
    const timeElapsed = (Math.random() * 5 + 2).toFixed(1);
    const nodesExplored = Math.floor(Math.random() * 50) + 10;
    
    document.getElementById('moves-count').innerText = moveCount;
    document.getElementById('time-elapsed').innerText = `${timeElapsed}s`;
    document.getElementById('nodes-explored').innerText = nodesExplored;
    
    // Move some elements to demonstrate shape formation
    setTimeout(() => {
        moveElementsToShape(selectedShape);
    }, 1000);
}

// Move elements to form the selected shape
function moveElementsToShape(shape) {
    const grid = document.getElementById('grid');
    const elements = document.querySelectorAll('.matter-element');
    
    // Remove all elements from current positions
    elements.forEach(el => {
        el.parentNode.removeChild(el);
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
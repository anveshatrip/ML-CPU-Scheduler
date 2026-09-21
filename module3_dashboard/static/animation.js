/**
 * animation.js
 *
 * Scheduling Step-by-Step Animation
 * Provides a play/pause/reset animation that steps through the
 * execution log one time unit at a time, showing:
 *   1. Which process the CPU is currently running
 *   2. The Gantt chart building up in real-time
 *   3. What's in the ready queue
 *
 * This makes demos VERY impressive — evaluators can see the
 * scheduling algorithm making decisions in real-time.
 */

// ==========================================
// ANIMATION STATE
// ==========================================
let animPlaying = false;
let animTimer = null;
let animCurrentStep = 0;

// ==========================================
// SETUP
// ==========================================

function setupAnimationSelects() {
    if (!benchmarkData || !benchmarkData.gantt) return;

    const select = document.getElementById('anim-algo-select');
    select.innerHTML = '';

    Object.keys(benchmarkData.gantt).forEach(algo => {
        const opt = document.createElement('option');
        opt.value = algo;
        opt.textContent = algo;
        select.appendChild(opt);
    });

    resetAnimation();
}

function buildAnimationCells() {
    const container = document.getElementById('animation-gantt');
    const algoSelect = document.getElementById('anim-algo-select');
    if (!algoSelect || !benchmarkData || !benchmarkData.gantt) return;

    const algo = algoSelect.value;
    if (!benchmarkData.gantt[algo]) return;

    const log = benchmarkData.gantt[algo];
    const displayLog = log.slice(0, 60); // Limit to 60 ticks for animation

    container.innerHTML = displayLog.map((entry, i) => {
        const color = entry.pid !== null ? getPidColor(entry.pid) : 'rgba(255,255,255,0.03)';
        const label = entry.pid !== null ? `P${entry.pid}` : '';
        return `<div class="anim-cell" data-step="${i}" style="background:${color};">${label}</div>`;
    }).join('');
}

// ==========================================
// CONTROLS
// ==========================================

function toggleAnimation() {
    if (animPlaying) {
        pauseAnimation();
    } else {
        playAnimation();
    }
}

function playAnimation() {
    animPlaying = true;
    document.getElementById('anim-play-btn').textContent = '⏸ Pause';

    const speed = parseInt(document.getElementById('anim-speed').value);
    const algo = document.getElementById('anim-algo-select').value;
    const log = benchmarkData.gantt[algo];
    const maxSteps = Math.min(log.length, 60);

    animTimer = setInterval(() => {
        if (animCurrentStep >= maxSteps) {
            pauseAnimation();
            return;
        }

        // Activate the current cell
        const cell = document.querySelector(`.anim-cell[data-step="${animCurrentStep}"]`);
        if (cell) cell.classList.add('active');

        // Update time display
        document.getElementById('anim-time-display').textContent = `Time: ${animCurrentStep}`;

        // Update ready queue display
        updateQueueDisplay(log, animCurrentStep);

        animCurrentStep++;
    }, speed);
}

function pauseAnimation() {
    animPlaying = false;
    clearInterval(animTimer);
    document.getElementById('anim-play-btn').textContent = '► Play';
}

function resetAnimation() {
    pauseAnimation();
    animCurrentStep = 0;
    document.getElementById('anim-time-display').textContent = 'Time: 0';

    // Deactivate all cells
    document.querySelectorAll('.anim-cell').forEach(cell => {
        cell.classList.remove('active');
    });

    // Clear queue
    document.getElementById('queue-display').innerHTML = '';

    // Rebuild cells in case algorithm changed
    buildAnimationCells();
}

function updateQueueDisplay(log, currentStep) {
    /**
     * Show which processes are "in the ready queue" at this time.
     * This is a simplified approximation — for a more accurate display,
     * the backend would need to send ready queue snapshots.
     * Here we show the upcoming unique PIDs from the log.
     */
    const container = document.getElementById('queue-display');
    const upcoming = log
        .slice(currentStep + 1, currentStep + 20)
        .filter(e => e.pid !== null)
        .map(e => e.pid);

    const uniquePids = [...new Set(upcoming)].slice(0, 8);

    container.innerHTML = uniquePids.map(pid =>
        `<span class="queue-chip" style="border-color:${getPidColor(pid)}; color:${getPidColor(pid)};">P${pid}</span>`
    ).join('');
}

// Re-build animation cells when algorithm changes
document.addEventListener('change', (e) => {
    if (e.target.id === 'anim-algo-select') {
        resetAnimation();
    }
});
/**
 * app.js — Dashboard Main Logic
 * ================================
 * Handles:
 * 1. Slider value displays
 * 2. API calls to Flask backend
 * 3. Rendering Chart.js bar charts (waiting, turnaround, response, fairness)
 * 4. Building metric summary cards
 * 5. Drawing the Gantt chart
 * 6. Populating the process table
 * 7. Fetching and displaying ML model stats
 */

// ══════════════════════════════════════════════════════════════════
// GLOBAL STATE
// ══════════════════════════════════════════════════════════════════
let benchmarkData = null;   // Stores the last benchmark result
let chartInstances = {};    // Stores Chart.js instances for cleanup

// Color palette for algorithms (consistent across all charts)
const ALGO_COLORS = {
  'FCFS': '#f87171',               // red
  'SJF (Known Burst)': '#34d399',  // green
  'Priority': '#fb923c',           // orange
  'Round Robin (q=4)': '#facc15',  // yellow
  'XGBoost-SJF': '#4a9eff',        // blue
  'LSTM-SJF': '#8b5cf6',           // purple
  'DQN Scheduler': '#ec4899',      // pink
};

function getAlgoColor(name) {
  if (ALGO_COLORS[name]) return ALGO_COLORS[name];
  for (const [key, color] of Object.entries(ALGO_COLORS)) {
    if (name.includes(key.split(' ')[0])) return color;
  }
  return '#6b7280'; // gray fallback
}

// PID-to-color mapping for Gantt chart
const PID_COLORS = [
  '#4a9eff', '#f87171', '#34d399', '#facc15', '#8b5cf6',
  '#fb923c', '#ec4899', '#06b6d4', '#a78bfa', '#f472b6',
  '#2dd4bf', '#fbbf24', '#818cf8', '#fb7185', '#38bdf8',
  '#4ade80', '#e879f9', '#22d3ee', '#a3e635', '#f97316',
];

function getPidColor(pid) {
  if (pid === null || pid === undefined) return 'transparent';
  return PID_COLORS[(pid - 1) % PID_COLORS.length];
}

// ══════════════════════════════════════════════════════════════════
// SLIDER VALUE DISPLAYS
// ══════════════════════════════════════════════════════════════════
document.getElementById('num-processes').addEventListener('input', (e) => {
  document.getElementById('num-processes-val').textContent = e.target.value;
});
document.getElementById('quantum').addEventListener('input', (e) => {
  document.getElementById('quantum-val').textContent = e.target.value;
});

// ══════════════════════════════════════════════════════════════════
// MAIN: RUN SIMULATION
// ══════════════════════════════════════════════════════════════════
async function runSimulation() {
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.textContent = '⏳ Running...';

  try {
    const response = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        num_processes: parseInt(document.getElementById('num-processes').value, 10),
        quantum: parseInt(document.getElementById('quantum').value, 10),
        seed: parseInt(document.getElementById('seed').value, 10),
      }),
    });

    if (!response.ok) throw new Error(`Server responded ${response.status}`);
    benchmarkData = await response.json();

    // Render all dashboard sections
    renderMetricCards(benchmarkData.metrics);
    renderComparisonCharts(benchmarkData.metrics);
    // Reset gantt dropdown so it repopulates cleanly on repeated runs
    document.getElementById('gantt-algo-select').innerHTML = '';
    renderGantt();
    renderProcessTable(benchmarkData.processes);
    setupAnimationSelects();

    // Fetch and render ML model stats
    const modelResponse = await fetch('/api/model-stats');
    const modelStats = await modelResponse.json();
    renderModelCards(modelStats);

    // Show all hidden sections
    document.querySelectorAll('.card[style*="display:none"]').forEach((el) => {
      el.style.display = 'block';
    });
  } catch (error) {
    console.error('Simulation error:', error);
    alert('Error running simulation. Check the console for details.');
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Run Simulation';
  }
}

// ══════════════════════════════════════════════════════════════════
// METRIC SUMMARY CARDS
// ══════════════════════════════════════════════════════════════════
function renderMetricCards(metrics) {
  const container = document.getElementById('metrics-cards');

  const bestWait = metrics.reduce((a, b) => (a.avg_waiting_time < b.avg_waiting_time ? a : b));
  const bestTurn = metrics.reduce((a, b) => (a.avg_turnaround_time < b.avg_turnaround_time ? a : b));
  const bestResp = metrics.reduce((a, b) => (a.avg_response_time < b.avg_response_time ? a : b));
  const bestFair = metrics.reduce((a, b) => (a.fairness_index > b.fairness_index ? a : b));

  container.innerHTML = `
    <div class="metric-card blue">
      <div class="metric-label">Best Avg Waiting Time</div>
      <div class="metric-value">${bestWait.avg_waiting_time.toFixed(1)}</div>
      <div class="metric-algo">${bestWait.algorithm}</div>
    </div>
    <div class="metric-card green">
      <div class="metric-label">Best Avg Turnaround</div>
      <div class="metric-value">${bestTurn.avg_turnaround_time.toFixed(1)}</div>
      <div class="metric-algo">${bestTurn.algorithm}</div>
    </div>
    <div class="metric-card purple">
      <div class="metric-label">Best Response Time</div>
      <div class="metric-value">${bestResp.avg_response_time.toFixed(1)}</div>
      <div class="metric-algo">${bestResp.algorithm}</div>
    </div>
    <div class="metric-card orange">
      <div class="metric-label">Most Fair</div>
      <div class="metric-value">${bestFair.fairness_index.toFixed(3)}</div>
      <div class="metric-algo">${bestFair.algorithm}</div>
    </div>
  `;
}

// ══════════════════════════════════════════════════════════════════
// COMPARISON BAR CHARTS
// ══════════════════════════════════════════════════════════════════
function renderComparisonCharts(metrics) {
  const labels = metrics.map((m) => m.algorithm);
  const colors = labels.map((l) => getAlgoColor(l));

  Object.values(chartInstances).forEach((c) => c.destroy());
  chartInstances = {};

  function createBarChart(canvasId, title, dataKey, lowerIsBetter = true) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const values = metrics.map((m) => m[dataKey]);

    const bestIdx = lowerIsBetter
      ? values.indexOf(Math.min(...values))
      : values.indexOf(Math.max(...values));

    const bgColors = colors.map((c, i) => (i === bestIdx ? c : c + '80'));

    chartInstances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: title,
          data: values,
          backgroundColor: bgColors,
          borderColor: colors,
          borderWidth: 1,
          borderRadius: 6,
        }],
      },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: title,
            color: '#e8e8f0',
            font: { size: 14, weight: '600' },
          },
          legend: { display: false },
        },
        scales: {
          x: {
            ticks: { color: '#9898b0', font: { size: 9 }, maxRotation: 45 },
            grid: { display: false },
          },
          y: {
            ticks: { color: '#9898b0' },
            grid: { color: 'rgba(255,255,255,0.04)' },
          },
        },
      },
    });
  }

  createBarChart('waiting-chart', 'Avg Waiting Time (lower is better)', 'avg_waiting_time', true);
  createBarChart('turnaround-chart', 'Avg Turnaround Time (lower is better)', 'avg_turnaround_time', true);
  createBarChart('response-chart', 'Avg Response Time (lower is better)', 'avg_response_time', true);
  createBarChart('fairness-chart', "Jain's Fairness Index (higher is better)", 'fairness_index', false);
}

// ══════════════════════════════════════════════════════════════════
// GANTT CHART
// ══════════════════════════════════════════════════════════════════
function renderGantt() {
  if (!benchmarkData || !benchmarkData.gantt) return;

  const select = document.getElementById('gantt-algo-select');
  const algos = Object.keys(benchmarkData.gantt);

  if (select.options.length === 0) {
    algos.forEach((algo) => {
      const opt = document.createElement('option');
      opt.value = algo;
      opt.textContent = algo;
      select.appendChild(opt);
    });
  }

  const selectedAlgo = select.value || algos[0];
  const log = benchmarkData.gantt[selectedAlgo];
  if (!log || log.length === 0) return;

  const container = document.getElementById('gantt-container');
  const legendContainer = document.getElementById('gantt-legend');

  const pids = [...new Set(log.filter((e) => e.pid !== null).map((e) => e.pid))];

  const displayLog = log.slice(0, 80); // Limit to first 80 ticks for readability

  let html = '<div class="gantt-row">';
  html += '<span class="gantt-label">CPU</span>';
  html += '<div class="gantt-bar">';
  displayLog.forEach((entry) => {
    const color = entry.pid !== null ? getPidColor(entry.pid) : 'transparent';
    const label = entry.pid !== null ? `P${entry.pid}` : '';
    html += `<div class="gantt-cell${entry.pid === null ? ' idle' : ''}" style="background:${color};" title="t=${entry.time}, PID=${entry.pid ?? 'idle'}">${label}</div>`;
  });
  html += '</div></div>';

  // Time axis
  html += '<div class="gantt-row">';
  html += '<span class="gantt-label">Time</span>';
  html += '<div class="gantt-bar">';
  displayLog.forEach((entry) => {
    const showLabel = entry.time % 5 === 0;
    html += `<div class="gantt-cell idle" style="font-size:0.6rem; color:#9898b0;">${showLabel ? entry.time : ''}</div>`;
  });
  html += '</div></div>';

  container.innerHTML = html;

  legendContainer.innerHTML = pids.slice(0, 15).map((pid) => `
    <span class="gantt-legend-item">
      <span class="gantt-legend-color" style="background:${getPidColor(pid)}"></span>
      P${pid}
    </span>
  `).join('');
}

// ══════════════════════════════════════════════════════════════════
// PROCESS TABLE
// ══════════════════════════════════════════════════════════════════
function renderProcessTable(processes) {
  const tbody = document.getElementById('process-tbody');
  if (!processes) return;

  function errorClass(err) {
    if (err === '—') return '';
    const v = parseFloat(err);
    if (v < 3) return 'error-good';
    if (v < 8) return 'error-ok';
    return 'error-bad';
  }

  tbody.innerHTML = processes.map((p) => {
    const xgbError = p.predicted_burst_xgb != null
      ? Math.abs(p.cpu_burst - p.predicted_burst_xgb).toFixed(1)
      : '—';
    const lstmError = p.predicted_burst_lstm != null
      ? Math.abs(p.cpu_burst - p.predicted_burst_lstm).toFixed(1)
      : '—';

    return `<tr>
      <td>${p.pid}</td>
      <td class="type-${p.process_type}">${p.process_type}</td>
      <td>${p.priority}</td>
      <td>${p.arrival_time}</td>
      <td>${p.cpu_burst}</td>
      <td>${p.predicted_burst_xgb ?? '—'}</td>
      <td>${p.predicted_burst_lstm ?? '—'}</td>
      <td class="${errorClass(xgbError)}">${xgbError}</td>
      <td class="${errorClass(lstmError)}">${lstmError}</td>
    </tr>`;
  }).join('');
}

// ══════════════════════════════════════════════════════════════════
// ML MODEL CARDS
// ══════════════════════════════════════════════════════════════════
function renderModelCards(stats) {
  const container = document.getElementById('model-cards');

  function statRow(label, value) {
    return `<div class="model-stat">
      <span class="stat-label">${label}</span>
      <span class="stat-value">${value}</span>
    </div>`;
  }

  let html = '';

  if (stats.xgboost) {
    html += `<div class="model-card">
      <h3>🌲 XGBoost</h3>
      ${statRow('MAE', stats.xgboost.mae.toFixed(2))}
      ${statRow('RMSE', stats.xgboost.rmse.toFixed(2))}
      ${statRow('R² Score', stats.xgboost.r2.toFixed(3))}
    </div>`;
  }

  if (stats.lstm) {
    html += `<div class="model-card">
      <h3>🔁 LSTM</h3>
      ${statRow('MAE', stats.lstm.mae.toFixed(2))}
      ${statRow('RMSE', stats.lstm.rmse.toFixed(2))}
      ${statRow('R² Score', stats.lstm.r2.toFixed(3))}
    </div>`;
  }

  if (stats.dqn) {
    html += `<div class="model-card">
      <h3>🎮 DQN Agent</h3>
      ${statRow('Avg Reward', stats.dqn.avg_reward.toFixed(1))}
      ${statRow('Episodes', stats.dqn.episodes_trained.toLocaleString())}
    </div>`;
  }

  container.innerHTML = html;

  if (stats.xgboost && stats.xgboost.feature_importances) {
    renderFeatureImportance(stats.xgboost.feature_importances);
  }
}

function renderFeatureImportance(importances) {
  const ctx = document.getElementById('feature-importance-chart').getContext('2d');
  if (chartInstances['feature-importance-chart']) {
    chartInstances['feature-importance-chart'].destroy();
  }

  const labels = Object.keys(importances);
  const values = Object.values(importances);

  chartInstances['feature-importance-chart'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Feature Importance',
        data: values,
        backgroundColor: '#4a9eff80',
        borderColor: '#4a9eff',
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: 'y', // Horizontal bar chart
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: 'XGBoost Feature Importances',
          color: '#e8e8f0',
          font: { size: 14, weight: '600' },
        },
        legend: { display: false },
      },
      scales: {
        x: {
          ticks: { color: '#9898b0' },
          grid: { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
          ticks: { color: '#9898b0', font: { size: 10 } },
          grid: { display: false },
        },
      },
    },
  });
}
/**
 * app.js — Dashboard Main Logic (Redesigned)
 * =============================================
 * - Larger, more readable Chart.js charts
 * - Distinct algorithm colors with high contrast
 * - Multi-objective Pareto analysis (research contribution)
 * - Cleaner metric cards and process table
 */

// ══════════════════════════════════════════════════════════════════
// GLOBAL STATE
// ══════════════════════════════════════════════════════════════════
let benchmarkData = null;
let chartInstances = {};

// Warm earth-tone palette — matches CSS variables
const ALGO_COLORS = {
  'FCFS':               '#c0392b',   // brick red
  'SJF (Known Burst)':  '#2e7d32',   // forest green
  'Priority':           '#d4750a',   // amber
  'Round Robin (q=4)':  '#6d4c41',   // warm brown
  'XGBoost-SJF':        '#1565c0',   // strong blue
  'LSTM-SJF':           '#6a1b9a',   // plum
  'DQN Scheduler':      '#00695c',   // teal
};

function getAlgoColor(name) {
  if (ALGO_COLORS[name]) return ALGO_COLORS[name];
  for (const [key, color] of Object.entries(ALGO_COLORS)) {
    if (name.includes(key.split(' ')[0])) return color;
  }
  return '#6b7280';
}

// PID colors for Gantt
const PID_COLORS = [
  '#3b82f6', '#ef4444', '#22c55e', '#eab308', '#8b5cf6',
  '#f97316', '#ec4899', '#06b6d4', '#a78bfa', '#f472b6',
  '#2dd4bf', '#fbbf24', '#818cf8', '#fb7185', '#38bdf8',
  '#4ade80', '#e879f9', '#22d3ee', '#a3e635', '#14b8a6',
];

function getPidColor(pid) {
  if (pid === null || pid === undefined) return 'transparent';
  return PID_COLORS[(pid - 1) % PID_COLORS.length];
}

// ══════════════════════════════════════════════════════════════════
// MAIN: RUN SIMULATION
// ══════════════════════════════════════════════════════════════════
async function runSimulation() {
  const btn = document.getElementById('run-btn');
  btn.disabled = true;
  btn.textContent = 'Running...';

  try {
    const numProcInput = document.getElementById('num-processes').value;
    const quantumInput = document.getElementById('quantum').value;
    const seedInput = document.getElementById('seed').value;
    const csInput = document.getElementById('context-switch').value;

    const response = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        num_processes: numProcInput ? parseInt(numProcInput, 10) : 50,
        quantum: quantumInput ? parseInt(quantumInput, 10) : 4,
        seed: seedInput ? parseInt(seedInput, 10) : 42,
        context_switch: csInput ? parseInt(csInput, 10) : 0,
      }),
    });

    if (!response.ok) throw new Error('Server responded ' + response.status);
    benchmarkData = await response.json();

    renderMetricCards(benchmarkData.metrics);
    renderComparisonCharts(benchmarkData.metrics);
    renderParetoAnalysis(benchmarkData.metrics);
    document.getElementById('gantt-algo-select').innerHTML = '';
    renderGantt();
    renderProcessTable(benchmarkData.processes);
    setupAnimationSelects();

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
    btn.textContent = '\u25B6 Run Simulation';
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

  container.innerHTML = [
    { label: 'Best Avg Waiting Time', value: bestWait.avg_waiting_time.toFixed(1), algo: bestWait.algorithm, cls: 'blue' },
    { label: 'Best Avg Turnaround', value: bestTurn.avg_turnaround_time.toFixed(1), algo: bestTurn.algorithm, cls: 'green' },
    { label: 'Best Response Time', value: bestResp.avg_response_time.toFixed(1), algo: bestResp.algorithm, cls: 'purple' },
    { label: 'Most Fair Scheduler', value: bestFair.fairness_index.toFixed(3), algo: bestFair.algorithm, cls: 'orange' },
  ].map(c => `
    <div class="metric-card ${c.cls}">
      <div class="metric-label">${c.label}</div>
      <div class="metric-value">${c.value}</div>
      <div class="metric-algo">${c.algo}</div>
    </div>
  `).join('');
}

// ══════════════════════════════════════════════════════════════════
// COMPARISON BAR CHARTS (larger, clearer)
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

    // Best bar is fully opaque, others are semi-transparent
    const bgColors = colors.map((c, i) => (i === bestIdx ? c : c + '55'));
    const borders = colors.map((c, i) => (i === bestIdx ? c : c + 'aa'));

    chartInstances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: title,
          data: values,
          backgroundColor: bgColors,
          borderColor: borders,
          borderWidth: 2,
          borderRadius: 6,
          barPercentage: 0.7,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 1.6,
        plugins: {
          title: {
            display: true,
            text: title,
            color: '#1a1916',
            font: { size: 13, weight: '700' },
            padding: { bottom: 15 },
          },
          legend: { display: false },
          tooltip: {
            backgroundColor: '#fff',
            titleColor: '#1a1916',
            bodyColor: '#6b6861',
            titleFont: { size: 12, weight: '700' },
            bodyFont: { size: 12 },
            borderColor: '#e4e2db',
            borderWidth: 1,
            padding: 10,
            cornerRadius: 6,
          },
        },
        scales: {
          x: {
            ticks: {
              color: '#6b6861',
              font: { size: 10, weight: '600' },
              maxRotation: 35,
              minRotation: 20,
            },
            grid: { display: false },
            border: { color: '#e4e2db' },
          },
          y: {
            ticks: {
              color: '#6b6861',
              font: { size: 11 },
            },
            grid: { color: 'rgba(0,0,0,0.05)' },
            border: { color: '#e4e2db' },
          },
        },
      },
    });
  }

  createBarChart('waiting-chart', 'Avg Waiting Time (lower = better)', 'avg_waiting_time', true);
  createBarChart('turnaround-chart', 'Avg Turnaround Time (lower = better)', 'avg_turnaround_time', true);
  createBarChart('response-chart', 'Avg Response Time (lower = better)', 'avg_response_time', true);
  createBarChart('fairness-chart', "Jain's Fairness Index (higher = better)", 'fairness_index', false);
}

// ══════════════════════════════════════════════════════════════════
// MULTI-OBJECTIVE PARETO ANALYSIS (Research Gap Feature)
// ══════════════════════════════════════════════════════════════════
function renderParetoAnalysis(metrics) {
  // --- Chart 1: Waiting Time vs Fairness (scatter) ---
  const ctx1 = document.getElementById('pareto-chart-1').getContext('2d');
  if (chartInstances['pareto-chart-1']) chartInstances['pareto-chart-1'].destroy();

  const scatterData1 = metrics.map(m => ({
    x: m.avg_waiting_time,
    y: m.fairness_index,
    label: m.algorithm,
  }));

  chartInstances['pareto-chart-1'] = new Chart(ctx1, {
    type: 'scatter',
    data: {
      datasets: metrics.map((m, i) => ({
        label: m.algorithm,
        data: [{ x: m.avg_waiting_time, y: m.fairness_index }],
        backgroundColor: getAlgoColor(m.algorithm) + 'cc',
        borderColor: getAlgoColor(m.algorithm),
        pointRadius: 10,
        pointHoverRadius: 14,
        pointStyle: 'circle',
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1.3,
      plugins: {
        title: {
          display: true,
          text: 'Trade-off: Waiting Time vs Fairness',
          color: '#1a1916',
          font: { size: 13, weight: '700' },
          padding: { bottom: 10 },
        },
        legend: {
          display: true,
          position: 'bottom',
          labels: {
            color: '#6b6861',
            font: { size: 10 },
            usePointStyle: true,
            padding: 12,
          },
        },
        tooltip: {
          backgroundColor: '#fff',
          titleColor: '#1a1916',
          bodyColor: '#6b6861',
          borderColor: '#e4e2db',
          borderWidth: 1,
          padding: 10,
          cornerRadius: 6,
          callbacks: {
            label: function(ctx) {
              const m = metrics[ctx.datasetIndex];
              return m.algorithm + ': Wait=' + m.avg_waiting_time.toFixed(1) + ', Fair=' + m.fairness_index.toFixed(3);
            }
          }
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Avg Waiting Time (lower = better)', color: '#6b6861', font: { size: 11 } },
          ticks: { color: '#6b6861' },
          grid: { color: 'rgba(0,0,0,0.05)' },
          border: { color: '#e4e2db' },
        },
        y: {
          title: { display: true, text: 'Fairness Index (higher = better)', color: '#6b6861', font: { size: 11 } },
          ticks: { color: '#6b6861' },
          grid: { color: 'rgba(0,0,0,0.05)' },
          border: { color: '#e4e2db' },
        },
      },
    },
  });

  // --- Chart 2: Waiting Time vs Response Time ---
  const ctx2 = document.getElementById('pareto-chart-2').getContext('2d');
  if (chartInstances['pareto-chart-2']) chartInstances['pareto-chart-2'].destroy();

  chartInstances['pareto-chart-2'] = new Chart(ctx2, {
    type: 'scatter',
    data: {
      datasets: metrics.map((m, i) => ({
        label: m.algorithm,
        data: [{ x: m.avg_waiting_time, y: m.avg_response_time }],
        backgroundColor: getAlgoColor(m.algorithm) + 'cc',
        borderColor: getAlgoColor(m.algorithm),
        pointRadius: 10,
        pointHoverRadius: 14,
        pointStyle: 'circle',
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1.3,
      plugins: {
        title: {
          display: true,
          text: 'Trade-off: Waiting Time vs Response Time',
          color: '#1a1916',
          font: { size: 13, weight: '700' },
          padding: { bottom: 10 },
        },
        legend: {
          display: true,
          position: 'bottom',
          labels: {
            color: '#6b6861',
            font: { size: 10 },
            usePointStyle: true,
            padding: 12,
          },
        },
        tooltip: {
          backgroundColor: '#fff',
          titleColor: '#1a1916',
          bodyColor: '#6b6861',
          borderColor: '#e4e2db',
          borderWidth: 1,
          padding: 10,
          cornerRadius: 6,
          callbacks: {
            label: function(ctx) {
              const m = metrics[ctx.datasetIndex];
              return m.algorithm + ': Wait=' + m.avg_waiting_time.toFixed(1) + ', Resp=' + m.avg_response_time.toFixed(1);
            }
          }
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Avg Waiting Time (lower = better)', color: '#6b6861', font: { size: 11 } },
          ticks: { color: '#6b6861' },
          grid: { color: 'rgba(0,0,0,0.05)' },
          border: { color: '#e4e2db' },
        },
        y: {
          title: { display: true, text: 'Avg Response Time (lower = better)', color: '#6b6861', font: { size: 11 } },
          ticks: { color: '#6b6861' },
          grid: { color: 'rgba(0,0,0,0.05)' },
          border: { color: '#e4e2db' },
        },
      },
    },
  });

  // --- Composite Pareto Scores ---
  // Normalize each metric to [0,1], then compute weighted score
  // Weights: waiting_time=40%, fairness=35%, response_time=25%
  const waitValues = metrics.map(m => m.avg_waiting_time);
  const fairValues = metrics.map(m => m.fairness_index);
  const respValues = metrics.map(m => m.avg_response_time);

  const minWait = Math.min(...waitValues), maxWait = Math.max(...waitValues);
  const minFair = Math.min(...fairValues), maxFair = Math.max(...fairValues);
  const minResp = Math.min(...respValues), maxResp = Math.max(...respValues);

  const scores = metrics.map((m, i) => {
    // Normalize: for waiting/response, lower is better so invert
    const normWait = maxWait === minWait ? 1 : 1 - (m.avg_waiting_time - minWait) / (maxWait - minWait);
    const normFair = maxFair === minFair ? 1 : (m.fairness_index - minFair) / (maxFair - minFair);
    const normResp = maxResp === minResp ? 1 : 1 - (m.avg_response_time - minResp) / (maxResp - minResp);

    const composite = (0.40 * normWait + 0.35 * normFair + 0.25 * normResp) * 100;
    return { algorithm: m.algorithm, score: composite };
  });

  // Sort by score descending
  scores.sort((a, b) => b.score - a.score);

  const container = document.getElementById('pareto-scores');
  container.innerHTML = scores.map((s, i) => {
    const rankClass = i === 0 ? 'rank-1' : (i === 1 ? 'rank-2' : (i === 2 ? 'rank-3' : ''));
    const rankLabel = i === 0 ? '#1 BEST OVERALL' : (i === 1 ? '#2' : (i === 2 ? '#3' : '#' + (i + 1)));
    return `
      <div class="pareto-score-card ${rankClass}">
        <div class="ps-algo">${s.algorithm}</div>
        <div class="ps-score">${s.score.toFixed(1)}</div>
        <div class="ps-rank">${rankLabel}</div>
      </div>
    `;
  }).join('');
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
  const pids = [...new Set(log.filter((e) => e.pid !== null && e.pid !== 'CS').map((e) => e.pid))];
  const displayLog = log.slice(0, 80);

  let html = '<div class="gantt-row">';
  html += '<span class="gantt-label">CPU</span>';
  html += '<div class="gantt-bar">';
  displayLog.forEach((entry) => {
    let color, label, cls;
    if (entry.pid === 'CS') {
      color = '#78716c';
      label = 'CS';
      cls = ' cs';
    } else if (entry.pid !== null) {
      color = getPidColor(entry.pid);
      label = 'P' + entry.pid;
      cls = '';
    } else {
      color = 'transparent';
      label = '';
      cls = ' idle';
    }
    html += '<div class="gantt-cell' + cls + '" style="background:' + color + ';" title="t=' + entry.time + ', ' + (entry.pid === 'CS' ? 'Context Switch' : 'PID=' + (entry.pid || 'idle')) + '">' + label + '</div>';
  });
  html += '</div></div>';

  html += '<div class="gantt-row">';
  html += '<span class="gantt-label">Time</span>';
  html += '<div class="gantt-bar">';
  displayLog.forEach((entry) => {
    const showLabel = entry.time % 5 === 0;
    html += '<div class="gantt-cell idle" style="font-size:0.6rem; color:#8888a8;">' + (showLabel ? entry.time : '') + '</div>';
  });
  html += '</div></div>';

  container.innerHTML = html;

  legendContainer.innerHTML = pids.slice(0, 15).map((pid) =>
    '<span class="gantt-legend-item"><span class="gantt-legend-color" style="background:' + getPidColor(pid) + '"></span>P' + pid + '</span>'
  ).join('');
}

// ══════════════════════════════════════════════════════════════════
// PROCESS TABLE
// ══════════════════════════════════════════════════════════════════
let allTableProcesses = [];
let isTableExpanded = false;
const DEFAULT_TABLE_PAGE_SIZE = 20;

function toggleProcessTableExpansion() {
  isTableExpanded = !isTableExpanded;
  updateProcessTableView();
}

function updateProcessTableView() {
  const tbody = document.getElementById('process-tbody');
  const countBadge = document.getElementById('process-count-badge');
  const pagControls = document.getElementById('table-pagination-controls');
  const showMoreBtn = document.getElementById('show-more-btn');
  const remainingCountSpan = document.getElementById('remaining-count');

  if (!tbody || !allTableProcesses) return;

  const total = allTableProcesses.length;
  const displayList = isTableExpanded ? allTableProcesses : allTableProcesses.slice(0, DEFAULT_TABLE_PAGE_SIZE);

  if (countBadge) {
    countBadge.textContent = isTableExpanded
      ? `Showing all ${total} processes`
      : `Showing ${Math.min(DEFAULT_TABLE_PAGE_SIZE, total)} of ${total} processes`;
  }

  function errorClass(err) {
    if (err === '\u2014') return '';
    const v = parseFloat(err);
    if (v < 3) return 'error-good';
    if (v < 8) return 'error-ok';
    return 'error-bad';
  }

  tbody.innerHTML = displayList.map((p) => {
    const xgbPred = p.predicted_burst_xgb != null ? p.predicted_burst_xgb : '\u2014';
    const lstmPred = p.predicted_burst_lstm != null ? p.predicted_burst_lstm : '\u2014';
    const xgbError = p.predicted_burst_xgb != null
      ? Math.abs(p.cpu_burst - p.predicted_burst_xgb).toFixed(1)
      : '\u2014';
    const lstmError = p.predicted_burst_lstm != null
      ? Math.abs(p.cpu_burst - p.predicted_burst_lstm).toFixed(1)
      : '\u2014';

    return '<tr>' +
      '<td>' + p.pid + '</td>' +
      '<td class="type-' + p.process_type + '">' + p.process_type + '</td>' +
      '<td>' + p.priority + '</td>' +
      '<td>' + p.arrival_time + '</td>' +
      '<td>' + p.cpu_burst + '</td>' +
      '<td>' + xgbPred + '</td>' +
      '<td>' + lstmPred + '</td>' +
      '<td class="' + errorClass(xgbError) + '">' + xgbError + '</td>' +
      '<td class="' + errorClass(lstmError) + '">' + lstmError + '</td>' +
      '</tr>';
  }).join('');

  if (pagControls && showMoreBtn) {
    if (total <= DEFAULT_TABLE_PAGE_SIZE) {
      pagControls.style.display = 'none';
    } else {
      pagControls.style.display = 'flex';
      if (isTableExpanded) {
        showMoreBtn.textContent = 'Show Less (First 20)';
      } else {
        const remaining = total - DEFAULT_TABLE_PAGE_SIZE;
        showMoreBtn.textContent = `Show All (${remaining} more)`;
      }
    }
  }
}

function renderProcessTable(processes) {
  allTableProcesses = processes || [];
  isTableExpanded = false;
  updateProcessTableView();
}

// ══════════════════════════════════════════════════════════════════
// ML MODEL CARDS
// ══════════════════════════════════════════════════════════════════
function renderModelCards(stats) {
  const container = document.getElementById('model-cards');

  function statRow(label, value) {
    return '<div class="model-stat"><span class="stat-label">' + label + '</span><span class="stat-value">' + value + '</span></div>';
  }

  let html = '';

  if (stats.xgboost) {
    html += '<div class="model-card"><h3>\uD83C\uDF32 XGBoost</h3>' +
      statRow('MAE', stats.xgboost.mae.toFixed(2)) +
      statRow('RMSE', stats.xgboost.rmse.toFixed(2)) +
      statRow('R\u00B2 Score', stats.xgboost.r2.toFixed(3)) +
      '</div>';
  }

  if (stats.lstm) {
    html += '<div class="model-card"><h3>\uD83D\uDD01 LSTM</h3>' +
      statRow('MAE', stats.lstm.mae.toFixed(2)) +
      statRow('RMSE', stats.lstm.rmse.toFixed(2)) +
      statRow('R\u00B2 Score', stats.lstm.r2.toFixed(3)) +
      '</div>';
  }

  if (stats.dqn) {
    html += '<div class="model-card"><h3>\uD83C\uDFAE DQN Agent</h3>' +
      statRow('Avg Reward', stats.dqn.avg_reward.toFixed(1)) +
      statRow('Episodes', stats.dqn.episodes_trained.toLocaleString()) +
      '</div>';
  }

  container.innerHTML = html;
}
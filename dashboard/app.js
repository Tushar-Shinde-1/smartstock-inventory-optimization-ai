/* ========================================
   SMARTSTOCK DASHBOARD — APP.JS
   ======================================== */

let DATA = null;
let INSIGHTS = null;

// ---- NAVIGATION ----
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', e => {
        e.preventDefault();
        const target = link.dataset.page;
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.getElementById('page-' + target).classList.add('active');
    });
});

// ---- LOAD BOTH DATA FILES ----
Promise.all([
    fetch('model_results.json').then(r => r.json()),
    fetch('smartstock_insights.json').then(r => r.json())
]).then(([modelData, insightsData]) => {
    DATA = modelData;
    INSIGHTS = insightsData;
    renderOverview();
    renderComparison();
    renderGeneralization();
    renderFeatures();
    renderPredictions();
    renderForecast();
    renderInventory();
    renderHealth();
    renderRecommendations();
}).catch(err => console.error('Load error:', err));

// ---- CHART DEFAULTS ----
Chart.defaults.color = '#9898b0';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = 'Inter';
Chart.defaults.font.size = 12;

const COLORS = {
    blue: '#6366f1', green: '#22c55e', red: '#ef4444',
    amber: '#f59e0b', cyan: '#06b6d4', purple: '#a855f7',
    blueA: 'rgba(99,102,241,0.5)', greenA: 'rgba(34,197,94,0.5)', redA: 'rgba(239,68,68,0.5)',
};

// ================================================
// MODEL ANALYTICS PAGES (unchanged logic)
// ================================================
function renderOverview() {
    const best = DATA.best_model;
    const m = DATA.model_results[best].test;
    document.getElementById('best-model-name').textContent = best;
    document.getElementById('hero-metrics').innerHTML = `
        <div class="hero-metric"><span class="label">R² Score</span><span class="value">${m.R2}</span></div>
        <div class="hero-metric"><span class="label">MAE</span><span class="value">${m.MAE}</span></div>
        <div class="hero-metric"><span class="label">RMSE</span><span class="value">${m.RMSE}</span></div>
        <div class="hero-metric"><span class="label">MAPE</span><span class="value">${m.MAPE}%</span></div>`;
    const ds = DATA.dataset;
    document.getElementById('dataset-stats').innerHTML = `
        <div class="stat-card"><span class="stat-icon">📋</span><span class="stat-value">${ds.total_records.toLocaleString()}</span><span class="stat-label">Total Records</span></div>
        <div class="stat-card"><span class="stat-icon">🧩</span><span class="stat-value">${ds.num_features}</span><span class="stat-label">Input Features</span></div>
        <div class="stat-card"><span class="stat-icon">📅</span><span class="stat-value">${ds.train_records.toLocaleString()}</span><span class="stat-label">Training Records</span></div>
        <div class="stat-card"><span class="stat-icon">🧪</span><span class="stat-value">${ds.test_records.toLocaleString()}</span><span class="stat-label">Test Records</span></div>`;
    const models = ['Linear Regression', 'Random Forest (Tuned)', 'XGBoost (Tuned)'];
    let t = '<table><thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th><th>MAPE</th></tr></thead><tbody>';
    models.forEach(n => {
        const r = DATA.model_results[n]?.test;
        if (!r) return;
        const cls = n === best ? ' class="cell-best"' : '';
        t += `<tr><td${cls}>${n === best ? '🏆 ' : ''}${n}</td><td${cls}>${r.MAE}</td><td${cls}>${r.RMSE}</td><td${cls}>${r.R2}</td><td${cls}>${r.MAPE}%</td></tr>`;
    });
    document.getElementById('quick-comparison-table').innerHTML = t + '</tbody></table>';
}

function renderComparison() {
    const models = ['Linear Regression', 'Random Forest', 'XGBoost', 'Random Forest (Tuned)', 'XGBoost (Tuned)'];
    const labels = ['LR', 'RF', 'XGB', 'RF Tuned', 'XGB Tuned'];
    const c = [COLORS.blue, COLORS.green, COLORS.red, COLORS.cyan, COLORS.amber];
    ['MAE', 'RMSE', 'R2', 'MAPE'].forEach(metric => {
        const vals = models.map(m => DATA.model_results[m]?.test?.[metric] || 0);
        new Chart(document.getElementById('chart-' + metric.toLowerCase()), {
            type: 'bar', data: { labels, datasets: [{ data: vals, backgroundColor: c.map(x => x + '99'), borderColor: c, borderWidth: 1.5, borderRadius: 6 }] },
            options: { responsive: true, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: metric !== 'R2', grid: { color: 'rgba(255,255,255,0.04)' } }, x: { grid: { display: false } } } }
        });
    });
    const rfP = DATA.model_results['Random Forest (Tuned)']?.best_params || {};
    const xgbP = DATA.model_results['XGBoost (Tuned)']?.best_params || {};
    document.getElementById('rf-params').innerHTML = renderParams(rfP);
    document.getElementById('xgb-params').innerHTML = renderParams(xgbP);
}

function renderParams(p) {
    let h = '<div class="param-grid">';
    for (const [k, v] of Object.entries(p)) h += `<div class="param-tag"><span class="param-name">${k}:</span><span class="param-val">${v}</span></div>`;
    return h + '</div>';
}

function renderGeneralization() {
    const cv = DATA.cv_results; const names = Object.keys(cv); const short = names.map(n => n.replace(' (Tuned)', '')); const c3 = [COLORS.blue, COLORS.green, COLORS.red];
    new Chart(document.getElementById('chart-cv-mae'), { type: 'bar', data: { labels: short, datasets: [{ data: names.map(n => cv[n].mae_mean), backgroundColor: c3.map(c => c + '80'), borderColor: c3, borderWidth: 1.5, borderRadius: 6 }] }, options: { responsive: true, plugins: { legend: { display: false }, title: { display: true, text: 'CV MAE (lower is better)', font: { size: 13, weight: '600' } } }, scales: { y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } }, x: { grid: { display: false } } } } });
    new Chart(document.getElementById('chart-cv-r2'), { type: 'bar', data: { labels: short, datasets: [{ data: names.map(n => cv[n].r2_mean), backgroundColor: c3.map(c => c + '80'), borderColor: c3, borderWidth: 1.5, borderRadius: 6 }] }, options: { responsive: true, plugins: { legend: { display: false }, title: { display: true, text: 'CV R² (higher is better)', font: { size: 13, weight: '600' } } }, scales: { y: { grid: { color: 'rgba(255,255,255,0.04)' } }, x: { grid: { display: false } } } } });

    const ofM = ['Linear Regression', 'Random Forest (Tuned)', 'XGBoost (Tuned)'];
    new Chart(document.getElementById('chart-overfit'), { type: 'bar', data: { labels: ofM.map(n => n.replace(' (Tuned)', '')), datasets: [{ label: 'Train R²', data: ofM.map(m => DATA.model_results[m]?.train?.R2 || 0), backgroundColor: COLORS.blueA, borderColor: COLORS.blue, borderWidth: 1.5, borderRadius: 6 }, { label: 'Test R²', data: ofM.map(m => DATA.model_results[m]?.test?.R2 || 0), backgroundColor: COLORS.redA, borderColor: COLORS.red, borderWidth: 1.5, borderRadius: 6 }] }, options: { responsive: true, plugins: { legend: { position: 'top' } }, scales: { y: { beginAtZero: true, max: 1, grid: { color: 'rgba(255,255,255,0.04)' } }, x: { grid: { display: false } } } } });

    [{ key: 'Linear Regression', id: 'chart-resid-lr', color: COLORS.blue }, { key: 'Random Forest (Tuned)', id: 'chart-resid-rf', color: COLORS.green }, { key: 'XGBoost (Tuned)', id: 'chart-resid-xgb', color: COLORS.red }].forEach(({ key, id, color }) => {
        const rd = DATA.residuals[key]; const lbl = rd.edges.slice(0, -1).map((e, i) => ((e + rd.edges[i + 1]) / 2).toFixed(1));
        new Chart(document.getElementById(id), { type: 'bar', data: { labels: lbl, datasets: [{ label: key.replace(' (Tuned)', ''), data: rd.counts, backgroundColor: color + '66', borderColor: color, borderWidth: 1, borderRadius: 3 }] }, options: { responsive: true, plugins: { legend: { display: false }, title: { display: true, text: key.replace(' (Tuned)', ''), font: { size: 13, weight: '600' } } }, scales: { y: { grid: { color: 'rgba(255,255,255,0.04)' } }, x: { grid: { display: false }, ticks: { maxTicksLimit: 8, font: { size: 11 } } } } } });
    });
}

function renderFeatures() {
    const rfFI = DATA.model_results['Random Forest (Tuned)']?.feature_importance || {};
    const xgbFI = DATA.model_results['XGBoost (Tuned)']?.feature_importance || {};
    renderFIChart('chart-fi-rf', rfFI, COLORS.green);
    renderFIChart('chart-fi-xgb', xgbFI, COLORS.red);
    const rfS = Object.entries(rfFI).sort((a, b) => b[1] - a[1]);
    const xgbS = Object.entries(xgbFI).sort((a, b) => b[1] - a[1]);
    let t = '<table><thead><tr><th>#</th><th>RF Feature</th><th>Score</th><th>XGB Feature</th><th>Score</th></tr></thead><tbody>';
    for (let i = 0; i < Math.min(rfS.length, 15); i++) t += `<tr><td>${i + 1}</td><td>${rfS[i][0]}</td><td>${rfS[i][1]}</td><td>${xgbS[i][0]}</td><td>${xgbS[i][1]}</td></tr>`;
    document.getElementById('feature-table').innerHTML = t + '</tbody></table>';
}

function renderFIChart(id, fi, color) {
    const s = Object.entries(fi).sort((a, b) => a[1] - b[1]).slice(-12);
    new Chart(document.getElementById(id), { type: 'bar', data: { labels: s.map(e => e[0]), datasets: [{ data: s.map(e => e[1]), backgroundColor: color + '66', borderColor: color, borderWidth: 1.5, borderRadius: 4 }] }, options: { indexAxis: 'y', responsive: true, plugins: { legend: { display: false } }, scales: { x: { grid: { color: 'rgba(255,255,255,0.04)' } }, y: { grid: { display: false }, ticks: { font: { size: 10.5 } } } } } });
}

function renderPredictions() {
    const pred = DATA.predictions; const actual = pred.actual;
    [{ key: 'Linear Regression', id: 'chart-scatter-lr', color: COLORS.blue }, { key: 'Random Forest (Tuned)', id: 'chart-scatter-rf', color: COLORS.green }, { key: 'XGBoost (Tuned)', id: 'chart-scatter-xgb', color: COLORS.red }].forEach(({ key, id, color }) => {
        const pts = actual.map((a, i) => ({ x: a, y: pred[key][i] }));
        const mn = Math.min(...actual, ...pred[key]) - 2, mx = Math.max(...actual, ...pred[key]) + 2;
        new Chart(document.getElementById(id), { type: 'scatter', data: { datasets: [{ label: 'Predictions', data: pts, backgroundColor: color + '55', borderColor: color, pointRadius: 3.5 }, { label: 'Perfect', data: [{ x: mn, y: mn }, { x: mx, y: mx }], type: 'line', borderColor: 'rgba(255,255,255,0.3)', borderDash: [6, 3], borderWidth: 1.5, pointRadius: 0 }] }, options: { responsive: true, aspectRatio: 1, plugins: { legend: { display: false } }, scales: { x: { title: { display: true, text: 'Actual' }, grid: { color: 'rgba(255,255,255,0.04)' } }, y: { title: { display: true, text: 'Predicted' }, grid: { color: 'rgba(255,255,255,0.04)' } } } } });
    });
    const lrP = pred['Linear Regression'], rfP = pred['Random Forest (Tuned)'], xP = pred['XGBoost (Tuned)'];
    let t = '<table><thead><tr><th>#</th><th>Actual</th><th>LR</th><th>RF</th><th>XGB</th><th>Best Err</th></tr></thead><tbody>';
    for (let i = 0; i < Math.min(20, actual.length); i++) {
        const a = actual[i]; const err = Math.min(Math.abs(a - lrP[i]), Math.abs(a - rfP[i]), Math.abs(a - xP[i])).toFixed(1);
        t += `<tr><td>${i + 1}</td><td><strong>${a}</strong></td><td>${lrP[i]}</td><td>${rfP[i]}</td><td>${xP[i]}</td><td class="cell-best">${err}</td></tr>`;
    }
    document.getElementById('pred-table').innerHTML = t + '</tbody></table>';
}

// ================================================
// SMARTSTOCK INSIGHT PAGES
// ================================================

// ---- DEMAND FORECAST ----
function renderForecast() {
    const pf = INSIGHTS.product_forecast;
    let cards = '';
    pf.forEach(p => {
        cards += `<div class="stat-card" data-color="blue"><span class="stat-icon">👟</span><span class="stat-value">${p.predicted_avg_demand}</span><span class="stat-label">${p.product} · Predicted Avg</span></div>`;
    });
    document.getElementById('forecast-summary-cards').innerHTML = cards;

    // Product demand chart
    new Chart(document.getElementById('chart-product-demand'), {
        type: 'bar',
        data: {
            labels: pf.map(p => p.product),
            datasets: [
                { label: 'Historical Avg', data: pf.map(p => p.avg_daily_demand), backgroundColor: COLORS.blue + '88', borderColor: COLORS.blue, borderWidth: 1.5, borderRadius: 6 },
                { label: 'Predicted Avg', data: pf.map(p => p.predicted_avg_demand), backgroundColor: COLORS.green + '88', borderColor: COLORS.green, borderWidth: 1.5, borderRadius: 6 }
            ]
        },
        options: {
            responsive: true,
            plugins: { legend: { position: 'top' } },
            scales: { y: { beginAtZero: true, title: { display: true, text: 'Quantity Sold' }, grid: { color: 'rgba(255,255,255,0.04)' } }, x: { grid: { display: false } } }
        }
    });

    // Detail table
    const pbf = INSIGHTS.product_brand_forecast;
    let t = '<table><thead><tr><th>Product</th><th>Brand</th><th>Avg Daily</th><th>Predicted</th><th>Std Dev</th><th>Avg Price</th><th>Stock</th></tr></thead><tbody>';
    pbf.forEach(r => {
        t += `<tr><td>${r.product}</td><td>${r.brand}</td><td>${r.avg_daily_demand}</td><td class="cell-best">${r.predicted_demand}</td><td>${r.demand_std}</td><td>₹${r.avg_price}</td><td>${r.current_stock}</td></tr>`;
    });
    document.getElementById('forecast-detail-table').innerHTML = t + '</tbody></table>';
}

// ---- INVENTORY OPTIMIZATION ----
function renderInventory() {
    const inv = INSIGHTS.inventory_optimization;
    const counts = { Critical: 0, Reorder: 0, Healthy: 0, Overstock: 0 };
    inv.forEach(i => counts[i.stock_status] = (counts[i.stock_status] || 0) + 1);

    document.getElementById('inventory-summary-cards').innerHTML = `
        <div class="stat-card" data-color="red"><span class="stat-icon">🚨</span><span class="stat-value">${counts.Critical || 0}</span><span class="stat-label">Critical Items</span></div>
        <div class="stat-card" data-color="amber"><span class="stat-icon">⚠️</span><span class="stat-value">${counts.Reorder || 0}</span><span class="stat-label">Need Reorder</span></div>
        <div class="stat-card" data-color="green"><span class="stat-icon">✅</span><span class="stat-value">${counts.Healthy || 0}</span><span class="stat-label">Healthy Stock</span></div>
        <div class="stat-card" data-color="purple"><span class="stat-icon">📦</span><span class="stat-value">${counts.Overstock || 0}</span><span class="stat-label">Overstock</span></div>`;

    // Table
    let t = '<table><thead><tr><th>Product</th><th>Brand</th><th>Avg Demand</th><th>Safety Stock</th><th>Reorder Point</th><th>EOQ</th><th>Current</th><th>Status</th></tr></thead><tbody>';
    inv.forEach(r => {
        const cls = r.stock_status.toLowerCase();
        t += `<tr><td>${r.product}</td><td>${r.brand}</td><td>${r.avg_daily_demand}</td><td>${r.safety_stock}</td><td>${r.reorder_point}</td><td>${r.eoq}</td><td>${r.current_stock}</td><td><span class="status-badge ${cls}">${r.stock_status}</span></td></tr>`;
    });
    document.getElementById('inventory-table').innerHTML = t + '</tbody></table>';

    // Chart — Stock vs ROP (top 15 items)
    const top = inv.slice(0, 15);
    new Chart(document.getElementById('chart-stock-vs-rop'), {
        type: 'bar',
        data: {
            labels: top.map(i => `${i.product.substr(0, 4)}-${i.brand.substr(0, 4)}`),
            datasets: [
                { label: 'Current Stock', data: top.map(i => i.current_stock), backgroundColor: COLORS.blue + '88', borderColor: COLORS.blue, borderWidth: 1, borderRadius: 4 },
                { label: 'Reorder Point', data: top.map(i => i.reorder_point), backgroundColor: COLORS.amber + '88', borderColor: COLORS.amber, borderWidth: 1, borderRadius: 4 },
                { label: 'Safety Stock', data: top.map(i => i.safety_stock), backgroundColor: COLORS.red + '88', borderColor: COLORS.red, borderWidth: 1, borderRadius: 4 }
            ]
        },
        options: {
            responsive: true,
            plugins: { legend: { position: 'top' } },
            scales: { y: { beginAtZero: true, title: { display: true, text: 'Units' }, grid: { color: 'rgba(255,255,255,0.04)' } }, x: { grid: { display: false }, ticks: { font: { size: 9 } } } }
        }
    });
}

// ---- STOCK HEALTH ----
function renderHealth() {
    const hs = INSIGHTS.health_summary;
    const healthColors = { 'Fast-Moving': 'cyan', 'Normal': 'blue', 'Slow-Moving': 'amber', 'Deadstock': 'red' };
    const healthIcons = { 'Fast-Moving': '⚡', 'Normal': '✅', 'Slow-Moving': '🐌', 'Deadstock': '💀' };
    let cards = '';
    for (const [h, count] of Object.entries(hs)) {
        cards += `<div class="stat-card" data-color="${healthColors[h] || 'blue'}"><span class="stat-icon">${healthIcons[h] || '📦'}</span><span class="stat-value">${count}</span><span class="stat-label">${h}</span></div>`;
    }
    document.getElementById('health-summary-cards').innerHTML = cards;

    // Pie chart
    const pieLabels = Object.keys(hs);
    const pieColors = ['#06b6d4', '#6366f1', '#f59e0b', '#ef4444'];
    new Chart(document.getElementById('chart-health-pie'), {
        type: 'doughnut',
        data: { labels: pieLabels, datasets: [{ data: Object.values(hs), backgroundColor: pieColors.slice(0, pieLabels.length), borderWidth: 0 }] },
        options: { responsive: true, plugins: { legend: { position: 'right' }, title: { display: true, text: 'Stock Health Distribution', font: { size: 13, weight: '600' } } } }
    });

    // Turnover ratio chart (top 15)
    const sh = INSIGHTS.stock_health.sort((a, b) => b.turnover_ratio - a.turnover_ratio).slice(0, 15);
    const turnoverColors = sh.map(i => i.health === 'Fast-Moving' ? COLORS.cyan : i.health === 'Deadstock' ? COLORS.red : i.health === 'Slow-Moving' ? COLORS.amber : COLORS.blue);
    new Chart(document.getElementById('chart-turnover'), {
        type: 'bar',
        data: { labels: sh.map(i => `${i.product.substr(0, 4)}-${i.brand.substr(0, 4)}`), datasets: [{ label: 'Turnover Ratio', data: sh.map(i => i.turnover_ratio), backgroundColor: turnoverColors.map(c => c + '88'), borderColor: turnoverColors, borderWidth: 1, borderRadius: 4 }] },
        options: { responsive: true, plugins: { legend: { display: false }, title: { display: true, text: 'Inventory Turnover Ratio (higher = faster)', font: { size: 13, weight: '600' } } }, scales: { y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } }, x: { grid: { display: false }, ticks: { font: { size: 11 }, maxRotation: 45, minRotation: 45 } } } }
    });

    // Health table
    let t = '<table><thead><tr><th>Product</th><th>Brand</th><th>Total Sold</th><th>Avg Daily</th><th>Last 30d</th><th>Stock</th><th>Turnover</th><th>Days Supply</th><th>Health</th><th>Action</th></tr></thead><tbody>';
    INSIGHTS.stock_health.forEach(r => {
        const cls = r.health === 'Fast-Moving' ? 'fast' : r.health === 'Deadstock' ? 'deadstock' : r.health === 'Slow-Moving' ? 'slow' : 'normal';
        t += `<tr><td>${r.product}</td><td>${r.brand}</td><td>${r.total_sold}</td><td>${r.avg_daily_sales}</td><td>${r.sold_last_30d}</td><td>${r.avg_stock}</td><td>${r.turnover_ratio}</td><td>${r.days_of_supply}</td><td><span class="status-badge ${cls}">${r.health}</span></td><td style="font-size:11px;max-width:200px">${r.action}</td></tr>`;
    });
    document.getElementById('health-table').innerHTML = t + '</tbody></table>';
}

// ---- RECOMMENDATIONS ----
function renderRecommendations() {
    const recs = INSIGHTS.recommendations;
    const as = INSIGHTS.alert_summary;

    const alertIcons = { 'Critical — Below Safety Stock': '🚨', 'Reorder Now': '⚠️', 'Healthy': '✅', 'Overstock Warning': '📦', 'Deadstock': '💀' };
    const alertColors = { 'Critical — Below Safety Stock': '#ef4444', 'Reorder Now': '#f59e0b', 'Healthy': '#22c55e', 'Overstock Warning': '#a855f7', 'Deadstock': '#ef4444' };

    // Summary cards
    const urgent = recs.filter(r => r.priority === 'Urgent').length;
    const high = recs.filter(r => r.priority === 'High').length;
    const totalOrder = recs.reduce((s, r) => s + r.order_qty, 0);
    document.getElementById('rec-summary-cards').innerHTML = `
        <div class="stat-card" data-color="red"><span class="stat-icon">🚨</span><span class="stat-value">${urgent}</span><span class="stat-label">Urgent Actions</span></div>
        <div class="stat-card" data-color="amber"><span class="stat-icon">⚠️</span><span class="stat-value">${high}</span><span class="stat-label">High Priority</span></div>
        <div class="stat-card" data-color="green"><span class="stat-icon">✅</span><span class="stat-value">${recs.filter(r => r.alert === 'Healthy').length}</span><span class="stat-label">Healthy Items</span></div>
        <div class="stat-card" data-color="blue"><span class="stat-icon">📋</span><span class="stat-value">${totalOrder}</span><span class="stat-label">Units to Order</span></div>`;

    // Sort: urgent first
    const priority = { 'Urgent': 0, 'High': 1, 'Medium': 2, 'Low': 3, 'Normal': 4 };
    const sorted = [...recs].sort((a, b) => (priority[a.priority] || 5) - (priority[b.priority] || 5));

    let feed = '';
    sorted.forEach(r => {
        const icon = alertIcons[r.alert] || '📦';
        const color = alertColors[r.alert] || '#94a3b8';
        feed += `
        <div class="alert-card" style="border-left-color:${color}">
            <div class="alert-icon">${icon}</div>
            <div class="alert-body">
                <div class="alert-title">${r.product} · ${r.brand}</div>
                <div class="alert-subtitle">${r.alert} · Priority: ${r.priority}</div>
                <div class="alert-action">💡 ${r.action}</div>
            </div>
            <div class="alert-badge">
                <span class="alert-qty" style="color:${color}">${r.current_stock}</span>
                <span class="alert-label">Current Stock</span>
            </div>
        </div>`;
    });
    document.getElementById('alert-feed').innerHTML = feed;
}

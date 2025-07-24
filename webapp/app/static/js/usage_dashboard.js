/**
 * Usage Dashboard JavaScript
 * Handles data loading, filtering, and chart visualization for the usage dashboard.
 */

// Initialize date inputs with current date range (last 30 days)
function initDateRange() {
    const today = new Date();
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(today.getDate() - 30);
    
    document.getElementById('historyStartDate').value = formatDate(thirtyDaysAgo);
    document.getElementById('historyEndDate').value = formatDate(today);
}

function formatDate(date) {
    return date.toISOString().split('T')[0];
}

// Load daily usage summary
function loadDailySummary() {
    showLoading('dailyUsageChart');
    
    fetch('/api/usage/daily-summary')
        .then(response => response.json())
        .then(data => {
            if (data.success && data.data.daily_summary && data.data.daily_summary.length > 0) {
                renderDailyChart(data.data.daily_summary);
            } else {
                showNoData('dailyUsageChart', 'No daily usage data available.');
            }
        })
        .catch(error => {
            console.error('Error loading daily summary:', error);
            showError('dailyUsageChart', 'Error loading daily usage data.');
        });
}

// Load model breakdown
function loadModelBreakdown() {
    showLoading('providerChart');
    showLoading('modelDistributionChart');
    showLoading('modelUsageChart');
    
    fetch('/api/usage/model-breakdown')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                if (data.data.by_provider && data.data.by_provider.length > 0) {
                    renderProviderChart(data.data.by_provider);
                } else {
                    showNoData('providerChart', 'No provider data available.');
                }
                
                if (data.data.by_model && data.data.by_model.length > 0) {
                    renderModelDistributionChart(data.data.by_model);
                    renderModelUsageChart(data.data.by_model);
                } else {
                    showNoData('modelDistributionChart', 'No model distribution data available.');
                    showNoData('modelUsageChart', 'No model usage data available.');
                }
            }
        })
        .catch(error => {
            console.error('Error loading model breakdown:', error);
            showError('providerChart', 'Error loading provider data.');
            showError('modelDistributionChart', 'Error loading model data.');
            showError('modelUsageChart', 'Error loading model data.');
        });
}

// Load history data
function loadHistory() {
    showLoading('historyTable');
    
    const startDate = document.getElementById('historyStartDate').value;
    const endDate = document.getElementById('historyEndDate').value;
    const service = document.getElementById('historyService').value;
    
    let url = '/api/usage/history?';
    if (startDate) url += `start_date=${encodeURIComponent(startDate)}&`;
    if (endDate) url += `end_date=${encodeURIComponent(endDate)}&`;
    if (service) url += `service=${encodeURIComponent(service)}&`;
    url += 'limit=100';
    
    fetch(url)
        .then(response => response.json())
        .then(data => {
            if (data.success && data.data.history) {
                renderHistoryTable(data.data.history);
            } else {
                showNoData('historyTable', 'No usage history data available for the selected filters.');
            }
        })
        .catch(error => {
            console.error('Error loading history:', error);
            showError('historyTable', 'Error loading usage history.');
        });
}

// Handle cost estimator
function initCostEstimator() {
    document.getElementById('costEstimatorForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const modelId = document.getElementById('estimator-model').value;
        const inputTokens = parseInt(document.getElementById('estimator-input-tokens').value) || 0;
        const outputTokens = parseInt(document.getElementById('estimator-output-tokens').value) || 0;
        
        if (!modelId) {
            alert('Please select a model');
            return;
        }
        
        showLoading('estimationResults');
        
        fetch('/api/usage/estimate-cost', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCsrfToken()
            },
            body: JSON.stringify({
                model_id: modelId,
                input_tokens: inputTokens,
                output_tokens: outputTokens
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                renderCostEstimate(data.data);
            } else {
                showError('estimationResults', data.error || 'Error calculating cost');
            }
        })
        .catch(error => {
            console.error('Error estimating cost:', error);
            showError('estimationResults', 'Error calculating cost');
        });
    });
}

// Get CSRF token from meta tag
function getCsrfToken() {
    return document.querySelector('meta[name="csrf-token"]').getAttribute('content');
}

// Utility functions for loading states
function showLoading(elementId) {
    document.getElementById(elementId).innerHTML = `
        <div class="d-flex justify-content-center align-items-center h-100">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        </div>
    `;
}

function showNoData(elementId, message) {
    document.getElementById(elementId).innerHTML = `
        <div class="text-center text-muted p-5">
            <p>${message}</p>
        </div>
    `;
}

function showError(elementId, message) {
    document.getElementById(elementId).innerHTML = `
        <div class="text-center text-danger p-5">
            <p>${message}</p>
        </div>
    `;
}

// Render charts and tables
function renderDailyChart(data) {
    const labels = data.map(item => item.day);
    const tokenValues = data.map(item => item.tokens_used);
    const costValues = data.map(item => item.cost);
    
    const ctx = document.getElementById('dailyUsageChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Tokens',
                    data: tokenValues,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    yAxisID: 'y',
                    fill: true
                },
                {
                    label: 'Cost ($)',
                    data: costValues,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.0)',
                    yAxisID: 'y1',
                    borderDash: [5, 5]
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Tokens'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Cost ($)'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
}

function renderProviderChart(data) {
    const labels = data.map(item => item.provider_display);
    const costValues = data.map(item => item.cost);
    const tokenValues = data.map(item => item.tokens_used);
    
    const backgroundColors = [
        'rgba(54, 162, 235, 0.7)',
        'rgba(255, 99, 132, 0.7)',
        'rgba(75, 192, 192, 0.7)',
        'rgba(255, 159, 64, 0.7)',
        'rgba(153, 102, 255, 0.7)'
    ];
    
    const ctx = document.getElementById('providerChart').getContext('2d');
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                label: 'Cost Distribution',
                data: costValues,
                backgroundColor: backgroundColors,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const index = context.dataIndex;
                            const cost = costValues[index].toFixed(2);
                            const tokens = tokenValues[index].toLocaleString();
                            return [
                                `Cost: $${cost}`,
                                `Tokens: ${tokens}`
                            ];
                        }
                    }
                }
            }
        }
    });
}

function renderModelDistributionChart(data) {
    if (!data || data.length === 0) {
        showNoData('modelDistributionChart', 'No model data available.');
        return;
    }
    
    // Get top 5 models by cost
    const sortedData = [...data].sort((a, b) => b.cost - a.cost);
    const top5 = sortedData.slice(0, 5);
    
    const labels = top5.map(item => item.display_name);
    const costValues = top5.map(item => item.cost);
    
    const backgroundColors = [
        'rgba(54, 162, 235, 0.7)',
        'rgba(255, 99, 132, 0.7)',
        'rgba(75, 192, 192, 0.7)',
        'rgba(255, 159, 64, 0.7)',
        'rgba(153, 102, 255, 0.7)'
    ];
    
    const ctx = document.getElementById('modelDistributionChart').getContext('2d');
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                label: 'Cost by Model',
                data: costValues,
                backgroundColor: backgroundColors,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    display: true
                },
                title: {
                    display: true,
                    text: 'Top 5 Models by Cost'
                }
            }
        }
    });
}

function renderModelUsageChart(data) {
    if (!data || data.length === 0) {
        showNoData('modelUsageChart', 'No model usage data available.');
        return;
    }
    
    // Sort by tokens used
    const sortedData = [...data].sort((a, b) => b.tokens_used - a.tokens_used);
    const top10 = sortedData.slice(0, 10); // Top 10 models
    
    const labels = top10.map(item => item.display_name);
    const tokenValues = top10.map(item => item.tokens_used);
    
    const ctx = document.getElementById('modelUsageChart').getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Tokens Used',
                data: tokenValues,
                backgroundColor: 'rgba(54, 162, 235, 0.7)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Top 10 Models by Token Usage'
                },
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Tokens'
                    }
                }
            }
        }
    });
}

function renderHistoryTable(data) {
    if (!data || data.length === 0) {
        showNoData('historyTable', 'No history data available for the selected filters.');
        return;
    }
    
    let html = `
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>Date/Time</th>
                    <th>Service</th>
                    <th>Endpoint</th>
                    <th>Tokens</th>
                    <th>Cost</th>
                    <th>Task Type</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    data.forEach(record => {
        const timestamp = new Date(record.timestamp).toLocaleString();
        html += `
            <tr>
                <td>${timestamp}</td>
                <td>${record.service}</td>
                <td>${record.endpoint}</td>
                <td>${record.tokens_used.toLocaleString()}</td>
                <td>$${record.cost.toFixed(4)}</td>
                <td>${record.task_type || '-'}</td>
            </tr>
        `;
    });
    
    html += `
            </tbody>
        </table>
    `;
    
    document.getElementById('historyTable').innerHTML = html;
}

function renderCostEstimate(data) {
    let html = `
        <div class="mb-4">
            <h6>Summary</h6>
            <div class="d-flex justify-content-between mb-2">
                <span>Model:</span>
                <span class="fw-bold">${data.display_name}</span>
            </div>
            <div class="d-flex justify-content-between mb-2">
                <span>Input Tokens:</span>
                <span>${data.input_tokens.toLocaleString()}</span>
            </div>
            <div class="d-flex justify-content-between mb-2">
                <span>Output Tokens:</span>
                <span>${data.output_tokens.toLocaleString()}</span>
            </div>
            <div class="d-flex justify-content-between mb-2">
                <span>Total Tokens:</span>
                <span>${(data.input_tokens + data.output_tokens).toLocaleString()}</span>
            </div>
        </div>
        
        <hr>
        
        <div class="mb-4">
            <h6>Cost Breakdown</h6>
            <div class="d-flex justify-content-between mb-2">
                <span>Input Cost:</span>
                <span>$${data.input_cost.toFixed(4)}</span>
            </div>
            <div class="d-flex justify-content-between mb-2">
                <span>Output Cost:</span>
                <span>$${data.output_cost.toFixed(4)}</span>
            </div>
            <div class="d-flex justify-content-between mb-2 fw-bold">
                <span>Total Cost:</span>
                <span>$${data.total_cost.toFixed(4)}</span>
            </div>
        </div>
        
        <div class="alert alert-info">
            <i class="bi bi-info-circle me-2"></i>
            This is an estimate based on current pricing. Actual costs may vary.
        </div>
    `;
    
    document.getElementById('estimationResults').innerHTML = html;
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize date range for history tab
    initDateRange();
    
    // Load data for the active tab (overview) first
    loadDailySummary();
    loadModelBreakdown();
    
    // Initialize cost estimator
    initCostEstimator();
    
    // Set up tab switching
    document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', function(event) {
            const targetTab = event.target.getAttribute('data-bs-target').substring(1);
            
            if (targetTab === 'history') {
                loadHistory();
            }
        });
    });
    
    // Set up filter button for history tab
    document.getElementById('loadHistoryBtn').addEventListener('click', loadHistory);
});
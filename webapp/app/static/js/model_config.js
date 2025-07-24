/**
 * Model Configuration Management
 * 
 * This script enhances the model configuration UI with
 * AJAX operations, filtering, and dynamic updates.
 */

// Main initialization
document.addEventListener('DOMContentLoaded', function() {
    // Initialize components
    initModelFiltering();
    initApiKeyManagement();
    initStatusToggles();
    initModelFilter();
    initCostEstimator();
    addEventListeners();
});

/**
 * Initialize model filtering 
 */
function initModelFiltering() {
    const filterDropdown = document.getElementById('modelFilterDropdown');
    if (!filterDropdown) return;
    
    const filterItems = filterDropdown.nextElementSibling.querySelectorAll('.dropdown-item');
    
    filterItems.forEach(item => {
        item.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Get filter value
            const filter = this.getAttribute('data-filter') || 'all';
            
            // Update dropdown button text
            filterDropdown.textContent = this.textContent;
            
            // Apply filter
            const rows = document.querySelectorAll('.model-row');
            rows.forEach(row => {
                const provider = row.getAttribute('data-provider');
                
                if (filter === 'all' || provider === filter) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        });
    });
}

/**
 * Initialize API key management
 */
function initApiKeyManagement() {
    // Toggle API key visibility
    const toggleButtons = document.querySelectorAll('.toggle-password');
    
    toggleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const input = this.previousElementSibling;
            const type = input.getAttribute('type') === 'password' ? 'text' : 'password';
            input.setAttribute('type', type);
            
            // Update icon
            const icon = this.querySelector('i');
            if (type === 'password') {
                icon.classList.remove('bi-eye-slash');
                icon.classList.add('bi-eye');
            } else {
                icon.classList.remove('bi-eye');
                icon.classList.add('bi-eye-slash');
            }
        });
    });
    
    // API key validation
    const apiKeyForms = document.querySelectorAll('.api-key-form');
    
    apiKeyForms.forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const provider = form.querySelector('input[name="provider"]').value;
            const apiKey = form.querySelector('input[name="api_key"]').value;
            
            // Simple validation
            if (!apiKey.trim()) {
                showAlert('Please enter an API key', 'danger');
                return;
            }
            
            // Save API key via AJAX
            fetch('/api/config/api-keys', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCsrfToken()
                },
                body: JSON.stringify({
                    provider: provider,
                    api_key: apiKey
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showAlert(data.error, 'danger');
                } else {
                    showAlert(data.message, 'success');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showAlert('An error occurred while saving the API key', 'danger');
            });
        });
    });
}

/**
 * Initialize model status toggles
 */
function initStatusToggles() {
    const activateButtons = document.querySelectorAll('.activate-model');
    const deactivateButtons = document.querySelectorAll('.deactivate-model');
    const defaultButtons = document.querySelectorAll('.set-default-model');
    
    // Activate model
    activateButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            const modelId = this.getAttribute('data-model-id');
            toggleModelStatus(modelId, true);
        });
    });
    
    // Deactivate model
    deactivateButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            const modelId = this.getAttribute('data-model-id');
            toggleModelStatus(modelId, false);
        });
    });
    
    // Set default model
    defaultButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            const modelId = this.getAttribute('data-model-id');
            setDefaultModel(modelId);
        });
    });
}

/**
 * Initialize model filter and provider filter
 */
function initModelFilter() {
    const searchInput = document.getElementById('model-search');
    if (!searchInput) return;
    
    searchInput.addEventListener('keyup', function() {
        const searchTerm = this.value.toLowerCase();
        
        const rows = document.querySelectorAll('.model-row');
        rows.forEach(row => {
            const modelName = row.querySelector('.model-name').textContent.toLowerCase();
            const modelId = row.querySelector('.model-id').textContent.toLowerCase();
            
            if (modelName.includes(searchTerm) || modelId.includes(searchTerm)) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        });
    });
}

/**
 * Add event listeners for other UI components
 */
function addEventListeners() {
    // Add listener for delete model buttons
    const deleteButtons = document.querySelectorAll('.delete-model');
    
    deleteButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            const modelId = this.getAttribute('data-model-id');
            const modelName = this.getAttribute('data-model-name');
            
            if (confirm(`Are you sure you want to delete the model "${modelName}"? This action cannot be undone.`)) {
                deleteModel(modelId);
            }
        });
    });
}

/**
 * Toggle model status (activate/deactivate)
 */
function toggleModelStatus(modelId, activate) {
    fetch(`/api/config/models/${modelId}/status`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCsrfToken()
        },
        body: JSON.stringify({
            activate: activate
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showAlert(data.error, 'danger');
        } else {
            showAlert(data.message, 'success');
            // Reload page to reflect changes
            setTimeout(() => window.location.reload(), 1000);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('An error occurred while updating model status', 'danger');
    });
}

/**
 * Set default model
 */
function setDefaultModel(modelId) {
    fetch(`/api/config/models/${modelId}/default`, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCsrfToken()
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showAlert(data.error, 'danger');
        } else {
            showAlert(data.message, 'success');
            // Reload page to reflect changes
            setTimeout(() => window.location.reload(), 1000);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('An error occurred while setting the default model', 'danger');
    });
}

/**
 * Delete model
 */
function deleteModel(modelId) {
    fetch(`/api/config/models/${modelId}`, {
        method: 'DELETE',
        headers: {
            'X-CSRFToken': getCsrfToken()
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showAlert(data.error, 'danger');
        } else {
            showAlert(data.message, 'success');
            // Reload page to reflect changes
            setTimeout(() => window.location.reload(), 1000);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('An error occurred while deleting the model', 'danger');
    });
}

/**
 * Show an alert message
 */
function showAlert(message, type = 'info') {
    const alertsContainer = document.getElementById('alerts-container');
    if (!alertsContainer) return;
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertsContainer.appendChild(alert);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alert.classList.remove('show');
        setTimeout(() => alert.remove(), 150);
    }, 5000);
}

/**
 * Initialize cost estimator
 */
function initCostEstimator() {
    // Add cost estimator container to the page
    // Find the card with "Usage Statistics" title
    let usageStatsCard = null;
    document.querySelectorAll('.card .card-header h5.mb-0').forEach(el => {
        if (el.textContent.trim() === 'Usage Statistics') {
            usageStatsCard = el.closest('.card');
        }
    });
    if (!usageStatsCard) return;
    
    const cardBody = usageStatsCard.querySelector('.card-body');
    
    // Add cost estimator form
    const estimatorHtml = `
        <div class="card bg-light mt-4">
            <div class="card-header">
                <h5 class="mb-0">Cost Estimator</h5>
            </div>
            <div class="card-body">
                <p>Estimate the cost of running a specific number of tokens through any model.</p>
                
                <div class="row g-3">
                    <div class="col-md-6">
                        <label for="estimator-model" class="form-label">Select Model</label>
                        <select class="form-select" id="estimator-model">
                            <option value="">Select a model...</option>
                            ${Array.from(document.querySelectorAll('.model-row[data-provider]')).map(row => {
                                const modelName = row.querySelector('.model-name').textContent.trim();
                                const modelId = row.getAttribute('data-model-id');
                                return `<option value="${modelId}" data-input-cost="${parseFloat(row.getAttribute('data-input-cost') || '0')}" data-output-cost="${parseFloat(row.getAttribute('data-output-cost') || '0')}">${modelName}</option>`;
                            }).join('')}
                        </select>
                    </div>
                    
                    <div class="col-md-3">
                        <label for="estimator-input-tokens" class="form-label">Input Tokens</label>
                        <input type="number" class="form-control" id="estimator-input-tokens" min="0" value="1000">
                    </div>
                    
                    <div class="col-md-3">
                        <label for="estimator-output-tokens" class="form-label">Output Tokens</label>
                        <input type="number" class="form-control" id="estimator-output-tokens" min="0" value="500">
                    </div>
                </div>
                
                <div class="mt-3 text-center">
                    <button type="button" class="btn btn-primary" id="estimate-cost-btn">Calculate Cost</button>
                </div>
                
                <div id="cost-result" class="alert alert-info mt-3 d-none">
                    <h5 class="mb-1">Estimated Cost: <span id="estimated-cost">$0.00</span></h5>
                    <div class="row small">
                        <div class="col-md-6">Input Cost: <span id="input-cost">$0.00</span></div>
                        <div class="col-md-6">Output Cost: <span id="output-cost">$0.00</span></div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Append to card body
    const estimatorContainer = document.createElement('div');
    estimatorContainer.innerHTML = estimatorHtml;
    cardBody.appendChild(estimatorContainer);
    
    // Add event listener for the calculate button
    const calculateBtn = document.getElementById('estimate-cost-btn');
    if (calculateBtn) {
        calculateBtn.addEventListener('click', calculateCost);
    }
}

/**
 * Calculate estimated cost based on input and output tokens
 */
function calculateCost() {
    const modelSelect = document.getElementById('estimator-model');
    const selectedOption = modelSelect.options[modelSelect.selectedIndex];
    
    if (!selectedOption || !selectedOption.value) {
        showAlert('Please select a model', 'warning');
        return;
    }
    
    const inputTokens = parseInt(document.getElementById('estimator-input-tokens').value) || 0;
    const outputTokens = parseInt(document.getElementById('estimator-output-tokens').value) || 0;
    
    const inputCostPer1k = parseFloat(selectedOption.getAttribute('data-input-cost')) || 0;
    const outputCostPer1k = parseFloat(selectedOption.getAttribute('data-output-cost')) || 0;
    
    // Calculate costs
    const inputCost = (inputTokens / 1000) * inputCostPer1k;
    const outputCost = (outputTokens / 1000) * outputCostPer1k;
    const totalCost = inputCost + outputCost;
    
    // Update the display
    document.getElementById('input-cost').textContent = `$${inputCost.toFixed(6)}`;
    document.getElementById('output-cost').textContent = `$${outputCost.toFixed(6)}`;
    document.getElementById('estimated-cost').textContent = `$${totalCost.toFixed(6)}`;
    
    // Show the result
    document.getElementById('cost-result').classList.remove('d-none');
}

/**
 * Get CSRF token from meta tag
 */
function getCsrfToken() {
    return document.querySelector('meta[name="csrf-token"]').getAttribute('content');
}
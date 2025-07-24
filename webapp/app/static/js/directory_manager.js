/**
 * Directory management functionality for Paper2Code
 */

// Get CSRF token for AJAX requests
function getCsrfToken() {
    return document.querySelector('meta[name="csrf-token"]').getAttribute('content');
}

// Format bytes to human-readable size
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
    
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

document.addEventListener('DOMContentLoaded', function() {
    // Set up path validation for directory forms
    setupPathValidation();
    
    // Setup move files checkbox toggle
    setupMoveFilesToggle();
    
    // Setup tooltips
    setupTooltips();
});

/**
 * Set up path validation for directory input fields
 */
function setupPathValidation() {
    const pathInput = document.getElementById('path');
    const pathFeedback = document.getElementById('path-feedback');
    
    if (!pathInput || !pathFeedback) return;
    
    // Add debounce function to avoid too many requests
    let timeoutId;
    
    pathInput.addEventListener('input', function() {
        // Clear previous timeout
        clearTimeout(timeoutId);
        
        // Set new timeout for validation
        timeoutId = setTimeout(() => {
            validatePath(pathInput.value, pathFeedback);
        }, 500); // 500ms debounce
    });
    
    // Initial validation if path is already filled
    if (pathInput.value) {
        validatePath(pathInput.value, pathFeedback);
    }
}

/**
 * Setup move files checkbox toggle based on path changes
 */
function setupMoveFilesToggle() {
    const moveFilesCheck = document.getElementById('move_files');
    const pathInput = document.getElementById('path');
    
    if (moveFilesCheck && pathInput) {
        const originalPath = pathInput.getAttribute('data-original-path') || pathInput.value;
        
        pathInput.addEventListener('input', function() {
            const pathChanged = this.value !== originalPath;
            
            // Only show the option if path changes
            moveFilesCheck.closest('.form-check').style.display = pathChanged ? 'block' : 'none';
            
            // Reset the checkbox when path changes back to original
            if (!pathChanged) {
                moveFilesCheck.checked = false;
            }
        });
        
        // Initial check
        moveFilesCheck.closest('.form-check').style.display = 
            (pathInput.value !== originalPath) ? 'block' : 'none';
    }
}

/**
 * Setup Bootstrap tooltips
 */
function setupTooltips() {
    // Check if Bootstrap's tooltip function exists
    if (typeof bootstrap !== 'undefined' && typeof bootstrap.Tooltip !== 'undefined') {
        const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
        const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    }
}

/**
 * Validate a directory path using the API
 * 
 * @param {string} path - The path to validate
 * @param {HTMLElement} feedbackElement - Element to show validation feedback
 */
function validatePath(path, feedbackElement) {
    if (!path) {
        feedbackElement.innerHTML = '';
        return;
    }
    
    // Show loading indicator
    feedbackElement.innerHTML = '<i class="bi bi-arrow-repeat spin"></i> Validating path...';
    
    // Call the validation API
    fetch('/api/directories/check-path?path=' + encodeURIComponent(path), {
        method: 'GET',
        headers: {
            'X-CSRFToken': getCsrfToken()
        }
    })
    .then(response => response.json())
    .then(data => {
        if (!data.success) {
            // Show error message
            feedbackElement.innerHTML = `<span class="text-danger"><i class="bi bi-exclamation-triangle"></i> ${data.error}</span>`;
            return;
        }
        
        // Show validation results
        let feedbackHtml = '';
        
        if (data.exists && data.is_directory) {
            feedbackHtml += '<span class="text-success"><i class="bi bi-check-circle"></i> Directory exists</span><br>';
            
            if (data.readable) {
                feedbackHtml += '<span class="text-success"><i class="bi bi-check-circle"></i> Directory is readable</span><br>';
            } else {
                feedbackHtml += '<span class="text-warning"><i class="bi bi-exclamation-triangle"></i> Directory is not readable</span><br>';
            }
            
            if (data.writable) {
                feedbackHtml += '<span class="text-success"><i class="bi bi-check-circle"></i> Directory is writable</span>';
            } else {
                feedbackHtml += '<span class="text-warning"><i class="bi bi-exclamation-triangle"></i> Directory is not writable</span>';
            }
            
            // Add usage info if available
            if (data.usage_info && data.usage_info.file_count) {
                feedbackHtml += `<br><span class="text-muted"><i class="bi bi-hdd"></i> ${data.usage_info.file_count} files, ${data.usage_info.used_space} used</span>`;
            }
        } else if (data.exists) {
            feedbackHtml = '<span class="text-warning"><i class="bi bi-exclamation-triangle"></i> Path exists but is not a directory</span>';
        } else {
            feedbackHtml = '<span class="text-info"><i class="bi bi-info-circle"></i> Directory does not exist (will be created)</span>';
        }
        
        feedbackElement.innerHTML = feedbackHtml;
    })
    .catch(error => {
        // Handle errors
        feedbackElement.innerHTML = `<span class="text-danger"><i class="bi bi-exclamation-triangle"></i> Error: ${error.message}</span>`;
    });
}
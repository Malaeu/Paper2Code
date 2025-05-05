/**
 * Directory management functionality for Paper2Code
 */

document.addEventListener('DOMContentLoaded', function() {
    // Set up path validation for directory forms
    setupPathValidation();
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
    feedbackElement.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Validating path...';
    
    // Call the validation API
    fetch('/api/directories/validate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ path }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'error') {
            // Show error message
            feedbackElement.innerHTML = `<span class="text-danger"><i class="fas fa-times-circle"></i> ${data.message}</span>`;
            return;
        }
        
        // Show validation results
        const permissions = data.data;
        let feedbackHtml = '';
        
        if (permissions.exists && permissions.is_directory) {
            feedbackHtml += '<span class="text-success"><i class="fas fa-check-circle"></i> Directory exists</span><br>';
            
            if (permissions.readable) {
                feedbackHtml += '<span class="text-success"><i class="fas fa-check-circle"></i> Directory is readable</span><br>';
            } else {
                feedbackHtml += '<span class="text-warning"><i class="fas fa-exclamation-triangle"></i> Directory is not readable</span><br>';
            }
            
            if (permissions.writable) {
                feedbackHtml += '<span class="text-success"><i class="fas fa-check-circle"></i> Directory is writable</span>';
            } else {
                feedbackHtml += '<span class="text-warning"><i class="fas fa-exclamation-triangle"></i> Directory is not writable</span>';
            }
        } else if (permissions.exists) {
            feedbackHtml = '<span class="text-warning"><i class="fas fa-exclamation-triangle"></i> Path exists but is not a directory</span>';
        } else {
            feedbackHtml = '<span class="text-info"><i class="fas fa-info-circle"></i> Directory does not exist (will be created)</span>';
        }
        
        feedbackElement.innerHTML = feedbackHtml;
    })
    .catch(error => {
        // Handle errors
        feedbackElement.innerHTML = `<span class="text-danger"><i class="fas fa-times-circle"></i> Error: ${error.message}</span>`;
    });
}
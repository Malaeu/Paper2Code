// Main JavaScript file for Paper2Code web application

document.addEventListener('DOMContentLoaded', function() {
    console.log('Paper2Code web application initialized');
    
    // Initialize UI components
    initializeUIComponents();
    
    // Setup event listeners
    setupEventListeners();
});

function initializeUIComponents() {
    // Flash message auto-dismiss
    const flashMessages = document.querySelectorAll('.alert');
    flashMessages.forEach(message => {
        setTimeout(() => {
            message.classList.add('fade-out');
            setTimeout(() => {
                message.remove();
            }, 500);
        }, 5000);
    });
}

function setupEventListeners() {
    // Example: Form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
                highlightInvalidFields(form);
            }
            form.classList.add('was-validated');
        });
    });
}

function highlightInvalidFields(form) {
    const invalidFields = form.querySelectorAll(':invalid');
    invalidFields.forEach(field => {
        field.classList.add('is-invalid');
        
        // Add error message
        const errorMessage = document.createElement('div');
        errorMessage.className = 'invalid-feedback';
        errorMessage.textContent = field.validationMessage;
        field.parentNode.appendChild(errorMessage);
    });
}
/**
 * Paper2Code Web Interface - Main JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // Handle file input styling
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name || 'No file chosen';
            const label = e.target.nextElementSibling;
            if (label && label.classList.contains('form-file-label')) {
                label.textContent = fileName;
            }
        });
    });

    // Handle form validation
    const forms = document.querySelectorAll('.needs-validation');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });

    // Dynamically add more variable mappings
    const addMappingBtn = document.getElementById('add-mapping');
    if (addMappingBtn) {
        addMappingBtn.addEventListener('click', function() {
            const mappingsContainer = document.getElementById('mappings-container');
            const mappingCount = mappingsContainer.children.length;
            
            const newRow = document.createElement('tr');
            newRow.innerHTML = `
                <td>
                    <input type="text" class="form-control" name="map_var${mappingCount}" placeholder="Original variable">
                </td>
                <td>
                    <select class="form-select" name="map_val${mappingCount}">
                        <option value="">Select a column</option>
                        ${Array.from(document.querySelectorAll('#mappings-container tr:first-child td:last-child select option')).map(opt => 
                            `<option value="${opt.value}">${opt.textContent}</option>`
                        ).join('')}
                    </select>
                </td>
            `;
            
            mappingsContainer.appendChild(newRow);
        });
    }

    // Auto-resize textareas
    const autoResizeTextareas = document.querySelectorAll('textarea.auto-resize');
    autoResizeTextareas.forEach(textarea => {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
        
        // Trigger the event once to set initial height
        const event = new Event('input');
        textarea.dispatchEvent(event);
    });

    // Handle tab navigation in code editor (if present)
    const codeEditors = document.querySelectorAll('textarea.code-editor');
    codeEditors.forEach(editor => {
        editor.addEventListener('keydown', function(e) {
            if (e.key === 'Tab') {
                e.preventDefault();
                
                // Insert tab at cursor position
                const start = this.selectionStart;
                const end = this.selectionEnd;
                
                this.value = this.value.substring(0, start) + 
                            '    ' + 
                            this.value.substring(end);
                
                // Move cursor after the inserted tab
                this.selectionStart = this.selectionEnd = start + 4;
            }
        });
    });
});
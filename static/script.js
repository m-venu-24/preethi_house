// JavaScript for House Price Predictor

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const loadingDiv = document.getElementById('loading');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const placeholderDiv = document.getElementById('placeholder');
    const predictedPriceSpan = document.getElementById('predictedPrice');
    const errorMessageSpan = document.getElementById('errorMessage');

    // Form submission handler
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Hide all result divs
        hideAllResults();
        
        // Show loading
        showLoading();
        
        try {
            // Get form data
            const formData = {
                avg_area_income: parseFloat(document.getElementById('avg_area_income').value),
                avg_area_house_age: parseFloat(document.getElementById('avg_area_house_age').value),
                avg_area_rooms: parseFloat(document.getElementById('avg_area_rooms').value),
                avg_area_bedrooms: parseFloat(document.getElementById('avg_area_bedrooms').value),
                area_population: parseFloat(document.getElementById('area_population').value)
            };
            
            // Validate inputs
            if (!validateInputs(formData)) {
                throw new Error('Please enter valid numbers for all fields');
            }
            
            // Make API call
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Show result
                predictedPriceSpan.textContent = data.prediction;
                showResult();
                
                // Add animation
                resultDiv.classList.add('fade-in-up');
            } else {
                throw new Error(data.error || 'Prediction failed');
            }
            
        } catch (error) {
            console.error('Error:', error);
            errorMessageSpan.textContent = error.message;
            showError();
        }
    });
    
    // Input validation
    function validateInputs(data) {
        for (const [key, value] of Object.entries(data)) {
            if (isNaN(value) || value < 0) {
                return false;
            }
        }
        return true;
    }
    
    // Show/hide functions
    function hideAllResults() {
        loadingDiv.classList.add('d-none');
        resultDiv.classList.add('d-none');
        errorDiv.classList.add('d-none');
        placeholderDiv.classList.add('d-none');
    }
    
    function showLoading() {
        loadingDiv.classList.remove('d-none');
    }
    
    function showResult() {
        resultDiv.classList.remove('d-none');
    }
    
    function showError() {
        errorDiv.classList.remove('d-none');
    }
    
    // Add some sample data buttons for quick testing
    addSampleDataButtons();
    
    function addSampleDataButtons() {
        const form = document.getElementById('predictionForm');
        
        // Create sample data buttons container
        const sampleContainer = document.createElement('div');
        sampleContainer.className = 'mt-3';
        sampleContainer.innerHTML = `
            <div class="d-grid gap-2">
                <button type="button" class="btn btn-outline-primary btn-sm" onclick="fillSampleData('luxury')">
                    <i class="fas fa-crown"></i> Luxury Area Sample
                </button>
                <button type="button" class="btn btn-outline-success btn-sm" onclick="fillSampleData('average')">
                    <i class="fas fa-home"></i> Average Area Sample
                </button>
                <button type="button" class="btn btn-outline-warning btn-sm" onclick="fillSampleData('budget')">
                    <i class="fas fa-dollar-sign"></i> Budget Area Sample
                </button>
            </div>
        `;
        
        form.appendChild(sampleContainer);
    }
    
    // Sample data sets
    window.fillSampleData = function(type) {
        const sampleData = {
            luxury: {
                avg_area_income: 85000,
                avg_area_house_age: 4.5,
                avg_area_rooms: 8.5,
                avg_area_bedrooms: 5.2,
                area_population: 25000
            },
            average: {
                avg_area_income: 65000,
                avg_area_house_age: 6.0,
                avg_area_rooms: 7.0,
                avg_area_bedrooms: 4.0,
                area_population: 35000
            },
            budget: {
                avg_area_income: 45000,
                avg_area_house_age: 8.0,
                avg_area_rooms: 5.5,
                avg_area_bedrooms: 3.0,
                area_population: 45000
            }
        };
        
        const data = sampleData[type];
        document.getElementById('avg_area_income').value = data.avg_area_income;
        document.getElementById('avg_area_house_age').value = data.avg_area_house_age;
        document.getElementById('avg_area_rooms').value = data.avg_area_rooms;
        document.getElementById('avg_area_bedrooms').value = data.avg_area_bedrooms;
        document.getElementById('area_population').value = data.area_population;
    };
    
    // Add number formatting for better UX
    const numberInputs = document.querySelectorAll('input[type="number"]');
    numberInputs.forEach(input => {
        input.addEventListener('blur', function() {
            if (this.value && !isNaN(this.value)) {
                // Format the number based on the field
                if (this.id === 'avg_area_income' || this.id === 'area_population') {
                    this.value = parseInt(this.value).toLocaleString();
                } else {
                    this.value = parseFloat(this.value).toFixed(1);
                }
            }
        });
        
        input.addEventListener('focus', function() {
            // Remove formatting when focused for easier editing
            if (this.value) {
                this.value = this.value.replace(/,/g, '');
            }
        });
    });
});

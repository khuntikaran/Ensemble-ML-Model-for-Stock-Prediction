let priceChart = null;

async function getPrediction() {
    const company = document.getElementById('company').value;
    const resultDiv = document.getElementById('result');
    
    resultDiv.innerHTML = '<p class="loading">Loading prediction...</p>';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ company: company })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            const predictionDate = new Date(data.timestamp);
            const formattedPrice = new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(data.prediction);

            resultDiv.innerHTML = `
                <div class="prediction-result">
                    <h3>${company} Stock Prediction</h3>
                    <p class="price">Predicted Price: ${formattedPrice}</p>
                    <p class="confidence">Confidence: ${(data.confidence * 100).toFixed(1)}%</p>
                    <p class="timestamp">As of: ${predictionDate.toLocaleString()}</p>
                </div>
            `;

            updateChart(company, data.prediction, predictionDate);
        } else {
            resultDiv.innerHTML = `<p class="error">Error: ${data.error}</p>`;
        }
    } catch (error) {
        resultDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
}

function updateChart(company, prediction, date) {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    if (priceChart) {
        priceChart.destroy();
    }
    
    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Current', 'Predicted'],
            datasets: [{
                label: `${company} Stock Price`,
                data: [prediction * 0.95, prediction], // Example: showing slight increase
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

// Initialize with first company when page loads
document.addEventListener('DOMContentLoaded', () => {
    const companySelect = document.getElementById('company');
    if (companySelect.value) {
        getPrediction();
    }
}); 
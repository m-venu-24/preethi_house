from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return '''
    <html>
    <head>
        <title>Test Page</title>
        <style>
            body { font-family: Arial; text-align: center; padding: 50px; }
            h1 { color: blue; }
        </style>
    </head>
    <body>
        <h1>🎉 Flask is Working! 🎉</h1>
        <p>If you can see this, your Flask server is running correctly!</p>
        <p>Now let's test the main app...</p>
        <a href="http://127.0.0.1:5000/main" style="background: green; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Go to House Price Predictor</a>
    </body>
    </html>
    '''

@app.route('/main')
def main():
    return '''
    <html>
    <head>
        <title>House Price Predictor</title>
        <style>
            body { font-family: Arial; padding: 20px; background: #f0f0f0; }
            .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .form-group { margin: 15px 0; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
            button { background: #27ae60; color: white; padding: 15px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; width: 100%; }
            button:hover { background: #229954; }
            .result { margin-top: 20px; padding: 15px; background: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏠 House Price Predictor</h1>
            <form id="predictionForm">
                <div class="form-group">
                    <label>Average Area Income ($):</label>
                    <input type="number" id="income" placeholder="e.g., 75000" required>
                </div>
                <div class="form-group">
                    <label>Average Area House Age (years):</label>
                    <input type="number" id="age" placeholder="e.g., 6.5" step="0.1" required>
                </div>
                <div class="form-group">
                    <label>Average Area Number of Rooms:</label>
                    <input type="number" id="rooms" placeholder="e.g., 7.2" step="0.1" required>
                </div>
                <div class="form-group">
                    <label>Average Area Number of Bedrooms:</label>
                    <input type="number" id="bedrooms" placeholder="e.g., 4.1" step="0.1" required>
                </div>
                <div class="form-group">
                    <label>Area Population:</label>
                    <input type="number" id="population" placeholder="e.g., 35000" required>
                </div>
                <button type="submit">🔮 Predict Price</button>
            </form>
            <div id="result" class="result" style="display: none;">
                <h3>Predicted Price: <span id="price"></span></h3>
            </div>
        </div>
        
        <script>
            document.getElementById('predictionForm').addEventListener('submit', function(e) {
                e.preventDefault();
                
                const income = parseFloat(document.getElementById('income').value);
                const age = parseFloat(document.getElementById('age').value);
                const rooms = parseFloat(document.getElementById('rooms').value);
                const bedrooms = parseFloat(document.getElementById('bedrooms').value);
                const population = parseFloat(document.getElementById('population').value);
                
                // Simple prediction formula based on our model coefficients
                const predictedPrice = -2437715 + (income * 20.62) + (age * 164583) + (rooms * 127534) + (bedrooms * -17905) + (population * 12.10);
                
                document.getElementById('price').textContent = '$' + predictedPrice.toLocaleString();
                document.getElementById('result').style.display = 'block';
            });
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("Starting simple test Flask app...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)

from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model once when the app starts
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    # Render the main page with buttons
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # For the 'Predict Weather' button action
    if request.method == 'POST':
        # Extract data from the form
        year = request.form.get('year')
        month = request.form.get('month')
        day = request.form.get('day')
        
        # Prepare data for model prediction
        new_data = pd.DataFrame({'year': [int(year)], 'month': [int(month)], 'day': [int(day)]})
        
        # Make prediction
        try:
            prediction = model.predict(new_data)
            result = prediction[0]  # Assuming a single prediction is made
        except Exception as e:
            result = f"Error in prediction: {e}"
        
        # Render the prediction result in index.html
        return render_template('index.html', pred=result)
    
    # For GET requests, render the prediction form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

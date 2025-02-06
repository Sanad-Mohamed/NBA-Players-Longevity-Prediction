import pickle
from flask import Flask, request, jsonify, render_template, flash
import uuid
import pandas as pd

app = Flask(__name__)

app.secret_key = str(uuid.uuid4())

def load_model_scaler():
    # Load both model and scaler from their respective pickle files
    with open('best_model.pkl', 'rb') as model_file, open('robust_scaler.pkl', 'rb') as scaler_file:
        return pickle.load(model_file), pickle.load(scaler_file)

best_model, robust_scaler = load_model_scaler()

# The optimal threshold for our binary classification problem (check the Jupyter Notebook file)
optimal_threshold = 0.73

# The predict function
def predict(X):
    X_scaled = robust_scaler.transform(X)
    y_proba = best_model.predict_proba(X_scaled)[0, 1]
    return int(y_proba >= optimal_threshold)

# Route for homepage
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict_class():
    try:
        # Features
        cols = ['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA', '3P%', 'FTM',
                'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV']

        # Get the input data from the form (POST method)
        data = [request.form.get(col) for col in cols]

        # Check if any input field is empty or missing
        if None in data or '' in data:
            return jsonify({'error': "Please fill out all the fields before submitting !!"})

        # Convert the data to floats
        data = [float(val) for val in data]

        # Convert data to pandas dataframe (1 row)
        features = pd.DataFrame([data], columns=cols)

        # Classify based on the optimal threshold
        predicted_class = predict(features)

        # Return the result as a JSON response
        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
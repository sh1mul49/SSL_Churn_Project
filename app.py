from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize Flask application
application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')  # Main page

# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  # Form page
    else:
        try:
            # Get SMS counts as comma-separated values from the form
            sms_counts = request.form.get('sms_counts')
            
            # Convert input string to a list of floats
            sms_counts_list = list(map(float, sms_counts.split(',')))

            # Create a CustomData instance
            custom_data = CustomData(sms_counts=sms_counts_list)

            # Convert to DataFrame
            input_data = custom_data.get_data_as_data_frame()

            # Initialize prediction pipeline and make predictions
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(input_data)

            # Render the results on the home page
            return render_template('home.html', results=results)

        except Exception as e:
            # Handle errors and display them on the page
            return render_template('home.html', error=str(e))

# Main entry point for the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0")

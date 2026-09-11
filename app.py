# ==========================================================
# ONLINE SHOPPERS PURCHASE INTENTION - FLASK APP
# ==========================================================

import pandas as pd
import joblib

from flask import Flask, render_template_string, request, jsonify


# ==========================================================
# LOAD SAVED MODEL ARTIFACTS
# ==========================================================

gradient_boosting = joblib.load(
    "model/gradient_boosting_model.pkl"
)

ordinal_encoder = joblib.load(
    "model/ordinal_encoder.pkl"
)

preprocessor = joblib.load(
    "model/preprocessor.pkl"
)

categorical_features = joblib.load(
    "model/categorical_features.pkl"
)

numerical_features = joblib.load(
    "model/numerical_features.pkl"
)

binary_features = joblib.load(
    "model/binary_features.pkl"
)


# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(__name__)


# ==========================================================
# HTML FORM
# ==========================================================

HTML_FORM = """
<!DOCTYPE html>
<html>

<head>

<meta charset="UTF-8">

<title>Online Purchase Intention</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: #f4f6f8;
    margin: 0;
    padding: 40px 20px;
}

.container {
    max-width: 900px;
    margin: auto;
    background: white;
    padding: 30px;
    border-radius: 10px;
}

h1 {
    text-align: center;
    color: #2c3e50;
}

.subtitle {
    text-align: center;
    color: #7f8c8d;
    margin-bottom: 30px;
}

form {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 15px 20px;
}

.field {
    display: flex;
    flex-direction: column;
}

label {
    margin-bottom: 5px;
    font-weight: 600;
}

input,
select {
    padding: 9px;
    border: 1px solid #cfd8dc;
    border-radius: 6px;
}

.submit-row {
    grid-column: 1 / -1;
    margin-top: 15px;
}

button {
    width: 100%;
    padding: 12px;
    background: #2980b9;
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 15px;
    cursor: pointer;
}

.result {
    margin-top: 25px;
    padding: 20px;
    text-align: center;
    border-radius: 8px;
}

.result.positive {
    background: #eafaf1;
    color: #1e8449;
}

.result.negative {
    background: #fef2f2;
    color: #c0392b;
}

.probability {
    margin-top: 8px;
    font-size: 18px;
    font-weight: bold;
}

.error {
    margin-top: 20px;
    padding: 15px;
    background: #fdecea;
    color: #c0392b;
    border-radius: 6px;
}

</style>

</head>

<body>

<div class="container">

<h1>Online Purchase Intention Predictor</h1>

<p class="subtitle">
Gradient Boosting model for predicting online purchase intention
</p>

<form action="/predict" method="POST">

    <div class="field">
        <label>Administrative Pages</label>
        <input type="number"
               name="Administrative"
               min="0"
               step="1"
               required>
    </div>

    <div class="field">
        <label>Administrative Duration</label>
        <input type="number"
               name="Administrative_Duration"
               min="0"
               step="any"
               required>
    </div>

    <div class="field">
        <label>Informational Pages</label>
        <input type="number"
               name="Informational"
               min="0"
               step="1"
               required>
    </div>

    <div class="field">
        <label>Informational Duration</label>
        <input type="number"
               name="Informational_Duration"
               min="0"
               step="any"
               required>
    </div>

    <div class="field">
        <label>Product Related Pages</label>
        <input type="number"
               name="ProductRelated"
               min="0"
               step="1"
               required>
    </div>

    <div class="field">
        <label>Product Related Duration</label>
        <input type="number"
               name="ProductRelated_Duration"
               min="0"
               step="any"
               required>
    </div>

    <div class="field">
        <label>Bounce Rate</label>
        <input type="number"
               name="BounceRates"
               min="0"
               max="1"
               step="any"
               required>
    </div>

    <div class="field">
        <label>Exit Rate</label>
        <input type="number"
               name="ExitRates"
               min="0"
               max="1"
               step="any"
               required>
    </div>

    <div class="field">
        <label>Page Value</label>
        <input type="number"
               name="PageValues"
               min="0"
               step="any"
               required>
    </div>

    <div class="field">
        <label>Special Day</label>
        <input type="number"
               name="SpecialDay"
               min="0"
               max="1"
               step="any"
               required>
    </div>

    <div class="field">
        <label>Month</label>

        <select name="Month" required>

            {% for value in month_values %}
            <option value="{{ value }}">
                {{ value }}
            </option>
            {% endfor %}

        </select>

    </div>

    <div class="field">
        <label>Operating System</label>

        <select name="OperatingSystems" required>

            {% for value in operating_system_values %}
            <option value="{{ value }}">
                {{ value }}
            </option>
            {% endfor %}

        </select>

    </div>

    <div class="field">
        <label>Browser</label>

        <select name="Browser" required>

            {% for value in browser_values %}
            <option value="{{ value }}">
                {{ value }}
            </option>
            {% endfor %}

        </select>

    </div>

    <div class="field">
        <label>Region</label>

        <select name="Region" required>

            {% for value in region_values %}
            <option value="{{ value }}">
                {{ value }}
            </option>
            {% endfor %}

        </select>

    </div>

    <div class="field">
        <label>Traffic Type</label>

        <select name="TrafficType" required>

            {% for value in traffic_type_values %}
            <option value="{{ value }}">
                {{ value }}
            </option>
            {% endfor %}

        </select>

    </div>

    <div class="field">
        <label>Visitor Type</label>

        <select name="VisitorType" required>

            {% for value in visitor_type_values %}
            <option value="{{ value }}">
                {{ value }}
            </option>
            {% endfor %}

        </select>

    </div>

    <div class="field">
        <label>Weekend</label>

        <select name="Weekend" required>

            <option value="0">No</option>
            <option value="1">Yes</option>

        </select>

    </div>

    <div class="submit-row">

        <button type="submit">
            Predict Purchase Intention
        </button>

    </div>

</form>


{% if result %}

<div class="result {{ 'positive' if result.prediction == 1 else 'negative' }}">

    <strong>
        {{ result.label }}
    </strong>

    <div class="probability">
        Purchase Probability:
        {{ result.probability }}%
    </div>

</div>

{% endif %}


{% if error %}

<div class="error">
    Error: {{ error }}
</div>

{% endif %}

</div>

</body>

</html>
"""


# ==========================================================
# PREPARE INPUT
# ==========================================================

def prepare_input(form_data):

    input_data = pd.DataFrame({

        'Administrative': [
            int(form_data['Administrative'])
        ],

        'Administrative_Duration': [
            float(form_data['Administrative_Duration'])
        ],

        'Informational': [
            int(form_data['Informational'])
        ],

        'Informational_Duration': [
            float(form_data['Informational_Duration'])
        ],

        'ProductRelated': [
            int(form_data['ProductRelated'])
        ],

        'ProductRelated_Duration': [
            float(form_data['ProductRelated_Duration'])
        ],

        'BounceRates': [
            float(form_data['BounceRates'])
        ],

        'ExitRates': [
            float(form_data['ExitRates'])
        ],

        'PageValues': [
            float(form_data['PageValues'])
        ],

        'SpecialDay': [
            float(form_data['SpecialDay'])
        ],

        'Month': [
            form_data['Month']
        ],

        'OperatingSystems': [
            str(form_data['OperatingSystems'])
        ],

        'Browser': [
            str(form_data['Browser'])
        ],

        'Region': [
            str(form_data['Region'])
        ],

        'TrafficType': [
            str(form_data['TrafficType'])
        ],

        'VisitorType': [
            form_data['VisitorType']
        ],

        'Weekend': [
            int(form_data['Weekend'])
        ]

    })

    return input_data


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/", methods=["GET"])
def home():

    return render_template_string(

        HTML_FORM,

        month_values=list(
            ordinal_encoder.categories_[0]
        ),

        operating_system_values=list(
            ordinal_encoder.categories_[1]
        ),

        browser_values=list(
            ordinal_encoder.categories_[2]
        ),

        region_values=list(
            ordinal_encoder.categories_[3]
        ),

        traffic_type_values=list(
            ordinal_encoder.categories_[4]
        ),

        visitor_type_values=list(
            ordinal_encoder.categories_[5]
        )

    )


# ==========================================================
# WEB PREDICTION
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        input_df = prepare_input(
            request.form
        )


        input_encoded = input_df.copy()


        input_encoded[categorical_features] = (
            ordinal_encoder.transform(
                input_df[categorical_features]
            )
        )


        input_processed = preprocessor.transform(
            input_encoded
        )


        prediction = int(
            gradient_boosting.predict(
                input_processed
            )[0]
        )


        probability = float(
            gradient_boosting.predict_proba(
                input_processed
            )[0][1]
        )


        result = {

            "prediction": prediction,

            "label": (
                "Likely to Purchase"
                if prediction == 1
                else
                "Unlikely to Purchase"
            ),

            "probability": round(
                probability * 100,
                2
            )

        }


        return render_template_string(

            HTML_FORM,

            month_values=list(
                ordinal_encoder.categories_[0]
            ),

            operating_system_values=list(
                ordinal_encoder.categories_[1]
            ),

            browser_values=list(
                ordinal_encoder.categories_[2]
            ),

            region_values=list(
                ordinal_encoder.categories_[3]
            ),

            traffic_type_values=list(
                ordinal_encoder.categories_[4]
            ),

            visitor_type_values=list(
                ordinal_encoder.categories_[5]
            ),

            result=result

        )


    except Exception as e:

        return render_template_string(

            HTML_FORM,

            month_values=list(
                ordinal_encoder.categories_[0]
            ),

            operating_system_values=list(
                ordinal_encoder.categories_[1]
            ),

            browser_values=list(
                ordinal_encoder.categories_[2]
            ),

            region_values=list(
                ordinal_encoder.categories_[3]
            ),

            traffic_type_values=list(
                ordinal_encoder.categories_[4]
            ),

            visitor_type_values=list(
                ordinal_encoder.categories_[5]
            ),

            error=str(e)

        )


# ==========================================================
# API PREDICTION
# ==========================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():

    try:

        data = request.get_json(
            force=True
        )

        input_df = prepare_input(
            data
        )


        input_encoded = input_df.copy()


        input_encoded[categorical_features] = (
            ordinal_encoder.transform(
                input_df[categorical_features]
            )
        )


        input_processed = preprocessor.transform(
            input_encoded
        )


        prediction = int(
            gradient_boosting.predict(
                input_processed
            )[0]
        )


        probability = float(
            gradient_boosting.predict_proba(
                input_processed
            )[0][1]
        )


        return jsonify({

            "prediction": prediction,

            "label": (
                "Likely to Purchase"
                if prediction == 1
                else
                "Unlikely to Purchase"
            ),

            "probability": round(
                probability * 100,
                2
            )

        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400
"""
ExtraaLearn Lead Conversion Prediction API
============================================
A minimal, production-oriented Flask API that serves the serialized
Random Forest (Tuned) lead-conversion pipeline produced in the project
notebook. The loaded object is the COMPLETE fitted sklearn Pipeline
(preprocessing ColumnTransformer + trained Random Forest classifier),
so this file never re-implements any encoding/scaling logic — raw lead
fields are handed straight to the pipeline.
"""

import os
import traceback

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the model ONCE at application startup (not per-request)
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "extraalearn_lead_conversion_model.joblib",
)

model = joblib.load(MODEL_PATH)

# Determine the index of the positive ("Converted" / status == 1) class safely
# from the fitted classifier, rather than assuming column 1 is always correct.
_classifier = model.named_steps["model"]
POSITIVE_CLASS_INDEX = list(_classifier.classes_).index(1)

# ---------------------------------------------------------------------------
# Expected raw input schema (matches the actual ExtraaLearn.csv columns,
# excluding the identifier `ID` and the target `status`)
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = [
    "age",
    "current_occupation",
    "first_interaction",
    "profile_completed",
    "website_visits",
    "time_spent_on_website",
    "page_views_per_visit",
    "last_activity",
    "print_media_type1",
    "print_media_type2",
    "digital_media",
    "educational_channels",
    "referral",
]

NUMERIC_FIELDS = {"age", "website_visits", "time_spent_on_website", "page_views_per_visit"}

# Note: categorical fields are intentionally NOT restricted to a fixed allow-list
# here. The pipeline's OneHotEncoder was fit with handle_unknown="ignore", so an
# unexpected category is safely encoded as all-zeros rather than raising an error.
# Enforcing a stricter allow-list in the API would conflict with that design.


@app.route("/", methods=["GET"])
def home():
    """Simple health/root endpoint."""
    return jsonify({
        "message": "ExtraaLearn Lead Conversion Prediction API is running",
        "status": "healthy",
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if data is None:
            return jsonify({"error": "Request body must contain valid JSON data"}), 400

        if not isinstance(data, dict):
            return jsonify({"error": "JSON payload must be an object mapping field names to values"}), 400

        # --- Validate required fields are present ---
        missing_fields = [field for field in REQUIRED_FIELDS if field not in data]
        if missing_fields:
            return jsonify({
                "error": "Missing required field(s)",
                "missing_fields": missing_fields,
            }), 400

        # --- Validate numeric fields can actually be converted to numbers ---
        invalid_numeric_fields = []
        cleaned_data = dict(data)
        for field in NUMERIC_FIELDS:
            try:
                cleaned_data[field] = float(cleaned_data[field])
            except (TypeError, ValueError):
                invalid_numeric_fields.append(field)

        if invalid_numeric_fields:
            return jsonify({
                "error": "Field(s) could not be interpreted as numeric",
                "invalid_numeric_fields": invalid_numeric_fields,
            }), 400

        # --- Build a single-row DataFrame with the RAW feature values ---
        # Only the expected columns are passed through, in a consistent order.
        input_df = pd.DataFrame([{field: cleaned_data[field] for field in REQUIRED_FIELDS}])

        # --- Predict using the complete pipeline (preprocessing handled internally) ---
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        conversion_probability = float(round(probabilities[POSITIVE_CLASS_INDEX], 4))

        response = {
            "prediction": prediction,
            "prediction_label": "Converted" if prediction == 1 else "Not Converted",
            "conversion_probability": conversion_probability,
        }

        return jsonify(response), 200

    except Exception:
        # Log the full traceback server-side for debugging, but never expose
        # internal stack traces or implementation details to the client.
        app.logger.error("Unhandled error during prediction:\n%s", traceback.format_exc())
        return jsonify({"error": "An unexpected error occurred while generating the prediction."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

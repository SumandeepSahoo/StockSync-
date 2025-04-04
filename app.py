from flask import Flask, request
from flask_cors import CORS
import analysis_module
import simplejson as json  # Using simplejson to ignore NaN values

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route("/api/run_analysis", methods=["POST"])
def run_analysis_api():
    try:
        data = request.get_json()
        symbol = data.get("symbol", "JPM").upper()
        start_date = data.get("startDate", "2024-01-01")
        end_date = data.get("endDate", "2025-01-01")
        results = analysis_module.run_analysis(symbol, start_date, end_date)
        response_json = json.dumps(results, ignore_nan=True)
        return app.response_class(response_json, mimetype="application/json")
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_json = json.dumps({"error": str(e)}, ignore_nan=True)
        return app.response_class(error_json, mimetype="application/json"), 500

if __name__ == "__main__":
    app.run(debug=True)

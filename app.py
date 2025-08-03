from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import mysql.connector
import os

from predict_face import predict_face
from predict_eye import predict_eye

app = Flask(__name__, static_folder='public')
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ✅ MySQL configuration
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "Huli@123",
    "database": "health_scanner_db"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def store_result_in_mysql(scan_type, result, filename):
    conn = None
    cursor = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        query = """
        INSERT INTO scan_results
        (scan_type, region, prediction, confidence, confidence_level, inference_time_sec, filename)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            scan_type,
            result["region"],
            result["prediction"],
            result["confidence"],
            result["confidence_level"],
            result["inference_time_sec"],
            filename
        )
        cursor.execute(query, values)
        conn.commit()
    except Exception as e:
        print("❌ DB insert error:", str(e))
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.route('/')
def home():
    return send_from_directory('public', 'index.html')
@app.route('/history', methods=['GET'])
def get_scan_history():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM scan_results ORDER BY timestamp DESC LIMIT 20")
        results = cursor.fetchall()
        return jsonify(results)
    except Exception as e:
        print("❌ History fetch error:", str(e))
        return jsonify({"error": str(e)}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

@app.route('/predict_face', methods=['POST'])
def predict_from_api():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    scan_type = request.form.get("scan_type", "face")

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        if scan_type == "face":
            result = predict_face(file_path)
        elif scan_type == "eye":
            result = predict_eye(file_path)
        else:
            raise ValueError("Invalid scan type. Use 'face' or 'eye'.")

        # ✅ Store result in MySQL
        store_result_in_mysql(scan_type, result, filename)

        response = {
            "region": result["region"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "confidence_level": result["confidence_level"],
            "predicted_classes": result["predicted_classes"],
            "confidences": result["confidences"],
            "top_predictions": result["top_predictions"],
            "all_probabilities": result["all_probabilities"],
            "inference_time_sec": result["inference_time_sec"]
        }

        return jsonify(response)

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.errorhandler(404)
def not_found(e):
    return send_from_directory('public', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
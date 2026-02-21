# ---- Ocultar logs y warnings ----
import os
import warnings
import sys
import contextlib
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

@contextlib.contextmanager
def suppress_tf_warnings():
    stderr = sys.stderr
    with open(os.devnull, 'w') as fnull:
        sys.stderr = fnull
        yield
    sys.stderr = stderr

# ---- Librerías principales ----
import mysql.connector
from minio import Minio
import numpy as np
from io import BytesIO
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ---- Configuración Flask ----
app = Flask(__name__)
CORS(app)

# ---- Logging ----
# Crea archivo de logs en el mismo directorio
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
app.logger.addHandler(logging.StreamHandler(sys.stdout))

# ---- Configuración general ----
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# ---- Configuraciones mediante variables de entorno ----
# Puedes definir estas en Azure App Service > Configuration > Application Settings
MINIO_URL = os.getenv("MINIO_URL", "minio.conani.gob.do")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "conani-dev")
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "Conani01+")
BUCKET_NAME = os.getenv("MINIO_BUCKET", "sirenna-dev")
PREFIX = os.getenv("MINIO_PREFIX", "profile_photos/")

DB_HOST = os.getenv("DB_HOST", "mysql-pro01.mysql.database.azure.com")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_DATABASE = os.getenv("DB_DATABASE", "sirennaDB")
DB_USERNAME = os.getenv("DB_USERNAME", "sirennauser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Conani01+")

# ---- Cliente MinIO ----
client = Minio(
    MINIO_URL,
    access_key=ACCESS_KEY,
    secret_key=SECRET_KEY,
    secure=True
)

# ---- Importar DeepFace sin warnings ----
with suppress_tf_warnings():
    from deepface import DeepFace

# ---- Funciones auxiliares ----
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def connect_db():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USERNAME,
            password=DB_PASSWORD,
            database=DB_DATABASE
        )
        return conn
    except mysql.connector.Error as err:
        app.logger.error(f"Error al conectar a MySQL: {err}")
        return None

def are_faces_similar(img1_path, img2_bytes, threshold=0.4):
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
            tmp_img.write(img2_bytes.getvalue())
            tmp_path = tmp_img.name

        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=tmp_path,
            model_name="Facenet",
            enforce_detection=False
        )

        os.unlink(tmp_path)

        distance = result["distance"]
        similarity = max(0.0, (1.0 - distance)) * 100
        return result["verified"] and distance < threshold, similarity
    except Exception as e:
        app.logger.error(f"Error comparando rostros: {e}")
        return False, 0.0

def find_matching_images(file_path, threshold=0.4):
    matches = []

    for obj in client.list_objects(BUCKET_NAME, prefix=PREFIX, recursive=True):
        try:
            response = client.get_object(BUCKET_NAME, obj.object_name)
            img_data = BytesIO(response.read())

            is_match, similarity = are_faces_similar(file_path, img_data, threshold)
            if is_match:
                matches.append({
                    "file_path": obj.object_name,
                    "similarity": round(similarity, 2)
                })
        except Exception as e:
            app.logger.error(f"Error al procesar imagen {obj.object_name}: {e}")

    return matches

def fetch_image_data(matches):
    db = connect_db()
    if not db:
        return []

    cursor = db.cursor(dictionary=True)
    results = []

    for match in matches:
        try:
            query = """
            SELECT 
                n.id,
                CONCAT(n.name, ' ', n.surname) AS nombre_completo,
                d.document_path
            FROM nna_document d
            JOIN nna n ON d.NNA_ID = n.id
            WHERE d.document_path = %s
            """
            cursor.execute(query, (match["file_path"],))
            result = cursor.fetchone()

            if result:
                results.append({
                    "nna_id": result["id"],
                    "nombre_completo": result["nombre_completo"],
                    "document_path": result["document_path"],
                    "similarity": match["similarity"]
                })
        except mysql.connector.Error as err:
            app.logger.error(f"Error en la consulta: {err}")

    cursor.close()
    db.close()
    return results

# ---- Endpoints ----
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "Facial Recognition API is running"
    }), 200

@app.route('/compare', methods=['POST'])
def compare_faces():
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "message": "No se envió ninguna imagen"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"success": False, "message": "Nombre de archivo vacío"}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "message": f"Tipo de archivo no permitido. Usa: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        threshold = float(request.form.get('threshold', 0.4))

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            file.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            app.logger.info(f"Buscando coincidencias para: {file.filename}")
            matches = find_matching_images(tmp_path, threshold)

            if not matches:
                return jsonify({
                    "success": True,
                    "matches_count": 0,
                    "matches": [],
                    "message": "No se encontraron coincidencias"
                }), 200

            results = fetch_image_data(matches)
            return jsonify({
                "success": True,
                "matches_count": len(results),
                "matches": results,
                "message": f"Se encontraron {len(results)} coincidencia(s)"
            }), 200

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        app.logger.error(f"Error en /compare: {str(e)}")
        return jsonify({"success": False, "message": f"Error interno: {str(e)}"}), 500

@app.route('/compare-by-id', methods=['POST'])
def compare_by_nna_id():
    try:
        data = request.get_json()
        if not data or 'nna_id' not in data:
            return jsonify({"success": False, "message": "Se requiere el parámetro 'nna_id'"}), 400

        nna_id = data['nna_id']
        threshold = float(data.get('threshold', 0.4))

        db = connect_db()
        if not db:
            return jsonify({"success": False, "message": "Error al conectar con la base de datos"}), 500

        cursor = db.cursor(dictionary=True)
        query = "SELECT document_path FROM nna_document WHERE NNA_ID = %s LIMIT 1"
        cursor.execute(query, (nna_id,))
        result = cursor.fetchone()
        cursor.close()
        db.close()

        if not result:
            return jsonify({"success": False, "message": f"No se encontró foto para el NNA ID: {nna_id}"}), 404

        document_path = result['document_path']
        response = client.get_object(BUCKET_NAME, document_path)
        img_data = response.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(img_data)
            tmp_path = tmp_file.name

        try:
            matches = find_matching_images(tmp_path, threshold)
            results = fetch_image_data(matches)
            results = [r for r in results if r['nna_id'] != nna_id]

            return jsonify({
                "success": True,
                "nna_id": nna_id,
                "matches_count": len(results),
                "matches": results,
                "message": f"Se encontraron {len(results)} coincidencia(s) diferentes al NNA consultado"
            }), 200

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        app.logger.error(f"Error en /compare-by-id: {str(e)}")
        return jsonify({"success": False, "message": f"Error interno: {str(e)}"}), 500


# ---- NOTA ----
# No se incluye app.run() porque en Azure se ejecutará con:
# gunicorn app:app --timeout 600 --workers 1

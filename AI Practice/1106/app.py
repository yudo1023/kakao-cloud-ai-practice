from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import magic
from PIL import Image
import hashlib
from datetime import datetime
from google.cloud import vision
import psycopg2
import json

app = Flask(__name__)

# 디렉토리 설정
UPLOAD_FOLDER = 'uploads'
THUMBNAIL_FOLDER = 'thumbnails'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(THUMBNAIL_FOLDER, exist_ok=True)

# 파일 제한
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
ALLOWED_MIME_TYPES = {'image/png', 'image/jpeg', 'image/bmp', 'image/webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['THUMBNAIL_FOLDER'] = THUMBNAIL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

vision_client = vision.ImageAnnotatorClient()

# PostgreSQL 연결 설정
DB_CONFIG = {
    'host': 'localhost',
    'database': 'ocr_db',
    'user': 'postgres',
    'password': 'pass1234',
    'port': 5432
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS ocr_results (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255),
            file_path VARCHAR(500),
            thumbnail_path VARCHAR(500),
            ocr_text TEXT,
            ocr_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    cur.close()
    conn.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def verify_mime_type(file_path):
    mime = magic.Magic(mime=True)
    return mime.from_file(file_path) in ALLOWED_MIME_TYPES

def generate_unique_filename(filename):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    hash_str = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
    ext = filename.rsplit('.', 1)[1].lower()
    return f"{timestamp}_{hash_str}.{ext}"

def create_thumbnail(image_path, thumbnail_path, size=(300, 300)):
    try:
        with Image.open(image_path) as img:
            img.thumbnail(size)
            img.save(thumbnail_path)
        return True
    except Exception:
        return False


# Google Vision OCR
def perform_google_ocr(path):
    try:
        with open(path, "rb") as img_file:
            content = img_file.read()

        image = vision.Image(content=content)
        response = vision_client.text_detection(image=image)
        texts = response.text_annotations

        if not texts:
            return {"text": ""}

        return {"text": texts[0].description.strip()}

    except Exception as e:
        return {"error": str(e)}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "파일이 없습니다."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "허용되지 않는 파일 형식입니다."}), 400

    original_filename = secure_filename(file.filename)
    unique_filename = generate_unique_filename(original_filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(file_path)

    if not verify_mime_type(file_path):
        os.remove(file_path)
        return jsonify({"error": "유효하지 않은 이미지 파일입니다."}), 400

    thumbnail_filename = f"thumb_{unique_filename}"
    thumbnail_path = os.path.join(app.config['THUMBNAIL_FOLDER'], thumbnail_filename)
    create_thumbnail(file_path, thumbnail_path)

    ocr_result = perform_google_ocr(file_path)
    # DB 저장
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO ocr_results (filename, file_path, thumbnail_path, ocr_text, ocr_json)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        ''', (
            original_filename,
            file_path,
            thumbnail_path,
            ocr_result.get("text", ""),
            json.dumps(ocr_result, ensure_ascii=False)
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        return jsonify({"error": f"DB 저장 실패: {e}"}), 500

    return jsonify({
        "success": True,
        "ocr_result": {"text": ocr_result.get("text", "")}
    })

@app.route('/uploads/<filename>')
def get_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/thumbnails/<filename>')
def get_thumbnail(filename):
    return send_from_directory(app.config['THUMBNAIL_FOLDER'], filename)

@app.route('/api/ocr', methods=['POST'])
def api_ocr():
    if 'image' not in request.files:
        return jsonify({"error": "이미지 파일이 필요합니다."}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "파일 이름이 비어 있습니다."}), 400
    if not allowed_file(image_file.filename):
        return jsonify({"error": "허용되지 않는 파일 형식입니다."}), 400

    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], generate_unique_filename(image_file.filename))
    image_file.save(temp_path)

    if not verify_mime_type(temp_path):
        os.remove(temp_path)
        return jsonify({"error": "유효하지 않은 이미지 파일입니다."}), 400

    ocr_result = perform_google_ocr(temp_path)
    os.remove(temp_path)

    return jsonify({"success": True, "result": ocr_result})


if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)

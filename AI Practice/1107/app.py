from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
import easyocr
import pdf2image, os, sqlite3, cv2, re, hashlib
import numpy as np
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import json
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import base64
from io import BytesIO
from datetime import datetime

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['SECRET_KEY'] = '1234'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "db.sqlite3")
UPLOAD_PATH = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_PATH, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_PATH

# ✅ EasyOCR 초기화 (한글 + 영문)
print("🚀 EasyOCR 모델 로드 중... (처음 실행 시 시간이 걸립니다)")
reader = easyocr.Reader(['ko', 'en'], gpu=False)
print("✅ EasyOCR 준비 완료")

# -----------------------------
# 🔐 로그인 데코레이터
# -----------------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "로그인이 필요합니다"}), 401
        return f(*args, **kwargs)
    return decorated_function

# -----------------------------
# 💾 DB 관련 함수
# -----------------------------
def init_db():
    """데이터베이스 초기화"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 사용자 테이블
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 계약서 테이블 (user_id 추가 + file_hash)
    c.execute('''
        CREATE TABLE IF NOT EXISTS contracts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            category TEXT NOT NULL,
            text TEXT,
            summary TEXT,
            ocr_data TEXT,
            image_path TEXT,
            file_hash TEXT,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    
    conn.commit()
    conn.close()
    print("✅ 데이터베이스 초기화 완료")

def save_contract(user_id, filename, category, text, summary, ocr_data, image_path):
    """계약서 정보를 DB에 저장"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    summary_json = json.dumps(summary, ensure_ascii=False) if summary else None
    ocr_json = json.dumps(ocr_data, ensure_ascii=False) if ocr_data else None
    c.execute(
        "INSERT INTO contracts (user_id, filename, category, text, summary, ocr_data, image_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (user_id, filename, category, text, summary_json, ocr_json, image_path)
    )
    conn.commit()
    conn.close()
    print(f"✅ DB 저장 완료: {filename} ({category})")

def get_contracts(user_id, category=None):
    """저장된 계약서 목록 조회 (사용자별)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    if category:
        c.execute("SELECT * FROM contracts WHERE user_id=? AND category=? ORDER BY upload_time DESC", (user_id, category))
    else:
        c.execute("SELECT * FROM contracts WHERE user_id=? ORDER BY upload_time DESC", (user_id,))

    rows = c.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        item = dict(row)
        if item.get('summary'):
            try:
                item['summary'] = json.loads(item['summary'])
            except:
                item['summary'] = None
        if item.get('ocr_data'):
            try:
                item['ocr_data'] = json.loads(item['ocr_data'])
            except:
                item['ocr_data'] = None
        result.append(item)
    
    return result

# -----------------------------
# 🔐 인증 라우트
# -----------------------------
@app.route('/register', methods=['POST'])
def register():
    """회원가입"""
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    email = data.get('email', '').strip()
    
    if not username or not password:
        return jsonify({"error": "아이디와 비밀번호를 입력해주세요"}), 400
    
    if len(password) < 4:
        return jsonify({"error": "비밀번호는 최소 4자 이상이어야 합니다"}), 400
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # 중복 체크
        c.execute("SELECT id FROM users WHERE username=?", (username,))
        if c.fetchone():
            conn.close()
            return jsonify({"error": "이미 존재하는 아이디입니다"}), 400
        
        # 비밀번호 해시화
        hashed_pw = generate_password_hash(password)
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                  (username, hashed_pw, email))
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        
        # 자동 로그인
        session['user_id'] = user_id
        session['username'] = username
        
        return jsonify({"success": True, "username": username})
    except Exception as e:
        print(f"회원가입 오류: {e}")
        return jsonify({"error": "회원가입 중 오류가 발생했습니다"}), 500

@app.route('/login', methods=['POST'])
def login():
    """로그인"""
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    
    if not username or not password:
        return jsonify({"error": "아이디와 비밀번호를 입력해주세요"}), 400
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=?", (username,))
        user = c.fetchone()
        conn.close()
        
        if not user or not check_password_hash(user['password'], password):
            return jsonify({"error": "아이디 또는 비밀번호가 일치하지 않습니다"}), 401
        
        session['user_id'] = user['id']
        session['username'] = user['username']
        
        return jsonify({"success": True, "username": user['username']})
    except Exception as e:
        print(f"로그인 오류: {e}")
        return jsonify({"error": "로그인 중 오류가 발생했습니다"}), 500

@app.route('/logout', methods=['POST'])
def logout():
    """로그아웃"""
    session.clear()
    return jsonify({"success": True})

@app.route('/check-auth', methods=['GET'])
def check_auth():
    """로그인 상태 확인"""
    if 'user_id' in session:
        return jsonify({"logged_in": True, "username": session.get('username')})
    return jsonify({"logged_in": False})

# -----------------------------
# 🧠 카테고리 자동 분류
# -----------------------------
def detect_category(text):
    """텍스트 내용을 분석하여 카테고리 자동 분류"""
    CATEGORY_KEYWORDS = {
        "웨딩홀": ["웨딩홀", "웨딩", "예식", "식음료", "홀대관", "피로연", "컨벤션", "연출", "대관료", "예식장"],
        "헤어메이크업": ["헤어", "메이크업", "드라이", "아티스트", "메컵", "분장", "스타일"],
        "스냅": ["촬영", "스냅", "앨범", "포토", "원판", "사진", "본식촬영", "야외촬영"],
        "드레스": ["드레스", "피팅", "본식드레스", "리허설", "예복", "턱시도", "웨딩드레스"],
    }

    text_lower = text.lower().replace(" ", "")
    scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        matches = 0
        for kw in keywords:
            pattern = re.compile(kw.replace(" ", "").lower())
            matches += len(pattern.findall(text_lower))
        scores[category] = matches

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "기타"

# -----------------------------
# 📄 주요 내용 추출 (대폭 개선)
# -----------------------------
def extract_summary(text, category=None):
    """텍스트에서 주요 정보 추출 (심플 & 개선 버전)"""
    if not text or len(text.strip()) < 10:
        return None
    
    print(f"\n{'='*60}")
    print(f"📋 정보 추출 시작 (카테고리: {category})")
    summary = {}

    # ========== 1️⃣ 업체명 ==========
    company_patterns = [
        # 명시적 라벨이 있는 경우 (최우선)
        r'(?:업체명|상호|회사명|업체\s*명|사업자명)\s*[:：]?\s*([가-힣A-Za-z0-9\s]{2,30})(?=\s|$|\n)',
        # 대표자명 (간결한 한글 이름)
        r'(?:대표자|대표|성명|이름)\s*[:：]?\s*([가-힣]{2,4})(?=\s|$|\n)',
        # 업종 키워드 포함
        r'([가-힣A-Za-z]{2,20}(?:스튜디오|웨딩|드레스|메이크업|헤어|샵|하우스|홀|필름|포토|그라피|사진관))',
        # 영문 업체명
        r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s*(?:STUDIO|WEDDING|SNAP|FILM|PHOTO)',
    ]
    
    # 제외할 패턴 (업체명이 아닌 것들)
    exclude_patterns = [
        r'(간의|합의|의해|환불|가능|변경|불특정|연기|경우|상호|계약|조건|사항|서비스|이용|확약|약정|동의)',
        r'(제\d+조|제\s*\d+\s*조)',
        r'(다음과|같이|위와|아래|있는|없는|한다|된다)',
        r'(홀$|샵$)',  # 단독으로 "홀", "샵"만 있는 경우
    ]
    
    found_company = None
    for pattern in company_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for m in matches:
            # 마지막 그룹 추출
            name = m.group(m.lastindex if m.lastindex > 1 else 1).strip()
            name = re.sub(r'^\s*(주식회사|유한회사|\(주\))\s*', '', name).strip()
            
            # 제외 패턴 체크
            is_valid = True
            for exclude in exclude_patterns:
                if re.search(exclude, name):
                    is_valid = False
                    break
            
            # 길이 및 조사로 끝나는지 체크
            if is_valid and 2 <= len(name) <= 30 and not name.endswith(('다', '가', '이', '을', '를', '은', '는')):
                found_company = name
                break
        
        if found_company:
            summary['company'] = found_company
            print(f"🏢 업체명: {found_company}")
            break

    # ========== 2️⃣ 금액 ==========
    t = text.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1').replace('|', '1')
    amount_patterns = [
        r'(?:금액|계약금|결제금액)\s*[:：]?\s*(\d{1,3}(?:[,，]\d{3})+)',
        r'(\d{1,3}(?:[,，]\d{3})+)\s*원',
        r'(\d+)\s*만\s*원'
    ]
    for pattern in amount_patterns:
        m = re.search(pattern, t)
        if m:
            val = m.group(1).replace(',', '').replace('，', '')
            if '만' in m.group(0): val = str(int(val) * 10000)
            amount = int(val)
            if 10000 <= amount <= 100000000:
                summary['amount'] = f"{amount:,}원"
                print(f"💰 금액: {summary['amount']}")
                break

    # ========== 3️⃣ 날짜 ==========
    date_patterns = [
        r'(?:예식일|촬영일|계약일|예약일|행사일)\s*[:：]?\s*(\d{4})[.\-/년\s]*(\d{1,2})[.\-/월\s]*(\d{1,2})[일]?',
        r'(\d{4})[.\-/년\s]+(\d{1,2})[.\-/월\s]+(\d{1,2})[일\s]',
        r'(\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})'
    ]
    for pattern in date_patterns:
        m = re.search(pattern, text)
        if m:
            y, mth, d = m.groups()
            if len(y) == 2: y = '20' + y
            try:
                year, month, day = int(y), int(mth), int(d)
                # 유효성 검사
                if 2020 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                    dt = datetime(year, month, day)
                    summary['date'] = dt.strftime("%Y년 %m월 %d일")
                    print(f"📅 날짜: {summary['date']}")
                    break
            except:
                continue

    # ========== 4️⃣ 연락처 ==========
    phone_pattern = r'(0\d{1,2}[-\s]?\d{3,4}[-\s]?\d{4})'
    m = re.search(phone_pattern, text)
    if m:
        summary['phone'] = m.group(1)
        print(f"📞 연락처: {summary['phone']}")

    # ========== 5️⃣ 이메일 ==========
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    m = re.search(email_pattern, text)
    if m:
        summary['email'] = m.group(0)
        print(f"📧 이메일: {summary['email']}")

    # ========== 6️⃣ 사업자등록번호 ==========
    biz_pattern = r'(\d{3}[-\s]?\d{2}[-\s]?\d{5})'
    m = re.search(biz_pattern, text)
    if m:
        summary['business_number'] = m.group(1)
        print(f"🏷️ 사업자번호: {summary['business_number']}")

    # ========== 7️⃣ 주소 ==========
    loc_patterns = [
        r'(?:주소|소재지|위치|장소|촬영\s*장소|예식\s*장소)\s*[:：]?\s*([가-힣]+(?:특별시|광역시|시|도)\s+[^\n\r]{10,80})',
        r'([가-힣]+(?:특별시|광역시|시|도)\s+[가-힣]+(?:구|군)\s+[가-힣]+(?:동|로|길)\s+\d+[^\n\r]{0,40})'
    ]
    for pattern in loc_patterns:
        m = re.search(pattern, text)
        if m:
            loc = m.group(1).strip()
            # 제외 패턴 (주소가 아닌 것) - 더 엄격하게
            exclude_in_loc = r'(변경|연기|경우|합의|환불|확약|약정|동의|서비스|이용자|올\s*확약)'
            if len(loc) >= 15 and not re.search(exclude_in_loc, loc):
                summary['location'] = loc
                print(f"📍 주소: {loc}")
                break

    # ========== 결과 요약 ==========
    print(f"\n📊 추출 결과: {summary if summary else '⚠️ 추출 실패'}")
    print(f"{'='*60}\n")
    return summary if summary else None

def convert_korean_number_to_int(korean_num):
    """한글 숫자를 정수로 변환"""
    units = {'십': 10, '백': 100, '천': 1000, '만': 10000, '억': 100000000}
    digits = {'일': 1, '이': 2, '삼': 3, '사': 4, '오': 5, '육': 6, '칠': 7, '팔': 8, '구': 9}
    
    result = 0
    temp = 0
    
    for char in korean_num:
        if char in digits:
            temp = digits[char]
        elif char in units:
            if temp == 0:
                temp = 1
            if units[char] >= 10000:
                result = (result + temp) * units[char]
                temp = 0
            else:
                temp *= units[char]
                result += temp
                temp = 0
    
    return result + temp

# -----------------------------
# 🖼️ OCR with 좌표 정보 (전처리 개선)
# -----------------------------
def convert_to_serializable(obj):
    """NumPy 타입을 Python 기본 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

def preprocess_image_for_ocr(image_np):
    """OCR을 위한 다단계 이미지 전처리"""
    try:
        # 1. Grayscale 변환
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # 2. 노이즈 제거 (약하게)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)
        
        # 3. 대비 향상 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast = clahe.apply(denoised)
        
        # 4. 샤프닝 (텍스트 경계 강조)
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(contrast, -1, kernel)
        
        # 5. 이진화 (Otsu's method)
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 6. 모폴로지 연산 (작은 노이즈 제거)
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    except Exception as e:
        print(f"전처리 오류: {e}")
        return image_np

def extract_text_with_coords(image_np):
    """EasyOCR로 텍스트와 좌표 추출 (다단계 전처리)"""
    try:
        print("\n=== EasyOCR 실행 ===")
        
        # 1차 시도: 원본 이미지
        print("🔍 1차 시도: 원본 이미지")
        result = reader.readtext(image_np, detail=1)
        print(f"   인식된 텍스트 블록 수: {len(result)}")
        
        # 2차 시도: 전처리 이미지 (결과가 적을 경우)
        if len(result) < 10:
            print("🔍 2차 시도: 전처리 이미지 (대비 향상 + 샤프닝 + 이진화)")
            processed = preprocess_image_for_ocr(image_np)
            result2 = reader.readtext(processed, detail=1)
            print(f"   인식된 텍스트 블록 수: {len(result2)}")
            
            # 더 많이 인식된 결과 선택
            if len(result2) > len(result):
                print("   ✅ 전처리 버전 선택")
                result = result2
        
        # 3차 시도: 적응형 이진화 (여전히 적을 경우)
        if len(result) < 10:
            print("🔍 3차 시도: 적응형 이진화")
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np
            
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            result3 = reader.readtext(adaptive, detail=1)
            print(f"   인식된 텍스트 블록 수: {len(result3)}")
            
            if len(result3) > len(result):
                print("   ✅ 적응형 이진화 버전 선택")
                result = result3
        
        print(f"\n📊 최종 선택: {len(result)}개 텍스트 블록")
        
        text_parts = []
        ocr_data = []
        
        for idx, (bbox, text, prob) in enumerate(result):
            text_parts.append(text)
            
            # 처음 5개만 상세 로그
            if idx < 5:
                print(f"[{idx}] '{text}' (신뢰도: {prob:.2f})")
            
            # bbox를 Python 기본 리스트로 변환
            if isinstance(bbox, np.ndarray):
                bbox_list = bbox.tolist()
            else:
                bbox_list = [[float(point[0]), float(point[1])] for point in bbox]
            
            ocr_data.append({
                "text": str(text),
                "bbox": bbox_list,
                "confidence": float(prob)
            })
        
        full_text = '\n'.join(text_parts)
        print(f"\n총 추출 텍스트 길이: {len(full_text)}자")
        print("=== EasyOCR 완료 ===\n")
        
        return full_text, ocr_data
    except Exception as e:
        print(f"❌ EasyOCR 오류: {e}")
        import traceback
        traceback.print_exc()
        return "", []

# PDF 페이지 처리 (좌표 포함)
def pdf_page_to_text_with_coords(pdf_path, page_num):
    """PDF 페이지에서 텍스트와 좌표 추출 (최소 전처리)"""
    try:
        images = convert_from_path(pdf_path, dpi=300, first_page=page_num, last_page=page_num)
        
        all_text = ""
        all_ocr_data = []
        
        for img in images:
            img_np = np.array(img)
            # 원본 이미지 직접 사용 (전처리 최소화)
            page_text, ocr_data = extract_text_with_coords(img_np)
            all_text += page_text + " "
            all_ocr_data.extend(ocr_data)
        
        return all_text, all_ocr_data
    except Exception as e:
        print(f"PDF 페이지 처리 오류: {e}")
        return "", []

# 이미지 파일 처리 (좌표 포함)
def image_to_text_with_coords(image_path):
    """이미지 파일에서 텍스트와 좌표 추출 (최소 전처리)"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "", []
        
        # 원본 이미지를 RGB로만 변환 (전처리 최소화)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        text, ocr_data = extract_text_with_coords(img_rgb)
        return text, ocr_data
    except Exception as e:
        print(f"이미지 OCR 오류: {e}")
        return "", []

# 이미지를 base64로 변환
def resize_image_for_display(pil_image, max_width=1200):
    """표시용 이미지 크기 조정"""
    width, height = pil_image.size
    
    if width > max_width:
        ratio = max_width / width
        new_width = max_width
        new_height = int(height * ratio)
        resized = pil_image.resize((new_width, new_height), Image.LANCZOS)
        print(f"   표시용 이미지 리사이즈: {width}x{height} → {new_width}x{new_height}")
        return resized, ratio
    
    return pil_image, 1.0

def scale_ocr_coordinates(ocr_data, scale_ratio):
    """OCR 좌표를 이미지 축소 비율에 맞게 조정"""
    if scale_ratio == 1.0:
        return ocr_data
    
    scaled_data = []
    for item in ocr_data:
        scaled_bbox = [[point[0] * scale_ratio, point[1] * scale_ratio] for point in item['bbox']]
        scaled_data.append({
            "text": item['text'],
            "bbox": scaled_bbox,
            "confidence": item['confidence']
        })
    return scaled_data
    """이미지를 base64 문자열로 변환"""
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return None

# 파일명 자동 정리
def generate_clean_filename(category, summary, original_filename):
    """카테고리, 요약 정보, 원본 파일명을 기반으로 깔끔한 파일명 생성"""
    from datetime import datetime
    
    # 현재 날짜 (등록일)
    today = datetime.now().strftime('%Y%m%d')
    
    # 업체명 추출 (요약에서)
    company = ""
    if summary and summary.get('company'):
        company = summary['company']
        # 파일명에 사용할 수 없는 문자 제거
        company = re.sub(r'[\\/:*?"<>|]', '', company)
        company = company.replace(' ', '_')
        company = company[:20]  # 최대 20자
    
    # 카테고리 약자
    category_short = {
        '웨딩홀': '홀',
        '헤어메이크업': '메이크업',
        '스냅': '스냅',
        '드레스': '드레스',
        '기타': '기타'
    }.get(category, category)
    
    # 원본 파일 확장자
    file_ext = os.path.splitext(original_filename)[1].lower()
    
    # 새 파일명 생성
    if company:
        new_filename = f"{today}_{category_short}_{company}{file_ext}"
    else:
        # 업체명이 없으면 카테고리만
        new_filename = f"{today}_{category_short}{file_ext}"
    
    print(f"📝 파일명 변경: {original_filename} → {new_filename}")
    return new_filename

# -----------------------------
# 📤 파일 업로드 + OCR + 저장
# -----------------------------
@app.route('/upload', methods=['POST'])
@login_required
def upload():
    """파일 업로드 및 OCR 처리"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "파일이 없습니다"}), 400
        
        files = request.files.getlist('file')
        if len(files) > 1:
            return jsonify({"error": "한 번에 한 계약서만 업로드해주세요"}), 400
        
        file = files[0]
        if file.filename == '':
            return jsonify({"error": "파일이 선택되지 않았습니다"}), 400

        # 사용자별 폴더 생성
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(session['user_id']))
        os.makedirs(user_folder, exist_ok=True)

        # 일단 임시 저장
        temp_path = os.path.join(user_folder, file.filename)
        file.save(temp_path)
        print(f"📁 임시 파일 저장: {temp_path}")

        # ✅ 파일 내용 해시 계산
        import hashlib
        def calculate_file_hash(filepath):
            h = hashlib.sha256()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    h.update(chunk)
            return h.hexdigest()
        
        file_hash = calculate_file_hash(temp_path)
        print(f"🔑 파일 해시: {file_hash[:20]}...")

        # ✅ 동일한 해시가 이미 업로드된 경우 차단
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM contracts WHERE user_id=? AND file_hash=?", (session['user_id'], file_hash))
        exists = c.fetchone()[0]
        conn.close()

        if exists > 0:
            os.remove(temp_path)
            print("⚠️ 중복 파일 업로드 차단됨")
            return jsonify({"error": "이미 동일한 파일을 업로드하셨습니다."}), 400

        # ✅ 업로드 시각 기반 파일명 생성
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = os.path.splitext(file.filename)[1].lower()
        clean_filename = f"{now_str}{ext}"
        final_path = os.path.join(user_folder, clean_filename)
        os.rename(temp_path, final_path)
        print(f"📝 파일명 변경: {file.filename} → {clean_filename}")

        # OCR 처리
        file_ext = os.path.splitext(clean_filename)[1].lower()
        full_text, ocr_data, image_base64 = "", [], None

        if file_ext == ".pdf":
            print("📄 PDF 처리 중...")
            reader_pdf = PdfReader(final_path)
            # DPI를 500으로 더 높여서 선명한 이미지 생성
            images = convert_from_path(final_path, dpi=500, first_page=1, last_page=1)
            pdf_image = images[0]
            # 원본 이미지 사용 (전처리는 OCR 함수에서)
            pdf_image_np = np.array(pdf_image)
            full_text, ocr_data = extract_text_with_coords(pdf_image_np)

            # 표시용 이미지는 크기 축소 + OCR 좌표도 조정
            display_image, scale_ratio = resize_image_for_display(pdf_image, max_width=1200)
            ocr_data = scale_ocr_coordinates(ocr_data, scale_ratio)
            
            buffered = BytesIO()
            display_image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        elif file_ext in [".png", ".jpg", ".jpeg"]:
            print("🖼️ 이미지 처리 중...")
            img = cv2.imread(final_path)
            if img is None:
                return jsonify({"error": "이미지를 읽을 수 없습니다"}), 400

            # 이미지 크기가 작으면 확대 (OCR 정확도 향상)
            height, width = img.shape[:2]
            if width < 2000 or height < 2000:
                scale = max(2000 / width, 2000 / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                print(f"   이미지 확대: {width}x{height} → {new_width}x{new_height}")

            # 원본 이미지 사용 (전처리는 OCR 함수에서)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            full_text, ocr_data = extract_text_with_coords(img_rgb)

            # 표시용 이미지는 크기 축소 + OCR 좌표도 조정
            pil_image = Image.fromarray(img_rgb)
            display_image, scale_ratio = resize_image_for_display(pil_image, max_width=1200)
            ocr_data = scale_ocr_coordinates(ocr_data, scale_ratio)
            
            buffered = BytesIO()
            display_image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            return jsonify({"error": "지원하지 않는 파일 형식입니다"}), 400

        if not full_text.strip():
            return jsonify({"error": "파일에서 텍스트를 추출할 수 없습니다"}), 400

        # ✅ 카테고리 분류 및 요약
        category = detect_category(full_text)
        summary = extract_summary(full_text, category)

        # ✅ DB 저장
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        summary_json = json.dumps(summary, ensure_ascii=False) if summary else None
        ocr_json = json.dumps(ocr_data, ensure_ascii=False) if ocr_data else None
        c.execute(
            """
            INSERT INTO contracts (user_id, filename, category, text, summary, ocr_data, image_path, file_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session['user_id'], clean_filename, category, full_text, summary_json, ocr_json, final_path, file_hash)
        )
        conn.commit()
        conn.close()
        print(f"✅ DB 저장 완료: {clean_filename} ({category})")

        display_text = full_text.strip()[:5000]

        return jsonify({
            "filename": clean_filename,
            "category": category,
            "text": display_text,
            "summary": summary,
            "ocr_data": ocr_data,
            "image": image_base64
        })

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"처리 중 오류: {str(e)}"}), 500

# -----------------------------
# 📋 계약서 목록 조회
# -----------------------------
@app.route('/contracts', methods=['GET'])
@login_required
def list_contracts():
    """저장된 계약서 목록 반환"""
    category = request.args.get('category', '').strip()
    data = get_contracts(session['user_id'], category if category else None)
    return jsonify(data)

# -----------------------------
# 🔍 계약서 검색
# -----------------------------
@app.route('/search', methods=['GET'])
@login_required
def search_contracts():
    """계약서 검색"""
    query = request.args.get('q', '').strip().lower()
    
    if not query:
        return jsonify([])
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    search_term = f"%{query}%"
    c.execute('''
        SELECT * FROM contracts 
        WHERE user_id=? AND (
            LOWER(filename) LIKE ? 
            OR LOWER(text) LIKE ? 
            OR LOWER(category) LIKE ?
        )
        ORDER BY upload_time DESC
    ''', (session['user_id'], search_term, search_term, search_term))
    
    rows = c.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        item = dict(row)
        if item.get('summary'):
            try:
                item['summary'] = json.loads(item['summary'])
            except:
                item['summary'] = None
        result.append(item)
    
    return jsonify(result)

# -----------------------------
# 🏠 메인 페이지
# -----------------------------
@app.route('/')
def index():
    """메인 페이지 렌더링"""
    return render_template('index.html')

# -----------------------------
# 🚀 앱 실행
# -----------------------------
if __name__ == '__main__':
    init_db()
    print("=" * 50)
    print("🎉 결혼 준비 계약서 관리 시스템 시작!")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
# app_payment.py
import qrcode
from io import BytesIO
from flask import send_file
from flask import render_template_string
import os
import re
import uuid
import traceback
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv


import mysql.connector
from mysql.connector import Error


from PIL import Image
import pytesseract


_cv2_available = True
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None
    _cv2_available = False

# TrOCR disabled - using Tesseract only for faster performance
_use_trocr = False

# Optional Twilio
_twilio_available = True
try:
    from twilio.rest import Client as TwilioClient
except Exception:
    TwilioClient = None
    _twilio_available = False

# ----------------------------
# Load .env
# ----------------------------
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "") or ""
DB_NAME = os.getenv("DB_NAME", "vending_app")

TW_SID = os.getenv("TW_SID", "") or ""
TW_TOKEN = os.getenv("TW_TOKEN", "") or ""
TW_NUMBER = os.getenv("TW_NUMBER", "") or ""

# If Windows and tesseract not in PATH, uncomment and adjust:
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------------------
# App init
# ----------------------------
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------
# DB helper
# ----------------------------
def get_conn():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        autocommit=False
    )

# ----------------------------
# Twilio init (optional)
# ----------------------------
twilio_client = None
if TwilioClient and TW_SID and TW_TOKEN:
    try:
        twilio_client = TwilioClient(TW_SID, TW_TOKEN)
    except Exception as e:
        print("Twilio init error:", e)
        twilio_client = None

def send_sms(phone, text):
    if not twilio_client:
        print("Twilio not configured - WhatsApp skipped:", text)
        return None
    try:
        p = phone.strip()
        p = re.sub(r'\D', '', p)
        if p.startswith('0'):
            p = '2' + p
        if not p.startswith('+'):
            p = '+' + p
        wa_to = 'whatsapp:' + p
        wa_from = "whatsapp:+14155238886"
        msg = twilio_client.messages.create(body=text, from_=wa_from, to=wa_to)
        print("WhatsApp sent SID:", msg.sid)
        return msg.sid
    except Exception as e:
        print("Twilio WhatsApp send error:", e)
        return None

# ----------------------------
# DB Init (create tables)
# ----------------------------
def init_db():
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS pins (
            id INT AUTO_INCREMENT PRIMARY KEY,
            plain_pin VARCHAR(10) NOT NULL,
            used TINYINT(1) DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY (plain_pin)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INT AUTO_INCREMENT PRIMARY KEY,
            order_id VARCHAR(100) UNIQUE,
            product VARCHAR(200),
            customer_phone VARCHAR(50),
            amount DECIMAL(10,2),
            status VARCHAR(20) DEFAULT 'PENDING',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS payments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            order_id VARCHAR(100),
            img_path VARCHAR(255),
            sender_phone VARCHAR(50),
            amount DECIMAL(10,2),
            ref_code VARCHAR(200),
            raw_text TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS claims (
            id INT AUTO_INCREMENT PRIMARY KEY,
            pin_used VARCHAR(10),
            phone VARCHAR(50),
            machine_id VARCHAR(100),
            assigned_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        cur.close()
    except Exception as e:
        print("init_db error:", e)
    finally:
        if conn:
            conn.close()

init_db()

# ----------------------------
# TrOCR lazy-loading helpers
# ----------------------------
_trocr_processor = None
_trocr_model = None

def ensure_trocr_loaded():
    global _trocr_processor, _trocr_model
    if not _use_trocr:
        return False
    if _trocr_model is None or _trocr_processor is None:
        try:
            print("Loading TrOCR model (this may take time)...")
            _trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            _trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            return True
        except Exception as e:
            print("Failed to load TrOCR:", e)
            return False
    return True

def trocr_ocr(image_path):
    if not _use_trocr:
        return ""
    if not ensure_trocr_loaded():
        return ""
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = _trocr_processor(images=image, return_tensors="pt").pixel_values
        generated_ids = _trocr_model.generate(pixel_values)
        text = _trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text
    except Exception as e:
        print("trocr_ocr error:", e)
        return ""

# ----------------------------
# Heuristic fake/edited image detection
# ----------------------------
def detect_fake_image(img_path):
    reasons = []
    if not _cv2_available:
        return False, ["opencv_not_installed"]
    try:
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return True, ["cannot_read_image"]
        h, w = img.shape[:2]
        if h < 200 or w < 200:
            reasons.append("low_resolution")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 20:
            reasons.append("low_detail_or_blur")
        return (len(reasons) > 0), reasons
    except Exception as e:
        print("detect_fake_image error:", e)
        return True, ["error"]

# ----------------------------
# OCR extraction (Tesseract)
# ----------------------------
def extract_payment_data_tesseract(image_path):
    try:
        try:
            raw = pytesseract.image_to_string(Image.open(image_path), lang="ara+eng")
        except Exception:
            raw = pytesseract.image_to_string(Image.open(image_path), lang="eng")
        raw = raw.strip()
        print("----- OCR RAW TEXT -----")
        print(raw)
        print("------------------------")
        trans_table = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
        raw_norm = raw.translate(trans_table)

        # amount patterns
        amount = None
        m = re.search(r"تم\s*استلام\s*مبلغ\s*[:,]?\s*([0-9]+(?:[.,][0-9]+)?)", raw_norm, re.IGNORECASE)
        if not m:
            m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*(?:جنيه|EGP|ج\.?م|LE|L\.E)", raw_norm, re.IGNORECASE)
        if m:
            amt_str = m.group(1).replace(",", ".")
            try:
                amount = float(amt_str)
            except:
                amount = None

        # phone pattern egyptian
        phone = None
        phone_match = re.search(r"(\+?2?0?1[0125][0-9]{8})", raw_norm)
        if phone_match:
            p = re.sub(r"\D", "", phone_match.group(1))
            if p.startswith("01") and len(p) == 11:
                phone = "+2" + p
            elif p.startswith("201") and len(p) == 12:
                phone = "+" + p
            elif p.startswith("2") and len(p) == 12:
                phone = "+" + p
            else:
                phone = "+" + p

        # ref code
        ref_code = None
        ref_m = re.search(r"رقم\s*العملية\s*[:,]?\s*(\d{6,20})", raw_norm)
        if ref_m:
            ref_code = ref_m.group(1)
        else:
            all_digits = re.findall(r"\b\d{6,20}\b", raw_norm)
            if all_digits:
                amt_str = None
                if amount is not None:
                    amt_str = str(int(amount)) if float(amount).is_integer() else str(amount).replace(".", "")
                for d in all_digits:
                    if amt_str and (amt_str in d or d == amt_str):
                        continue
                    ref_code = d
                    break

        return {"amount": amount, "sender_phone": phone, "ref_code": ref_code, "raw_text": raw_norm}
    except Exception as e:
        print("extract_payment_data_tesseract error:", e)
        return None

def extract_payment_data(image_path):
    t_res = extract_payment_data_tesseract(image_path) or {}
    if _use_trocr:
        try:
            trocr_text = trocr_ocr(image_path)
            if trocr_text and len(trocr_text.strip()) > 5:
                combined_raw = (t_res.get("raw_text") or "") + "\n" + trocr_text
                # simple re-parse
                m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*(?:جنيه|EGP|LE)", combined_raw, re.IGNORECASE)
                amount = float(m.group(1).replace(",", ".")) if m else t_res.get("amount")
                pm = re.search(r"(\+?2?0?1[0125][0-9]{8})", combined_raw)
                phone = None
                if pm:
                    p = re.sub(r"\D", "", pm.group(1))
                    if p.startswith("01") and len(p) == 11:
                        phone = "+2" + p
                    elif p.startswith("201") and len(p) == 12:
                        phone = "+" + p
                    else:
                        phone = "+" + p
                ref_m = re.search(r"رقم\s*العملية\s*[:,]?\s*(\d{6,20})", combined_raw)
                ref_code = ref_m.group(1) if ref_m else t_res.get("ref_code")
                return {"amount": amount, "sender_phone": phone, "ref_code": ref_code, "raw_text": combined_raw}
        except Exception as e:
            print("trocr merge error:", e)
    return t_res

# ----------------------------
# Assign pin transactionally
# ----------------------------
def assign_pin_to_order(conn, order_row, machine_id=None):
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT id, plain_pin FROM pins WHERE used=0 ORDER BY id ASC LIMIT 1 FOR UPDATE")
        pin = cur.fetchone()
        if not pin:
            cur.close()
            return None
        pin_id = pin["id"]; pin_code = pin["plain_pin"]
        cur.execute("UPDATE pins SET used=1 WHERE id=%s", (pin_id,))
        cur.execute("INSERT INTO claims (pin_used, phone, machine_id, assigned_at) VALUES (%s,%s,%s,%s)",
                    (pin_code, order_row["customer_phone"], machine_id, datetime.now()))
        cur.execute("UPDATE orders SET status='COMPLETE' WHERE order_id=%s", (order_row["order_id"],))
        conn.commit()
        cur.close()
        return pin_code
    except Exception as e:
        print("assign_pin_to_order error:", e)
        try:
            conn.rollback()
        except:
            pass
        return None

# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"msg": "Vending Payment API running with OCR & AI (if enabled)"}), 200

# Create order
@app.route("/create_order", methods=["POST"])
def create_order():
    try:
        data = request.get_json() or {}
        phone = data.get("phone") or data.get("customer_phone")
        product = data.get("product") or data.get("product_name")
        amount = data.get("amount") or data.get("price")
        if not phone or not product or amount is None:
            return jsonify({"error": "Missing fields: phone/product/amount"}), 400
        order_id = str(uuid.uuid4())
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("INSERT INTO orders (order_id, product, customer_phone, amount) VALUES (%s,%s,%s,%s)",
                    (order_id, product, phone, float(amount)))
        conn.commit()
        cur.close(); conn.close()
        return jsonify({"order_id": order_id})
    except Error as e:
        print("create_order DB error:", e)
        return jsonify({"error": "DB error", "details": str(e)}), 500
    except Exception as e:
        print("create_order error:", e)
        return jsonify({"error": "Server error", "details": str(e)}), 500

# Upload payment (runs OCR)
@app.route("/upload_payment", methods=["POST"])
def upload_payment():
    try:
        order_id = request.form.get("order_id")
        file = request.files.get("file")
        if not order_id or not file:
            return jsonify({"error": "Missing order_id or file"}), 400

        safe_name = f"{order_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        saved_path = os.path.join(UPLOAD_FOLDER, safe_name)
        file.stream.seek(0)
        with open(saved_path, "wb") as f:
            f.write(file.stream.read())

        is_suspect, reasons = detect_fake_image(saved_path)
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT order_id, amount, customer_phone, status FROM orders WHERE order_id=%s", (order_id,))
        row = cur.fetchone()
        if not row:
            cur.close(); conn.close()
            return jsonify({"error": "Invalid order_id"}), 400
        expected_amount = float(row[1]) if row[1] is not None else None

        cur.execute("INSERT INTO payments (order_id, img_path, created_at) VALUES (%s,%s,%s)",
                    (order_id, saved_path, datetime.now()))
        conn.commit()

        if is_suspect:
            cur.execute("SELECT id FROM payments WHERE order_id=%s ORDER BY id DESC LIMIT 1", (order_id,))
            payrow = cur.fetchone()
            pay_id = payrow[0] if payrow else None
            if pay_id:
                cur.execute("UPDATE payments SET raw_text=%s WHERE id=%s", (f"FAKE_DETECT:{','.join(reasons)}", pay_id))
                conn.commit()
            cur.close(); conn.close()
            return jsonify({"status": "uploaded", "note": "image_suspect", "reasons": reasons}), 200

        ocr = extract_payment_data(saved_path)
        if not ocr:
            cur.execute("SELECT id FROM payments WHERE order_id=%s ORDER BY id DESC LIMIT 1", (order_id,))
            payrow = cur.fetchone()
            pay_id = payrow[0] if payrow else None
            if pay_id:
                cur.execute("UPDATE payments SET raw_text=%s WHERE id=%s", ("OCR_FAILED", pay_id))
                conn.commit()
            cur.close(); conn.close()
            return jsonify({"status": "uploaded", "note": "ocr_failed"}), 200

        # --- Security: check duplicate ref_code ---
        extracted_ref = ocr.get("ref_code")
        if extracted_ref:
            cur2 = conn.cursor(buffered=True)
            cur2.execute("SELECT id FROM payments WHERE ref_code=%s", (extracted_ref,))
            existing_ref = cur2.fetchone()
            cur2.close()
            if existing_ref:
                cur.close(); conn.close()
                return jsonify({"error": "duplicate_ref_code", "note": "هذا الإيصال مستخدم من قبل ولا يمكن إعادة استخدامه"}), 400

        cur.execute("SELECT id FROM payments WHERE order_id=%s ORDER BY id DESC LIMIT 1", (order_id,))
        payrow = cur.fetchone()
        pay_id = payrow[0] if payrow else None
        cur.execute("UPDATE payments SET sender_phone=%s, amount=%s, ref_code=%s, raw_text=%s WHERE id=%s",
                    (ocr.get("sender_phone"), ocr.get("amount"), ocr.get("ref_code"), ocr.get("raw_text"), pay_id))
        conn.commit()

        auto_approve = False
        if expected_amount is not None and ocr.get("amount") is not None:
            try:
                ocr_amt = float(ocr["amount"])
                if abs(expected_amount - ocr_amt) <= 0.5:
                    auto_approve = True
            except:
                auto_approve = False

        result = {"status": "uploaded", "ocr": ocr, "auto_approve": auto_approve}

        if auto_approve:
            cur.execute("SELECT order_id, customer_phone, amount FROM orders WHERE order_id=%s", (order_id,))
            r = cur.fetchone()
            order_row = {"order_id": r[0], "customer_phone": r[1], "amount": r[2]}
            pin_code = assign_pin_to_order(conn, order_row, machine_id=None)
            if pin_code:
                send_sms(order_row["customer_phone"], f"Your vending machine PIN is: {pin_code}")
                result["status"] = "auto_approved"
                result["pin"] = pin_code
            else:
                result["note"] = "no_pins_available"

        cur.close(); conn.close()
        return jsonify(result), 200

    except Error as e:
        print("upload_payment db error:", e)
        traceback.print_exc()
        return jsonify({"error": "Server DB error", "details": str(e)}), 500
    except Exception as e:
        print("upload_payment error:", e)
        traceback.print_exc()
        return jsonify({"error": "Server error", "details": str(e)}), 500

# Admin endpoints
@app.route("/admin/pending_payments", methods=["GET"])
def admin_pending_payments():
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT o.order_id, o.product, o.amount, o.customer_phone, o.status,
                   p.id AS payment_id, p.img_path, p.sender_phone, p.amount AS ocr_amount, p.ref_code, p.raw_text
            FROM orders o
            LEFT JOIN payments p ON o.order_id = p.order_id
            WHERE o.status='PENDING'
            ORDER BY o.created_at DESC
        """)
        rows = cur.fetchall()
        cur.close(); conn.close()
        return jsonify(rows), 200
    except Exception as e:
        print("admin_pending_payments error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/admin/approve", methods=["POST"])
def admin_approve():
    try:
        data = request.get_json() or {}
        order_id = data.get("order_id")
        machine_id = data.get("machine_id")
        if not order_id:
            return jsonify({"error": "Missing order_id"}), 400
        conn = get_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT order_id, customer_phone, amount FROM orders WHERE order_id=%s", (order_id,))
        order = cur.fetchone()
        if not order:
            cur.close(); conn.close()
            return jsonify({"error": "Order not found"}), 404
        pin_code = assign_pin_to_order(conn, order, machine_id=machine_id)
        cur.close(); conn.close()
        if not pin_code:
            return jsonify({"error": "No pins available"}), 500
        send_sms(order["customer_phone"], f"Your vending machine PIN is: {pin_code}")
        return jsonify({"status": "APPROVED", "pin": pin_code}), 200
    except Exception as e:
        print("admin_approve error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# NEW: Admin reject route
@app.route("/admin/reject", methods=["POST"])
def admin_reject():
    try:
        data = request.get_json() or {}
        order_id = data.get("order_id")
        reason = data.get("reason", "")
        if not order_id:
            return jsonify({"error": "Missing order_id"}), 400
        conn = get_conn()
        cur = conn.cursor()
        # set order status to REJECTED
        cur.execute("UPDATE orders SET status=%s WHERE order_id=%s", ("REJECTED", order_id))
        # update last payment raw_text with rejection note
        cur.execute("SELECT id FROM payments WHERE order_id=%s ORDER BY id DESC LIMIT 1", (order_id,))
        r = cur.fetchone()
        if r:
            pid = r[0]
            note = f"REJECTED: {reason}" if reason else "REJECTED"
            cur.execute("UPDATE payments SET raw_text=%s WHERE id=%s", (note, pid))
        conn.commit()
        cur.close(); conn.close()
        return jsonify({"status": "REJECTED", "order_id": order_id}), 200
    except Exception as e:
        print("admin_reject error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/admin/pins", methods=["GET"])
def admin_pins():
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT id, plain_pin, used, created_at FROM pins ORDER BY id ASC")
        rows = cur.fetchall()
        cur.close(); conn.close()
        return jsonify(rows), 200
    except Exception as e:
        print("admin_pins error:", e)
        return jsonify({"error": str(e)}), 500
    
    # -------------------- GET PIN (ESP32 polls this) --------------------
@app.route("/get_pin", methods=["GET"])
def get_pin():
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT c.pin_used FROM claims c
            WHERE c.acked = 0
            ORDER BY c.assigned_at DESC LIMIT 1
        """)
        row = cur.fetchone()
        cur.close(); conn.close()
        if row:
            return jsonify({"pin": row["pin_used"]}), 200
        return jsonify({"pin": None}), 200
    except Exception as e:
        print("get_pin error:", e)
        return jsonify({"error": str(e)}), 500

# -------------------- ACK PIN (ESP32 calls after door opens) --------------------
@app.route("/ack_pin", methods=["POST"])
def ack_pin():
    try:
        data = request.get_json() or {}
        pin = data.get("pin")
        if not pin:
            return jsonify({"error": "missing pin"}), 400
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("UPDATE claims SET acked=1 WHERE pin_used=%s", (pin,))
        conn.commit()
        cur.close(); conn.close()
        return jsonify({"status": "acked"}), 200
    except Exception as e:
        print("ack_pin error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
@app.route("/pay/<order_id>", methods=["GET"])
def pay_page(order_id):
    html = """
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Smart Vending Payment</title>
        <style>
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                min-height: 100vh;
                font-family: Arial, sans-serif;
                background: linear-gradient(135deg, #06111f, #0b1f35);
                color: #e8f4f0;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }

            .box {
                width: 100%;
                max-width: 430px;
                background: #0d1829;
                border: 1px solid rgba(0, 229, 160, 0.25);
                border-radius: 22px;
                padding: 28px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.45);
            }

            .logo {
                width: 64px;
                height: 64px;
                margin: 0 auto 16px;
                border-radius: 18px;
                background: linear-gradient(135deg, #00e5a0, #0ea5e9);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 30px;
            }

            h2 {
                text-align: center;
                color: #00e5a0;
                margin-bottom: 8px;
            }

            .subtitle {
                text-align: center;
                color: #8aa8b5;
                font-size: 14px;
                margin-bottom: 24px;
                line-height: 1.6;
            }

            .order {
                background: #111f35;
                border: 1px solid rgba(0, 229, 160, 0.15);
                border-radius: 14px;
                padding: 14px;
                margin-bottom: 20px;
                font-size: 13px;
                word-break: break-all;
            }

            .order span {
                color: #00e5a0;
                font-weight: bold;
            }

            .upload-zone {
                border: 2px dashed rgba(0, 229, 160, 0.35);
                border-radius: 16px;
                background: rgba(0, 229, 160, 0.06);
                padding: 26px 16px;
                text-align: center;
                cursor: pointer;
                margin-bottom: 18px;
            }

            .upload-zone:hover {
                background: rgba(0, 229, 160, 0.12);
            }

            .upload-icon {
                font-size: 42px;
                margin-bottom: 10px;
            }

            .upload-text {
                color: #e8f4f0;
                font-weight: bold;
                margin-bottom: 6px;
            }

            .upload-hint {
                color: #8aa8b5;
                font-size: 13px;
            }

            input[type="file"] {
                display: none;
            }

            button {
                width: 100%;
                padding: 15px;
                border: none;
                border-radius: 14px;
                background: linear-gradient(135deg, #00e5a0, #00b889);
                color: #001a10;
                font-size: 17px;
                font-weight: bold;
                cursor: pointer;
                box-shadow: 0 8px 24px rgba(0,229,160,0.3);
            }

            button:active {
                transform: scale(0.98);
            }

            .footer {
                text-align: center;
                margin-top: 18px;
                color: #6b8fa0;
                font-size: 12px;
            }

            .steps {
                margin-top: 20px;
                background: #081322;
                border-radius: 14px;
                padding: 14px;
                color: #8aa8b5;
                font-size: 13px;
                line-height: 1.8;
            }

            .steps b {
                color: #00e5a0;
            }
        </style>
    </head>

    <body>
        <div class="box">
            <div class="logo">🥤</div>

            <h2>رفع إيصال الدفع</h2>
            <p class="subtitle">
                ارفع سكرين شوت الدفع بعد تحويل المبلغ، وسيتم فحص الإيصال تلقائيًا.
            </p>

            <div class="order">
                <span>Order ID:</span><br>
                {{ order_id }}
            </div>

            <form action="/upload_payment_web" method="POST" enctype="multipart/form-data">
                <input type="hidden" name="order_id" value="{{ order_id }}">

                <label class="upload-zone" for="fileInput">
                    <div class="upload-icon">📸</div>
                    <div class="upload-text" id="fileText">اضغط لاختيار سكرين شوت الدفع</div>
                    <div class="upload-hint">PNG / JPG / JPEG</div>
                </label>

                <input id="fileInput" type="file" name="file" accept="image/*" required>

                <button type="submit">رفع الإيصال وفحص الدفع</button>
            </form>

            <div class="steps">
                <div><b>1.</b> ادفع المبلغ المطلوب</div>
                <div><b>2.</b> التقط سكرين شوت واضح</div>
                <div><b>3.</b> ارفع الصورة هنا</div>
                <div><b>4.</b> بعد الموافقة سيظهر لك PIN</div>
            </div>

            <div class="footer">
                Smart Vending Payment Portal
            </div>
        </div>

        <script>
            const fileInput = document.getElementById("fileInput");
            const fileText = document.getElementById("fileText");

            fileInput.addEventListener("change", function() {
                if (fileInput.files.length > 0) {
                    fileText.innerText = "تم اختيار: " + fileInput.files[0].name;
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html, order_id=order_id)
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Upload Payment</title>
        <style>
            body { font-family: Arial; padding: 30px; background: #f5f5f5; }
            .box { background: white; padding: 25px; border-radius: 12px; max-width: 420px; margin: auto; }
            input, button { width: 100%; padding: 12px; margin-top: 12px; }
            button { background: #111; color: white; border: none; border-radius: 8px; }
        </style>
    </head>
    <body>
        <div class="box">
            <h2>Upload Payment Screenshot</h2>
            <p><b>Order ID:</b> {{ order_id }}</p>

            <form action="/upload_payment_web" method="POST" enctype="multipart/form-data">
                <input type="hidden" name="order_id" value="{{ order_id }}">
                <label>Payment Screenshot:</label>
                <input type="file" name="file" accept="image/*" required>
                <button type="submit">Upload Screenshot</button>
            </form>
        </div>
    </body>
    </html>
    """
    return render_template_string(html, order_id=order_id)
@app.route("/upload_payment_web", methods=["POST"])
def upload_payment_web():
    try:
        order_id = request.form.get("order_id")
        file = request.files.get("file")

        if not order_id or not file:
            return "Missing order_id or file", 400

        safe_name = f"{order_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        saved_path = os.path.join(UPLOAD_FOLDER, safe_name)

        file.stream.seek(0)
        with open(saved_path, "wb") as f:
            f.write(file.stream.read())

        is_suspect, reasons = detect_fake_image(saved_path)

        conn = get_conn()
        cur = conn.cursor()

        cur.execute("SELECT order_id, amount, customer_phone, status FROM orders WHERE order_id=%s", (order_id,))
        row = cur.fetchone()

        if not row:
            cur.close()
            conn.close()
            return "Invalid order_id", 400

        expected_amount = float(row[1]) if row[1] is not None else None

        cur.execute(
            "INSERT INTO payments (order_id, img_path, created_at) VALUES (%s,%s,%s)",
            (order_id, saved_path, datetime.now())
        )
        conn.commit()

        if is_suspect:
            cur.close()
            conn.close()
            return f"""
            <h2>Payment uploaded but image is suspect</h2>
            <p>Reasons: {', '.join(reasons)}</p>
            <p>Please upload a clearer screenshot.</p>
            """

        ocr = extract_payment_data(saved_path)

        if not ocr:
            cur.close()
            conn.close()
            return """
            <h2>OCR failed</h2>
            <p>Please upload a clearer payment screenshot.</p>
            """

        extracted_ref = ocr.get("ref_code")

        if extracted_ref:
            cur2 = conn.cursor(buffered=True)
            cur2.execute("SELECT id FROM payments WHERE ref_code=%s", (extracted_ref,))
            existing_ref = cur2.fetchone()
            cur2.close()

            if existing_ref:
                cur.close()
                conn.close()
                return """
                <h2>Rejected</h2>
                <p>This receipt was used before.</p>
                """

        cur.execute("SELECT id FROM payments WHERE order_id=%s ORDER BY id DESC LIMIT 1", (order_id,))
        payrow = cur.fetchone()
        pay_id = payrow[0] if payrow else None

        cur.execute(
            "UPDATE payments SET sender_phone=%s, amount=%s, ref_code=%s, raw_text=%s WHERE id=%s",
            (ocr.get("sender_phone"), ocr.get("amount"), ocr.get("ref_code"), ocr.get("raw_text"), pay_id)
        )
        conn.commit()

        auto_approve = False

        if expected_amount is not None and ocr.get("amount") is not None:
            ocr_amt = float(ocr["amount"])
            if abs(expected_amount - ocr_amt) <= 0.5:
                auto_approve = True

        if auto_approve:
            cur.execute("SELECT order_id, customer_phone, amount FROM orders WHERE order_id=%s", (order_id,))
            r = cur.fetchone()

            order_row = {
                "order_id": r[0],
                "customer_phone": r[1],
                "amount": r[2]
            }

            pin_code = assign_pin_to_order(conn, order_row, machine_id=None)

            cur.close()
            conn.close()

            if pin_code:
                return f"""
                <html>
                <body style="font-family: Arial; text-align:center; padding:40px;">
                    <h2>Payment Approved ✅</h2>
                    <p>Your PIN is:</p>
                    <h1 style="font-size:48px;">{pin_code}</h1>
                    <p>Enter this PIN on the vending machine keypad.</p>
                </body>
                </html>
                """
            else:
                return """
                <h2>Payment approved</h2>
                <p>But no PINs available.</p>
                """

        cur.close()
        conn.close()

        return """
        <h2>Payment uploaded</h2>
        <p>Waiting for admin approval.</p>
        """

    except Exception as e:
        traceback.print_exc()
        return f"Server error: {str(e)}", 500
    
    
    
@app.route("/qr/<order_id>", methods=["GET"])
def generate_qr(order_id):
    pay_url = f"https://thriving-spirit-production-f1ac.up.railway.app/pay/{order_id}"

    img = qrcode.make(pay_url)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    return send_file(buffer, mimetype="image/png")
@app.route("/ocr_test", methods=["GET"])
def ocr_test():
    try:
        version = str(pytesseract.get_tesseract_version())
        langs = pytesseract.get_languages(config="")
        return jsonify({
            "tesseract_version": version,
            "languages": langs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Run
if __name__ == "__main__":
    print("Starting Vending Payment API...")
    try:
        t_exists = bool(pytesseract.get_tesseract_version())
    except Exception:
        t_exists = False
    print("Tesseract available:", t_exists)
    print("TrOCR enabled:", _use_trocr)
    print("OpenCV available:", _cv2_available)
    print("Twilio configured:", bool(twilio_client))
    print("DB:", DB_HOST, DB_USER, DB_NAME)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

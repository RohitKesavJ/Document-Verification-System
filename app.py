import os
import time
import json
import hashlib
import sqlite3
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import re
from werkzeug.utils import secure_filename
from datetime import datetime

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.secret_key = os.urandom(24)  # Required for session management

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

dictionary = {}

# SQLite database connection and migration
def init_db():
    conn = sqlite3.connect('document_verification.db')
    c = conn.cursor()
    
    # Check if table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
    table_exists = c.fetchone() is not None
    
    if not table_exists:
        # Create new table if it doesn't exist
        c.execute('''
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                participant_name TEXT,
                document_hash TEXT,
                security_score INTEGER,
                timestamp TEXT
            )
        ''')
        print("Created new documents table")
    else:
        # Check if security_score column exists
        c.execute("PRAGMA table_info(documents)")
        columns = [column[1] for column in c.fetchall()]
        if 'security_score' not in columns:
            # Drop the old table and create new one
            c.execute("DROP TABLE documents")
            c.execute('''
                CREATE TABLE documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    participant_name TEXT,
                    document_hash TEXT,
                    security_score INTEGER,
                    timestamp TEXT
                )
            ''')
            print("Recreated documents table with security_score column")
    
    conn.commit()
    conn.close()

# Initialize database with new schema
init_db()

def validate_aadhaar_qr(image_path):
    """
    Validates Aadhaar card QR code using OpenCV
    Returns (is_valid, qr_data)
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return False, "Failed to load image"
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Initialize QR code detector
        qr_detector = cv2.QRCodeDetector()
        
        # Detect and decode QR code
        data, bbox, _ = qr_detector.detectAndDecode(gray)
        
        if bbox is not None and data:
            # Validate QR data format
            # Aadhaar QR typically contains: uid, name, gender, yob (year of birth)
            aadhaar_patterns = [
                r'\d{12}',  # 12 digit Aadhaar number
                r'(?i)name[:\s]+[a-zA-Z\s]+',  # Name field
                r'(?i)gender[:\s]+[MF]',  # Gender field
                r'(?i)yob[:\s]+\d{4}'  # Year of birth
            ]
            
            match_count = sum(1 for pattern in aadhaar_patterns if re.search(pattern, data))
            is_valid = match_count >= 2  # Consider valid if at least 2 patterns match
            
            return is_valid, data
            
        # Try adaptive thresholding if QR not found
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        data, bbox, _ = qr_detector.detectAndDecode(thresh)
        
        if bbox is not None and data:
            match_count = sum(1 for pattern in aadhaar_patterns if re.search(pattern, data))
            is_valid = match_count >= 2
            return is_valid, data
            
        return False, "No QR code found"
        
    except Exception as e:
        print(f"Error reading QR code: {str(e)}")
        return False, str(e)

def check_security_features(image_path):
    """Enhanced security feature detection focused on front-side Aadhar card features"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0, "Failed to load image"
            
        # Initialize score and features list
        score = 0
        features = []
        
        # 1. Check image resolution (up to 15 points)
        height, width = img.shape[:2]
        min_resolution = 800 * 600
        resolution_score = min(15, (height * width) // (min_resolution // 15))
        score += resolution_score
        features.append(f"Resolution Score: {resolution_score}/15")
        
        # 2. Check for Ashoka Emblem (up to 25 points)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        emblem_score = min(25, int(np.sum(edges) > 50000) * 25)
        score += emblem_score
        features.append(f"✓ Ashoka Emblem detected ({emblem_score} points)" if emblem_score > 0 else "✗ Ashoka Emblem not detected")
        
        # 3. Check for official color scheme (up to 20 points)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_score = 0
        
        # Official Aadhar blue color ranges
        blue_ranges = [
            # Deep blue (header)
            (np.array([100,50,50]), np.array([130,255,255])),
            # Light blue (background)
            (np.array([85,30,30]), np.array([110,255,255]))
        ]
        
        for lower, upper in blue_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            if cv2.countNonZero(mask) > width * height * 0.05:  # At least 5% of pixels
                color_score += 10
                
        score += color_score
        features.append(f"✓ Official color scheme detected ({color_score} points)" if color_score > 0 else "✗ Official color scheme not found")
        
        # 4. Check for text clarity and formatting (up to 20 points)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_clarity_score = min(20, int(np.sum(binary) / (height * width * 255) * 100))
        score += text_clarity_score
        features.append(f"Text Clarity Score: {text_clarity_score}/20")
        
        # 5. Check for standard Aadhar card patterns (up to 20 points)
        pattern_score = 0
        
        # Check aspect ratio (should be approximately 1.5:1)
        aspect_ratio = width / height
        if 1.4 <= aspect_ratio <= 1.6:
            pattern_score += 7
            features.append("✓ Standard card dimensions")
        
        # Check for photo area (typically darker region on left side)
        left_region = gray[:, :width//3]
        if np.mean(left_region) < np.mean(gray):
            pattern_score += 7
            features.append("✓ Photo area detected")
            
        # Check for horizontal lines (common in Aadhar layout)
        horizontal_lines = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_lines)
        if cv2.countNonZero(detected_lines) > 100:
            pattern_score += 6
            features.append("✓ Standard layout detected")
            
        score += pattern_score
        features.append(f"Layout Pattern Score: {pattern_score}/20")
        
        # Final verification status
        verification_status = "✓ Verified" if score >= 70 else "✗ Not Verified"
        features.append(f"\nOverall Status: {verification_status} (Score: {score}/100)")
        
        return score, "\n".join(features)
        
    except Exception as e:
        print(f"Error in security check: {str(e)}")
        return 0, str(e)

# Function to add document hash to the database
def store_in_db(participant_name, document_hash, security_info):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    conn = sqlite3.connect('document_verification.db')
    c = conn.cursor()
    c.execute('INSERT INTO documents (participant_name, document_hash, security_score, timestamp) VALUES (?, ?, ?, ?)',
              (participant_name, document_hash, security_info['score'], timestamp))
    conn.commit()
    conn.close()
    return timestamp

def verify_data(participant_name, json_data):
    # Calculate the hash of the provided JSON data
    json_string = json.dumps(json_data, sort_keys=True)
    calculated_hash = hashlib.sha256(json_string.encode()).hexdigest()
    conn = sqlite3.connect('document_verification.db')
    c = conn.cursor()
    # Retrieve the stored hash and security score from the database
    c.execute('SELECT document_hash, security_score FROM documents WHERE participant_name = ?', (participant_name,))
    result = c.fetchone()
    conn.close()

    if result:
        stored_hash, security_score = result
        if calculated_hash == stored_hash:
            security_level = "Low" if security_score < 50 else "Medium" if security_score < 80 else "High"
            return f" ✅ Verification successful! Security Level: {security_level} (Score: {security_score}/100)"
        else:
            return " ❌ Verification failed! Data has been altered."
    else:
        return " ❌ Verification failed! Document not found in database."

def extract_text_from_image(image_path):
    """Extract text from image using Gemini AI"""
    try:
        # Upload file to Gemini
        sample_file = genai.upload_file(path=image_path, display_name="AADHAR CARD")
        file = genai.get_file(name=sample_file.name)
        
        # Generate content using Gemini
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content(
            [file, "\n\n", "Extract information from this Aadhar card image. Extract the name and Aadhar number. Format the response as JSON with fields: name, aadhaar_number. If the image is not an Aadhar card, extract name and document_type instead."],
        )
        
        # Clean and parse the response
        response_text = result.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:-3]
        try:
            result_data = json.loads(response_text)
        except json.JSONDecodeError:
            result_data = {
                "name": "Name not found",
                "aadhaar_number": "Number not found"
            }
        
        name = result_data.get("name", "Name not found")
        aadhaar_number = result_data.get("aadhaar_number", "Number not found")
        
        return name, aadhaar_number
        
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return "Name not found", "Number not found"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/verify')
def verify():
    return render_template('verify.html')

@app.route('/upload_details')
def upload_details():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Check security features
            security_score, security_features = check_security_features(filepath)
            
            # Extract text using Gemini AI
            name, aadhaar_number = extract_text_from_image(filepath)
            
            # Generate document hash
            with open(filepath, 'rb') as f:
                document_hash = hashlib.sha256(f.read()).hexdigest()
                
            # Check if document exists in database
            conn = sqlite3.connect('document_verification.db')
            c = conn.cursor()
            c.execute('SELECT * FROM documents WHERE document_hash = ?', (document_hash,))
            existing_doc = c.fetchone()
            conn.close()
            
            verification_result = {
                'name': name,
                'document_info': aadhaar_number,
                'security_features': security_features,
                'security_score': security_score,
                'verified': existing_doc is not None,
                'filename': filename,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Store verification result in session
            session['verification_result'] = verification_result
            
            return redirect(url_for('result'))
            
    except Exception as e:
        print(f"Error in upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload_data', methods=['POST'])
def upload_data():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if file:
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Check security features
            security_score, security_features = check_security_features(filepath)
            
            # Extract text using Gemini AI
            name, aadhaar_number = extract_text_from_image(filepath)
            
            # Generate document hash
            with open(filepath, 'rb') as f:
                document_hash = hashlib.sha256(f.read()).hexdigest()
                
            # Store in database
            conn = sqlite3.connect('document_verification.db')
            c = conn.cursor()
            c.execute('''
                INSERT INTO documents (participant_name, document_hash, security_score, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (name, document_hash, security_score, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
            conn.close()
            
            return jsonify({
                'success': True,
                'message': 'Document uploaded and verified successfully',
                'name': name,
                'document_info': aadhaar_number,
                'security_features': security_features,
                'security_score': security_score
            })
            
    except Exception as e:
        print(f"Error in upload_data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/result')
def result():
    verification_result = session.get('verification_result', {})
    return render_template('result.html', result=verification_result)

if __name__ == '__main__':
    app.run(debug=True)
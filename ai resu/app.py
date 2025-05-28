from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import json
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from werkzeug.utils import secure_filename
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
from fuzzywuzzy import fuzz
import re
from openai import OpenAI
from dotenv import load_dotenv
from flask import session
from flask_session import Session
from datetime import datetime


# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
app.secret_key = 'secret'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)




# In-memory visit tracking
visit_log = []
online_users = {}




global_ranked_results = []

job_data = []
job_title_classifier = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])



app.debug = True

def normalize_text(text):
    if isinstance(text, list):
        return [normalize_text(item) for item in text]
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return ' '.join(text.split())

def extract_keywords(text):
    """Extract important keywords from text without OpenAI"""
    keywords = set()
    # Extract capitalized words (likely proper nouns)
    keywords.update(re.findall(r'\b[A-Z][a-zA-Z]+\b', text))
    # Extract skills section if exists
    skills_match = re.search(r'skills:\s*(.*?)(?:\n\n|$)', text, re.IGNORECASE)
    if skills_match:
        keywords.update(s.strip() for s in skills_match.group(1).split(','))
    # Extract education qualifications
    education_match = re.search(r'education:\s*(.*?)(?:\n\n|$)', text, re.IGNORECASE)
    if education_match:
        keywords.update(s.strip() for s in education_match.group(1).split(','))
    return list(keywords)

def generate_ai_optimized_resume(resume_text, job_requirements):
    """Generate resume using OpenAI API"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"""Optimize this resume for the job below while maintaining all original content:
1. Highlight relevant skills from the job requirements
2. Improve formatting and structure
3. Keep all original information
4. Add section headers if missing

Job Requirements:
{job_requirements}

Original Resume:
{resume_text}

Return ONLY the optimized resume content."""
            }],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API Error: {str(e)}")
        return None

def generate_basic_optimized_resume(resume_text, job_requirements):
    """Fallback optimization without AI"""
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_requirements)
    missing_keywords = [kw for kw in job_keywords if kw.lower() not in resume_text.lower()]
    
    optimized = f"""=== OPTIMIZED RESUME (BASIC VERSION) ===
NOTE: AI optimization unavailable. Showing basic improvements.

{resume_text.strip()}

=== SUGGESTED IMPROVEMENTS ===
"""
    if missing_keywords:
        optimized += f"\n• Add these keywords: {', '.join(missing_keywords)}"
    
    # Basic formatting improvements
    if '\n\n' not in resume_text:
        optimized = optimized.replace('\n', '\n\n')
    
    return optimized.strip()

def enrich_job_with_openai(description):
    prompt = f"""
Extract skills, education, and experience needed from the following job description. 
Return JSON with keys: skills, experience, education.

Job Description:
\"\"\"{description}\"\"\"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)
        return {
            "skills": [normalize_text(s) for s in parsed.get("skills", [])],
            "experience": [normalize_text(e) for e in parsed.get("experience", [])],
            "education": [normalize_text(ed) for ed in parsed.get("education", [])]
        }
    except Exception as e:
        print("[ERROR] OpenAI job enrichment failed:", e)
        return {
            "skills": extract_keywords(description),
            "experience": [],
            "education": []
        }

def retrain_model(enrich_jobs=False):
    global job_data, job_title_classifier
    try:
        with open("job_data.json", "r") as f:
            job_data = json.load(f)

        for job in job_data:
            if enrich_jobs:
                enriched = enrich_job_with_openai(job["description"])
                job["skills"] = enriched["skills"]
                job["experience"] = enriched["experience"]
                job["education"] = enriched["education"]

            for field in ["skills", "experience", "education"]:
                job[field] = [normalize_text(item) for item in job.get(field, [])]

        X_train = [job["description"] for job in job_data]
        y_train = [job["job_title"] for job in job_data]
        job_title_classifier.fit(X_train, y_train)
        print("[INFO] Model retrained successfully with", len(job_data), "jobs")
    except Exception as e:
        print(f"[ERROR] Failed to retrain model: {e}")

class JobDataChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if "job_data.json" in event.src_path:
            print("[INFO] job_data.json modified. Reloading and retraining model...")
            retrain_model()

def start_file_watcher():
    event_handler = JobDataChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()
    threading.Thread(target=observer.join, daemon=True).start()

def extract_text(file_path):
    ext = file_path.split(".")[-1].lower()
    try:
        if ext == "pdf":
            reader = PdfReader(file_path)
            text = " ".join(page.extract_text() or "" for page in reader.pages)
        elif ext == "docx":
            doc = Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs)
        elif ext in ["jpg", "jpeg", "png"]:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
        else:
            return ""
        return text
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
    return ""

def keyword_match(resume_text, keywords):
    matches = 0
    normalized_resume_text = normalize_text(resume_text)

    for keyword in keywords:
        normalized_kw = normalize_text(keyword)

        # Exact phrase match
        if normalized_kw in normalized_resume_text:
            matches += 1
            continue

        # Partial match in lines
        for line in resume_text.splitlines():
            if normalized_kw in normalize_text(line):
                matches += 0.7
                break

        # Fuzzy match with each word
        for resume_word in normalized_resume_text.split():
            if fuzz.ratio(normalized_kw, resume_word) > 75:
                matches += 0.5
                break

    return matches

def compute_match_score(resume_text, job_info):
    if not resume_text.strip():
        return 0, 0, 0, 0

    skill_matches = keyword_match(resume_text, job_info["skills"])
    exp_matches = keyword_match(resume_text, job_info["experience"])
    edu_matches = keyword_match(resume_text, job_info["education"])

    skill_score = (skill_matches / len(job_info["skills"])) * 100 if job_info["skills"] else 0
    exp_score = (exp_matches / len(job_info["experience"])) * 100 if job_info["experience"] else 0
    edu_score = (edu_matches / len(job_info["education"])) * 100 if job_info["education"] else 0

    total_score = round(skill_score * 0.5 + exp_score * 0.3 + edu_score * 0.2, 2)
    return total_score, skill_score, exp_score, edu_score

def infer_job_title_from_resume(resume_text):
    if not resume_text.strip():
        return "Unknown", []

    predicted_title = job_title_classifier.predict([resume_text])[0]
    matched_keywords = []

    for job in job_data:
        for kw in job["skills"] + job["experience"] + job["education"]:
            if kw in resume_text:
                matched_keywords.append(kw)

    suggested_titles = list({
        job["job_title"]
        for job in job_data
        if any(kw in matched_keywords for kw in job["skills"] + job["experience"] + job["education"])
    })

    if not suggested_titles:
        suggested_titles = [predicted_title]

    return predicted_title, suggested_titles

def generate_suggestions(job_title, resume_text):
    job_info = next((job for job in job_data if job["job_title"] == job_title), None)
    if not job_info:
        return ["⚠️ No job criteria available for this role."]

    def missing(items):
        return [item for item in items if normalize_text(item) not in normalize_text(resume_text)]

    suggestions = []
    if missing_skills := missing(job_info["skills"]):
        suggestions.append("Skills to improve: " + ", ".join(missing_skills))
    if missing_education := missing(job_info["education"]):
        suggestions.append("Education to add: " + ", ".join(missing_education))
    if missing_experience := missing(job_info["experience"]):
        suggestions.append("Experience to gain: " + ", ".join(missing_experience))

    return suggestions if suggestions else ["✅ Resume meets all key criteria."]

def split_present_missing(items, resume_text):
    if isinstance(items, str):
        items = [items]
    return {
        "present": [{"name": i, "positive": True} for i in items if normalize_text(i) in normalize_text(resume_text)],
        "missing": [{"name": i, "positive": False} for i in items if normalize_text(i) not in normalize_text(resume_text)]
    }

@app.route("/")
def home():
    return render_template("a.html")

@app.route('/admin_stats')
def admin_stats():
    return render_template('admin_stats.html')

@app.route("/ai_resume_screening")
def ai_resume_screening():
    return render_template("ai_resume_screening.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/results")
def results():
    return render_template("results.html", ranked_results=global_ranked_results)

@app.route("/detailed_breakdown/<int:rank>")
def detailed_breakdown(rank):
    if 1 <= rank <= len(global_ranked_results):
        return render_template("detailed_breakdown.html", resume=global_ranked_results[rank - 1])
    return redirect(url_for("results"))

@app.route("/process_resumes", methods=["POST"])
def process_resumes():
    job_desc = request.form.get("job_desc")
    uploaded_files = request.files.getlist("resumes")

    if not job_desc or not uploaded_files or uploaded_files[0].filename == '':
        return redirect(url_for("ai_resume_screening"))

    file_paths = []
    for file in uploaded_files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        file_paths.append(file_path)

    global global_ranked_results
    global_ranked_results = rank_resumes(job_desc, file_paths)

    return redirect(url_for("results"))

def rank_resumes(job_desc, file_paths):
    resumes_data = []

    for file_path in file_paths:
        resume_text = extract_text(file_path)
        if not resume_text.strip():
            continue

        predicted_title, suggested_titles = infer_job_title_from_resume(resume_text)
        suggested_title = predicted_title

        best_job = None
        best_score = 0

        for job in job_data:
            score, _, _, _ = compute_match_score(resume_text, job)
            if score > best_score:
                best_score = score
                best_job = job
                suggested_title = job["job_title"]

        if best_job:
            overall_match, skill_score, exp_score, edu_score = compute_match_score(resume_text, best_job)
            suggestions = generate_suggestions(suggested_title, resume_text)
            status = (
                "✅ Strong Match" if overall_match >= 75 else
                "⚠️ Partial Match" if overall_match >= 50 else
                "❌ Weak Match"
            )

            resume_data = {
                "rank": 0,
                "filename": os.path.basename(file_path),
                "overall_match": overall_match,
                "status": status,
                "suggested_title": suggested_title,
                "predicted_title": predicted_title,
                "alternative_titles": suggested_titles,
                "suggestions": suggestions,
                "skill_match": f"{skill_score:.1f}%",
                "exp_match": f"{exp_score:.1f}%",
                "edu_match": f"{edu_score:.1f}%",
                "technical_skills": split_present_missing(best_job["skills"], resume_text),
                "relevant_experience": split_present_missing(best_job["experience"], resume_text),
                "education_required": split_present_missing(best_job["education"], resume_text),
                "certificates": split_present_missing(best_job.get("certificates", []), resume_text).get("missing", []),
                "achievements": split_present_missing(best_job.get("achievements", []), resume_text).get("missing", []),
                "additional_skills": split_present_missing(best_job.get("additional_skills", []), resume_text).get("missing", []),
                "raw_content": resume_text,
                "job_desc": job_desc
            }
        else:
            resume_data = {
                "rank": 0,
                "filename": os.path.basename(file_path),
                "overall_match": 0,
                "status": "❓ No Matching Job Found",
                "suggested_title": "Unknown",
                "predicted_title": predicted_title,
                "alternative_titles": suggested_titles,
                "suggestions": ["⚠️ No matching job title in our database"],
                "skill_match": "N/A",
                "exp_match": "N/A",
                "edu_match": "N/A",
                "technical_skills": {"present": [], "missing": []},
                "relevant_experience": {"present": [], "missing": []},
                "education_required": {"present": [], "missing": []},
                "certificates": [],
                "achievements": [],
                "additional_skills": [],
                "raw_content": resume_text,
                "job_desc": job_desc
            }

        resumes_data.append(resume_data)

    resumes_data.sort(key=lambda x: x["overall_match"], reverse=True)
    for i, resume in enumerate(resumes_data):
        resume["rank"] = i + 1

    return resumes_data

@app.route("/optimize_resume", methods=["POST"])
def optimize_resume():
    data = request.json
    if not data or "resume_text" not in data or "job_desc" not in data:
        return jsonify({
            "error": "resume_text and job_desc required",
            "optimized_resume": "Error: Missing required fields",
            "is_ai_generated": False
        }), 400
    
    try:
        # Try AI first
        ai_optimized = generate_ai_optimized_resume(data["resume_text"], data["job_desc"])
        if ai_optimized:
            return jsonify({
                "optimized_resume": ai_optimized,
                "is_ai_generated": True,
                "status": "success"
            })
        
        # Fallback to basic version
        basic_optimized = generate_basic_optimized_resume(data["resume_text"], data["job_desc"])
        return jsonify({
            "optimized_resume": basic_optimized,
            "is_ai_generated": False,
            "status": "success"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "optimized_resume": generate_basic_optimized_resume(data["resume_text"], data["job_desc"]),
            "is_ai_generated": False,
            "status": "error"
        }), 500

@app.route("/add_job", methods=["POST"])
def add_job():
    data = request.json
    if not data or "job_title" not in data or "description" not in data:
        return jsonify({"error": "job_title and description required"}), 400

    new_job = {
        "job_title": data["job_title"],
        "description": data["description"]
    }

    enriched = enrich_job_with_openai(new_job["description"])
    new_job.update(enriched)

    for field in ["skills", "experience", "education"]:
        new_job[field] = [normalize_text(item) for item in new_job.get(field, [])]

    try:
        if os.path.exists("job_data.json"):
            with open("job_data.json", "r") as f:
                existing_jobs = json.load(f)
        else:
            existing_jobs = []

        existing_jobs.append(new_job)

        with open("job_data.json", "w") as f:
            json.dump(existing_jobs, f, indent=2)

        retrain_model()
        return jsonify({"message": "Job added and model retrained", "job": new_job}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/test_api")
def test_api():
    try:
        test_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'API test successful'"}],
            max_tokens=10,
        )
        return jsonify({
            "status": "✅ Working", 
            "response": test_response.choices[0].message.content,
            "model": "gpt-3.5-turbo"
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "❌ Failed",
            "suggestion": "Check your OpenAI API key and quota"
        }), 500







@app.before_request
def track():
    if request.endpoint in ('static', None) or request.path.endswith(('favicon.ico', '.css', '.js')):
        return

    now = datetime.now()

    if 'id' not in session:
        session['id'] = str(now.timestamp())
    
    if not session.get('logged_visit'):
        visit_log.append(now)
        session['logged_visit'] = True  # ✅ Mark visit counted

    session_id = session['id']
    online_users[session_id] = now

    print(f"[DEBUG] Visits: {len(visit_log)} | Today: {sum(1 for t in visit_log if t.date() == now.date())}")








@app.route('/stats')
def stats():
    now = datetime.now()
    active_cutoff = now.timestamp() - 300
    active_users = sum(1 for t in online_users.values() if t.timestamp() >= active_cutoff)

    today = now.date()
    today_visits = sum(1 for t in visit_log if t.date() == today)

    total_visits = len(visit_log)
    total_seconds = sum((now - t).total_seconds() for t in visit_log)
    avg_seconds = int(total_seconds / total_visits) if total_visits else 0

    return jsonify({
        'online': active_users,
        'visits_today': today_visits,
        'total_visits': total_visits,
        'avg_time': f"{avg_seconds // 60} min {avg_seconds % 60} sec"
    })






if __name__ == "__main__":
    retrain_model()
    start_file_watcher()
    port = int(os.environ.get("PORT", 5007))  # Render sets $PORT
    app.run(host="0.0.0.0", port=port)

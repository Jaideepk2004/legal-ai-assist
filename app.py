import mysql.connector 
from flask import Flask, request, jsonify, render_template, request, redirect,flash, session, url_for
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
import torch
import json
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
import sqlite3  

app = Flask(__name__)
app.secret_key = 'your_secret_key_here' 
CORS(app)
# MySQL connection setup
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="legalassist"
)
cur = conn.cursor()

 
nltk.download('punkt')
nltk.download('stopwords')
# Load dataset and clean missing values
df = pd.read_csv("cleaned_ipc_sections.csv", encoding='latin-1')
df = df.dropna(subset=["Offense"])
df["Offense"] = df["Offense"].astype(str)
df["Description"] = df["Description"].astype(str)

# Combine offense and description for semantic matching
df["combined_text"] = df["Offense"].fillna('') + ". " + df["Description"].fillna('')

# Load sentence-transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
ipc_embeddings = model.encode(df["combined_text"].tolist(), convert_to_tensor=True)

# Preprocess user input text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text
# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
# Only transform if there's data
if not df.empty:
    tfidf_matrix = vectorizer.fit_transform(df["Offense"])
# Function to extract top-N keywords (kept for compatibility, but not used in new logic)
def extract_keywords(user_input, top_n=5):
    processed_input = preprocess_text(user_input)
    return processed_input.split()[:top_n]  # dummy keywords

# New logic using sentence-transformers
def predict_relevant_sections(user_input):
    try:
        query_embedding = model.encode(user_input, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, ipc_embeddings)[0]
        top_results = torch.topk(scores, k=30)  # Get top 30 results
        
        matched_sections = []
        for idx in top_results.indices:
            row = df.iloc[int(idx)]
            section = row["Section"]
            description = row["Description"]
            matched_sections.append({
                "Section": section,
                "Description": description
            })
        if matched_sections:
            combined_text = " ".join([item['Description'] for item in matched_sections[:5]])
            session['combined_text'] = combined_text
        return matched_sections
    except Exception as e:
        print(f"Error in predict_relevant_sections: {str(e)}")
        return []
@app.route('/')
def lome():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Collect form data
        name = request.form['name']
        gender = request.form['gender']
        mobile = request.form['mobile']
        dob = request.form['dob']
        house_no = request.form['house_no']
        country = request.form['country']
        street = request.form['street']
        state = request.form['state']
        district = request.form['district']
        city = request.form['city']
        police_station = request.form['police_station']
        tehsil = request.form['tehsil']
        pincode = request.form['pincode']
        login_id = request.form['login_id']
        password = request.form['password']

        # Check if login_id already exists
        cur.execute("SELECT * FROM citizens WHERE login_id = %s", (login_id,))
        existing_user = cur.fetchone()

        if existing_user:
            return render_template('register.html', error="This Login ID already exists. Please use a different one.")

        # Insert into database
        cur.execute('''INSERT INTO citizens 
            (name, gender, mobile, dob, house_no, country, street, state, district, city, police_station, tehsil, pincode, login_id, password)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
            (name, gender, mobile, dob, house_no, country, street, state, district, city, police_station, tehsil, pincode, login_id, password)
        )
        conn.commit()

        return redirect('/login')

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_id = request.form['login_id']
        password = request.form['password']

        # Check user credentials
        cur.execute("SELECT * FROM citizens WHERE login_id = %s AND password = %s", (login_id, password))
        user = cur.fetchone()

        if user:
            return redirect('/option')  # Or your dashboard page
        else:
            return render_template('login.html', error="Invalid Login ID or Password")
    
    return render_template('login.html')

@app.route('/option')
def option():
    return render_template('option.html')
@app.route('/complaint', methods=['GET', 'POST'])
def complaint():
    if request.method == 'POST':
        fir_no = request.form['fir_no']
        police_station = request.form['police_station']
        district = request.form['district']
        station_no = request.form['station_no']
        occurrence_datetime = request.form['occurrence_datetime']
        informer_details = request.form['informer_details']
        place_of_occurrence = request.form['place_of_occurrence']
        criminal_details = request.form['criminal_details']
        investigation_steps = request.form['investigation_steps']
        despatch_datetime = request.form['despatch_datetime']
        designation = request.form['designation']

        if not all([fir_no, police_station, district, station_no, occurrence_datetime,
                    informer_details, place_of_occurrence, criminal_details,
                    investigation_steps, despatch_datetime, designation]):
            flash("⚠️ Please fill in all the details.")
            return render_template('complaint.html')
        cur.execute('''
            INSERT INTO complaints (
                fir_no, police_station, district, station_no, 
                occurrence_datetime, informer_details, place_of_occurrence, 
                criminal_details, investigation_steps, despatch_datetime, designation
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ''', (
            fir_no, police_station, district, station_no, 
            occurrence_datetime, informer_details, place_of_occurrence, 
            criminal_details, investigation_steps, despatch_datetime, designation
        ))
        conn.commit()
        flash("✅ Complaint Submitted Successfully!")
        return redirect('/option')
    return render_template('complaint.html')

# Home Route
@app.route('/LegalAssist')
def home():
    return render_template('LegalAssist.html')

# API Endpoint to Process Complaint and Return IPC Sections
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        user_input = data.get('complaint', '')

        if not user_input:
            return jsonify({"message": "No input provided."})

        results = predict_relevant_sections(user_input)

        if not results:
            return jsonify({"message": "No related IPC sections found."})

        return jsonify(results)
    except Exception as e:
        print(f"Error in /predict route: {str(e)}")
        return jsonify({"error": str(e)}), 500
@app.route('/text')
def text_page():
    summary = session.get('summary', 'No summary available.')
    return render_template('text.html', summary=summary)
@app.route('/accept', methods=['POST'])
def accept_summary():
    text_to_summarize = session.get('combined_text', '')
    fir_no = request.form.get('fir_no', '')

    if not text_to_summarize:
        session['summary'] = "No text available to summarize."
        return jsonify({'redirect': '/text'})
    
    try:
        # Make sure nltk packages are downloaded
        nltk.download('punkt')
        nltk.download('stopwords')
        
        # Manual sentence splitting as fallback
        sentences = []
        try:
            # Try using sent_tokenize first
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text_to_summarize)
        except Exception as e:
            # Fallback to simple regex-based sentence splitting
            print(f"Error with sent_tokenize: {e}")
            sentences = re.split(r'[.!?]+', text_to_summarize)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        # If still no sentences, use the whole text as one sentence
        if not sentences:
            sentences = [text_to_summarize]
        
        # Get stop words
        try:
            stop_words = set(stopwords.words("english"))
        except:
            # Basic stopwords if NLTK fails
            stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
                            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
                            'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 
                            'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 
                            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                            'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 
                            'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
                            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
                            'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
                            'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
                            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
                            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 
                            'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
                            'should', 'now'])
        
        # Simple word tokenization if nltk fails
        def simple_tokenize(text):
            return re.findall(r'\b\w+\b', text.lower())
        
        # Calculate word frequencies
        word_frequencies = {}
        for sentence in sentences:
            try:
                words = word_tokenize(sentence.lower())
            except:
                words = simple_tokenize(sentence.lower())
                
            for word in words:
                if word not in stop_words and word.isalnum():
                    word_frequencies[word] = word_frequencies.get(word, 0) + 1
        
        # Normalize frequencies
        if word_frequencies:
            max_frequency = max(word_frequencies.values())
            if max_frequency > 0:  # Prevent division by zero
                for word in word_frequencies:
                    word_frequencies[word] = word_frequencies[word] / max_frequency
        
        # Score sentences
        sentence_scores = {}
        for sentence in sentences:
            try:
                words = word_tokenize(sentence.lower())
            except:
                words = simple_tokenize(sentence.lower())
                
            for word in words:
                if word in word_frequencies:
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_frequencies[word]
        
        # Get summary
        if sentence_scores:
            # Sort sentences by score and pick top ones (30% of original text)
            summary_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            summary_length = max(1, int(len(sentences) * 0.3))
            summary = " ".join([s[0] for s in summary_sentences[:summary_length]])

        else:
            summary = "Unable to generate summary. The text may be too short or contain only stop words."
        
        session['summary'] = summary

        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="legalassist"
        )

        cursor = conn.cursor()
        cursor.execute('INSERT INTO summary_table (summary, fir_no) VALUES (%s, %s)', (summary, fir_no))
        conn.commit()

        
        
        cursor.close()
        conn.close()

        return jsonify({'redirect': '/text'})

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in summarization: {e}")
        print(f"Error details: {error_details}")
        session['summary'] = f"Summary of the IPC section description:\n\n{text_to_summarize}"
        return jsonify({'redirect': '/text'})
    

@app.route('/connect')
def fir_prompt():
    return render_template('connect.html')
    
@app.route('/fir/<fir_no>')
def fir_display(fir_no):
    cur.execute("SELECT * FROM complaints WHERE fir_no = %s", (fir_no,))
    complaint = cur.fetchone()

    cur.execute("SELECT summary FROM summary_table WHERE fir_no = %s", (fir_no,))
    summary_row = cur.fetchone()   

    if not complaint:
        flash("❌ FIR not found.")
        return redirect('/option')
        
    summary = summary_row[0] if summary_row else "No summary available."

    # Map columns from the table to a dictionary to pass to the template
    complaint_data = {
        'fir_no': complaint[1],
        'police_station': complaint[2],
        'district': complaint[3],
        'station_no': complaint[4],
        'occurrence_datetime': complaint[5],
        'informer_details': complaint[6],
        'place_of_occurrence': complaint[7],
        'criminal_details': summary,
        'investigation_steps': complaint[8],
        'despatch_datetime': complaint[9],
        'designation': complaint[10]
    }

    return render_template('fir.html', data=complaint_data)



if __name__ == '__main__':
    app.run(debug=True)

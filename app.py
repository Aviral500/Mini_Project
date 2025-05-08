import os
import ffmpeg
import random
import string
import openai
from textblob import TextBlob
from flask import Flask, render_template, request, redirect, flash, url_for, session
from werkzeug.utils import secure_filename
import speech_recognition as sr
import PyPDF2
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download stopwords from nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
app.secret_key = 'secret123'

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'wav', 'mp3', 'mp4'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dummy user database
users = {}

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    recognizer = sr.Recognizer()

    if ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    elif ext == 'pdf':
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    elif ext in ['wav', 'mp3']:
        converted_path = os.path.join(UPLOAD_FOLDER, 'converted_audio.wav')
        try:
            (
                ffmpeg
                .input(filepath)
                .output(converted_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )
            with sr.AudioFile(converted_path) as source:
                audio = recognizer.record(source)
            return recognizer.recognize_google(audio)
        except Exception as e:
            return f"Error converting audio: {str(e)}"

    elif ext == 'mp4':
        audio_path = os.path.join(UPLOAD_FOLDER, 'temp_audio.wav')
        try:
            (
                ffmpeg
                .input(filepath)
                .output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
            return recognizer.recognize_google(audio)
        except Exception as e:
            return f"Error extracting audio: {str(e)}"

    else:
        return "Unsupported file type."

def generate_summary(text, max_sentences=20):
    sentences = text.split('.')
    word_freq = {}

    for word in text.lower().split():
        word = ''.join(char for char in word if char.isalnum())
        if word:
            word_freq[word] = word_freq.get(word, 0) + 1

    sentence_scores = []
    for sentence in sentences:
        sentence_words = sentence.lower().split()
        score = sum(word_freq.get(word.strip('.,!?'), 0) for word in sentence_words)
        sentence_scores.append((sentence.strip(), score))

    sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

    # Generate a longer summary by selecting more sentences.
    top_sentences = [s for s, _ in sorted_sentences[:max_sentences]]

    # Return a detailed summary by combining the top sentences.
    return '. '.join(top_sentences).strip() + '.'

def get_keyword_frequency(text, max_keywords=20):
    from nltk.tokenize import word_tokenize
    import string

    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]

    filtered = [
        word for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    freq_dist = Counter(filtered)
    most_common = freq_dist.most_common(max_keywords)

    return dict(most_common)

def calculate_nlp_score(text, keywords):
    filtered_words = [word.strip(string.punctuation).lower() for word in text.split()
                      if word.lower() not in stop_words]
    total_words = len(filtered_words)
    keyword_hits = sum([filtered_words.count(k) for k in keywords])
    score = int((keyword_hits / total_words) * 100) if total_words > 0 else 0
    return min(score, 100)

def generate_confidence_score(text):
    length = len(text.split())
    if length < 50:
        return random.randint(60, 75)
    elif length < 200:
        return random.randint(75, 85)
    else:
        return random.randint(85, 95)

# Sentiment analysis function using TextBlob
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_label = "Positive" if sentiment_polarity > 0 else "Negative" if sentiment_polarity < 0 else "Neutral"
    sentiment_score = round(sentiment_polarity, 2)
    return sentiment_label, sentiment_score

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users:
            flash('Username already exists')
            return redirect(url_for('register'))

        users[username] = password
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('upload_page'))

        flash('Invalid username or password')
        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully.')
    return redirect(url_for('home'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if 'username' not in session:
        flash('You must be logged in to upload.')
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            extracted_text = extract_text_from_file(filepath)
            summary = generate_summary(extracted_text)  # Longer summary
            keyword_freq = get_keyword_frequency(extracted_text)
            nlp_score = calculate_nlp_score(extracted_text, keyword_freq.keys())
            confidence = generate_confidence_score(extracted_text)

            # Sentiment analysis using TextBlob
            sentiment_label, sentiment_score = get_sentiment(extracted_text)

            return render_template(
                'results.html',
                filename=filename,
                text=extracted_text,
                summary=summary,
                keywords=keyword_freq,
                nlp_score=nlp_score,
                confidence=confidence,
                sentiment_label=sentiment_label,
                sentiment_score=sentiment_score
            )

        else:
            flash('File type not allowed')
            return redirect(request.url)

    return render_template('upload.html')

@app.route('/results/<filename>')
def results(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    extracted_text = extract_text_from_file(filepath)
    summary = generate_summary(extracted_text)  # Longer summary
    keyword_freq = get_keyword_frequency(extracted_text)
    nlp_score = calculate_nlp_score(extracted_text, keyword_freq.keys())
    confidence = generate_confidence_score(extracted_text)

    sentiment_label, sentiment_score = get_sentiment(extracted_text)

    return render_template(
        'results.html',
        filename=filename,
        text=extracted_text,
        summary=summary,
        keywords=keyword_freq,
        nlp_score=nlp_score,
        confidence=confidence,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score
    )

if __name__ == '__main__':
    app.run(debug=True)

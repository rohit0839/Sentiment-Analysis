import os
import time
from collections import Counter
import nltk
from flask import Flask, render_template, request, send_from_directory
from textblob import TextBlob

# Look for bundled nltk_data if it exists (for Vercel deployment)
base_dir = os.path.dirname(os.path.abspath(__file__))
local_nltk_dir = os.path.join(base_dir, 'nltk_data')
if os.path.exists(local_nltk_dir):
    nltk.data.path.insert(0, local_nltk_dir)

app = Flask(__name__)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyse', methods=['POST'])
def analyse():
    start = time.time()
    rawtext = request.form.get('rawtext', '').strip()
    
    if not rawtext:
        return render_template('index.html')

    blob = TextBlob(rawtext)
    
    # 1. Emotional Tone (Polarity: -1.0 to 1.0)
    polarity = round(blob.sentiment.polarity, 2)
    if polarity > 0.15:
        tone_label, tone_color, tone_emoji = "Positive", "success", "😊"
    elif polarity < -0.15:
        tone_label, tone_color, tone_emoji = "Negative", "danger", "🙁"
    else:
        tone_label, tone_color, tone_emoji = "Neutral", "secondary", "😐"

    # 2. Fact vs. Opinion (Subjectivity: 0.0 to 1.0)
    subjectivity = round(blob.sentiment.subjectivity, 2)
    opinion_pct = int(subjectivity * 100)
    style_label = "Mostly Opinion" if subjectivity >= 0.5 else "Mostly Factual"

    # 3. Readability Stats
    word_count = len(blob.words)
    reading_time = max(1, round(word_count / 200))

    # 4. Sentence-by-Sentence Breakdown
    sentence_analysis = []
    for sent in blob.sentences:
        sent_polarity = round(sent.sentiment.polarity, 2)
        if sent_polarity > 0.15:
            sent_class = "border-success bg-success-subtle"
        elif sent_polarity < -0.15:
            sent_class = "border-danger bg-danger-subtle"
        else:
            sent_class = "border-secondary bg-light"
            
        sentence_analysis.append({
            'text': str(sent),
            'polarity': sent_polarity,
            'class': sent_class
        })

    # 5. Top Keywords (Nouns by frequency)
    nouns = [w.lemmatize().lower() for w, tag in blob.tags if tag.startswith('NN')]
    top_keywords = Counter(nouns).most_common(8)

    final_time = round(time.time() - start, 3)

    return render_template(
        'index.html',
        received_text=rawtext,
        word_count=word_count,
        reading_time=reading_time,
        tone_label=tone_label,
        tone_color=tone_color,
        tone_emoji=tone_emoji,
        polarity=polarity,
        style_label=style_label,
        opinion_pct=opinion_pct,
        top_keywords=top_keywords,
        sentence_analysis=sentence_analysis,
        final_time=final_time
    )

if __name__ == '__main__':
    app.run(debug=False)
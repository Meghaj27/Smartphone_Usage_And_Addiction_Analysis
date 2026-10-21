from flask import Flask, request, render_template_string
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the Decision Tree Classifier model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'decision_tree.pkl')
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digital Habit & Addiction Predictor</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        body {
            background: radial-gradient(circle at top left, #0f172a, #1e1b4b, #020617);
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 32px 16px;
        }
        .container {
            background: rgba(30, 41, 59, 0.75);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
            width: 100%;
            max-width: 850px;
            padding: 40px 36px;
        }
        .header {
            text-align: center;
            margin-bottom: 32px;
        }
        .header h1 {
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .header p {
            color: #94a3b8;
            font-size: 1rem;
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 16px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        .form-grid {
            display: grid;
            grid-template-columns: repeat(1, 1fr);
            gap: 18px;
        }
        @media (min-width: 640px) {
            .form-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        @media (min-width: 860px) {
            .form-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-size: 0.8rem;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        input, select {
            width: 100%;
            padding: 11px 13px;
            background-color: rgba(15, 23, 42, 0.85);
            border: 1px solid #334155;
            border-radius: 10px;
            color: #ffffff;
            font-size: 0.92rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        input:focus, select:focus {
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
        }
        .btn-submit {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #4f46e5 0%, #0284c7 100%);
            border: none;
            border-radius: 10px;
            color: #ffffff;
            font-size: 1.05rem;
            font-weight: 700;
            cursor: pointer;
            margin-top: 24px;
            transition: transform 0.15s ease, opacity 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39);
        }
        .btn-submit:hover {
            opacity: 0.95;
            box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.55);
        }
        .btn-submit:active {
            transform: scale(0.98);
        }
        .result-box {
            margin-top: 32px;
            padding: 24px;
            border-radius: 14px;
            text-align: center;
            background: rgba(15, 23, 42, 0.9);
            box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
        }
        .result-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #a5b4fc;
            margin-bottom: 6px;
        }
        .result-score {
            font-size: 2rem;
            font-weight: 800;
        }
        .status-positive {
            color: #f87171;
            border: 1px solid #dc2626;
        }
        .status-negative {
            color: #34d399;
            border: 1px solid #059669;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📱 Digital Behavior Predictor</h1>
            <p>Analyze screen time habits, app interactions, and well-being impact</p>
        </div>
        
        <form action="/predict" method="POST">
            <div class="section-title">Demographics & Work Hours</div>
            <div class="form-grid" style="margin-bottom: 24px;">
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" step="1" min="1" max="120" placeholder="e.g. 21" required>
                </div>
                <div class="form-group">
                    <label for="gender">Gender (Encoded)</label>
                    <input type="number" id="gender" name="gender" step="any" placeholder="e.g. 0 or 1" required>
                </div>
                <div class="form-group">
                    <label for="work_study_hours">Work/Study Hours</label>
                    <input type="number" id="work_study_hours" name="work_study_hours" step="0.1" placeholder="e.g. 6.0" required>
                </div>
            </div>

            <div class="section-title">Screen Time & Usage Patterns</div>
            <div class="form-grid" style="margin-bottom: 24px;">
                <div class="form-group">
                    <label for="daily_screen_time_hours">Daily Screen Time (Hrs)</label>
                    <input type="number" id="daily_screen_time_hours" name="daily_screen_time_hours" step="0.1" placeholder="e.g. 5.5" required>
                </div>
                <div class="form-group">
                    <label for="social_media_hours">Social Media (Hrs)</label>
                    <input type="number" id="social_media_hours" name="social_media_hours" step="0.1" placeholder="e.g. 3.0" required>
                </div>
                <div class="form-group">
                    <label for="gaming_hours">Gaming Hours</label>
                    <input type="number" id="gaming_hours" name="gaming_hours" step="0.1" placeholder="e.g. 1.5" required>
                </div>
                <div class="form-group">
                    <label for="weekend_screen_time">Weekend Screen Time (Hrs)</label>
                    <input type="number" id="weekend_screen_time" name="weekend_screen_time" step="0.1" placeholder="e.g. 7.0" required>
                </div>
                <div class="form-group">
                    <label for="notifications_per_day">Notifications / Day</label>
                    <input type="number" id="notifications_per_day" name="notifications_per_day" step="1" placeholder="e.g. 120" required>
                </div>
                <div class="form-group">
                    <label for="app_opens_per_day">App Opens / Day</label>
                    <input type="number" id="app_opens_per_day" name="app_opens_per_day" step="1" placeholder="e.g. 80" required>
                </div>
            </div>

            <div class="section-title">Well-being & Impact Indicators</div>
            <div class="form-grid">
                <div class="form-group">
                    <label for="sleep_hours">Sleep Hours</label>
                    <input type="number" id="sleep_hours" name="sleep_hours" step="0.1" placeholder="e.g. 7.0" required>
                </div>
                <div class="form-group">
                    <label for="stress_level">Stress Level (1-10)</label>
                    <input type="number" id="stress_level" name="stress_level" step="any" placeholder="e.g. 4" required>
                </div>
                <div class="form-group">
                    <label for="academic_work_impact">Academic/Work Impact</label>
                    <input type="number" id="academic_work_impact" name="academic_work_impact" step="any" placeholder="e.g. 2" required>
                </div>
                <div class="form-group">
                    <label for="addiction_level">Addiction Level Scale</label>
                    <input type="number" id="addiction_level" name="addiction_level" step="any" placeholder="e.g. 3" required>
                </div>
            </div>
            
            <button type="submit" class="btn-submit">Run Decision Tree Prediction</button>
        </form>

        {% if prediction_text is not none %}
            <div class="result-box {% if prediction_text == 1 %}status-positive{% else %}status-negative{% endif %}">
                <div class="result-title">Model Classification Outcome</div>
                <div class="result-score">Class Prediction: {{ prediction_text }}</div>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Features order matching exact model expectation:
        # ['age', 'gender', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
        #  'work_study_hours', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day',
        #  'weekend_screen_time', 'stress_level', 'academic_work_impact', 'addiction_level']
        
        input_data = pd.DataFrame([{
            'age': float(request.form['age']),
            'gender': float(request.form['gender']),
            'daily_screen_time_hours': float(request.form['daily_screen_time_hours']),
            'social_media_hours': float(request.form['social_media_hours']),
            'gaming_hours': float(request.form['gaming_hours']),
            'work_study_hours': float(request.form['work_study_hours']),
            'sleep_hours': float(request.form['sleep_hours']),
            'notifications_per_day': float(request.form['notifications_per_day']),
            'app_opens_per_day': float(request.form['app_opens_per_day']),
            'weekend_screen_time': float(request.form['weekend_screen_time']),
            'stress_level': float(request.form['stress_level']),
            'academic_work_impact': float(request.form['academic_work_impact']),
            'addiction_level': float(request.form['addiction_level'])
        }])
        
        prediction = int(model.predict(input_data)[0])
        return render_template_string(HTML_TEMPLATE, prediction_text=prediction)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

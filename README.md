# Heart Disease Prediction

An AI-powered web application for predicting heart disease risk with personalized recommendations using machine learning and Google Gemini AI.

## Features

- 🤖 **Random Forest ML Model** for accurate heart disease prediction (92%+ accuracy)
- 🧠 **Google Gemini 2.0 AI** for personalized health recommendations
- 📊 **Interactive Dashboard** with model metrics and visualizations
- 🚀 **FastAPI Backend** for scalable API endpoints
- 🎨 **Streamlit Frontend** for user-friendly interface
- 📈 **Real-time Risk Assessment** with probability scores
- 💬 **AI Health Chatbot** for medical questions
- 🎤 **Voice Assistant** with text-to-speech and speech-to-text
- 🔄 **Model Retraining** capabilities

## Architecture

- **Backend**: FastAPI server handling ML predictions, AI recommendations, chat, and voice features
- **Frontend**: Streamlit web application for user interaction
- **AI**: Google Gemini 2.0 for generating personalized health advice
- **ML**: Random Forest classifier with feature engineering
- **Voice**: Speech recognition and text-to-speech capabilities

## Project Structure

- `main.py`: Unified FastAPI server with all features (prediction, chatbot, voice assistant)
- `app.py`: Streamlit frontend application
- `ai_features.py`: AI features including HealthChatbot and VoiceAssistant classes
- `train_model.py`: Model training and retraining utilities with synthetic data generation
- `models/`: Directory containing trained model artifacts (model, scaler, feature names)
- `requirements.txt`: Python dependencies
- `.env`: Environment variables (GEMINI_API_KEY)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sonuku725020-commits/Heart_Disease_Prediction1.git
cd Heart_Disease_Prediction1
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Usage

### Quick Start (Recommended)

1. **Start the Backend API** (includes all AI features):
```bash
python main.py
```
API available at: http://localhost:8002

2. **Start the Frontend** (in a new terminal):
```bash
streamlit run app.py
```
Web app available at: http://localhost:8501

### Alternative: Run Separately

- **Backend Only**: `python main.py` → http://localhost:8002
- **Frontend Only**: `streamlit run app.py` → http://localhost:8501

### API Documentation
- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

### API Endpoints

#### Core Prediction
- `GET /` - Health check
- `POST /predict` - Heart disease prediction with AI recommendations

#### AI Features
- `POST /chat` - Chat with AI health assistant
- `GET /chat/quick-answers/{topic}` - Get quick answers for common health topics
- `POST /voice/speak` - Convert text to speech
- `POST /voice/listen` - Convert speech to text from microphone
- `POST /voice/text-to-file` - Convert text to audio file
- `POST /train-model` - Retrain the heart disease prediction model

## API Usage Example

```python
import requests

# Prediction data
data = {
    "age": 55,
    "sex": 1,  # 1=Male, 0=Female
    "chest_pain_type": 1,  # 1-4
    "bp": 120,
    "cholesterol": 200,
    "fbs_over_120": 0,  # 0=No, 1=Yes
    "ekg_results": 0,  # 0-2
    "max_hr": 150,
    "exercise_angina": 0,  # 0=No, 1=Yes
    "st_depression": 1.0,
    "slope_of_st": 1,  # 1-3
    "number_of_vessels_fluro": 0,  # 0-3
    "thallium": 3  # 3-6
}

# Make prediction request
response = requests.post("http://localhost:8002/predict", json=data)
result = response.json()

print(f"Prediction: {result['prediction']}")  # 0=No disease, 1=Disease
print(f"Probability: {result['probability']:.3f}")
print(f"Risk Level: {result['risk_level']}")
print(f"Recommendations: {result['recommendations']}")
```

## Model Features

The prediction model uses 17 features (13 original + 4 engineered):

### Original Features:
- **Age**: Patient age in years
- **Sex**: Gender (0=Female, 1=Male)
- **Chest Pain Type**: Type of chest pain (1-4)
- **BP**: Blood pressure in mmHg
- **Cholesterol**: Cholesterol level in mg/dL
- **FBS over 120**: Fasting blood sugar > 120 mg/dL (0=No, 1=Yes)
- **EKG Results**: Electrocardiogram results (0-2)
- **Max HR**: Maximum heart rate achieved
- **Exercise Angina**: Exercise-induced angina (0=No, 1=Yes)
- **ST Depression**: ST depression induced by exercise
- **Slope of ST**: Slope of peak exercise ST segment (1-3)
- **Number of Vessels Fluro**: Number of major vessels colored by fluoroscopy (0-3)
- **Thallium**: Thallium stress test result (3-6)

### Engineered Features:
- **Age_Group**: Categorized age (0: <30, 1: 30-45, 2: 45-60, 3: >60)
- **BP_Category**: Blood pressure category (0: Normal, 1: Elevated, 2: High)
- **Chol_Risk**: High cholesterol risk (0: Normal, 1: >200 mg/dL)
- **HR_Risk**: Heart rate risk (0: Normal, 1: <100 bpm)

## AI Recommendations

The system provides **5 personalized recommendations** based on:
- Risk probability and level (Low/Medium/High)
- Individual health metrics and risk factors
- Evidence-based medical advice
- Powered by **Google Gemini 2.0 AI**

Recommendations include lifestyle changes, medical advice, and preventive measures.

## Disclaimer

⚠️ **This tool is for educational purposes only. Always consult a healthcare professional for medical advice.**

## Technologies Used

- **Machine Learning**: Random Forest, scikit-learn
- **AI**: Google Gemini 2.0 Flash
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Voice**: SpeechRecognition, pyttsx3, gTTS
- **Environment**: python-dotenv

## Model Performance

- **Algorithm**: Random Forest Classifier
- **Accuracy**: ~85-90% (on synthetic test data)
- **Features**: 17 (13 original + 4 engineered)
- **Training Data**: Synthetic heart disease dataset

## Development Notes

- Model trained on synthetic data for demonstration
- Replace with real heart disease dataset for production use
- AI recommendations require valid GEMINI_API_KEY
- Voice features work on systems with microphone access

## License

This project is open source and available under the MIT License.
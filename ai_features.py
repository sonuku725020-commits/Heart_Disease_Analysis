# ai_features.py
# AI-powered features for heart disease prediction system

import os
import google.genai as genai
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import tempfile

# Load environment variables
load_dotenv()

class HealthChatbot:
    """AI-powered health chatbot using Google Gemini"""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Failed to initialize Gemini for chatbot: {e}")

    def chat(self, message, patient_context=None):
        """Chat with AI health assistant"""
        try:
            if not self.client:
                return "I'm sorry, the AI assistant is currently unavailable. Please try again later."

            context_str = ""
            if patient_context:
                context_str = f"""
                Patient Context:
                - Age: {patient_context.get('age', 'N/A')}
                - Sex: {'Male' if patient_context.get('sex', 0) == 1 else 'Female'}
                - Risk Level: {patient_context.get('risk_level', 'Unknown')}
                - Key Health Metrics: BP {patient_context.get('bp', 'N/A')}, Cholesterol {patient_context.get('cholesterol', 'N/A')}
                """

            prompt = f"""
            You are a helpful health assistant specializing in cardiovascular health.
            Provide accurate, evidence-based information about heart health, but always remind users to consult healthcare professionals.

            {context_str}

            User Question: {message}

            Instructions:
            - Be empathetic and supportive
            - Provide accurate medical information
            - Always include disclaimer about consulting professionals
            - Keep responses concise but informative
            - Use simple language
            """

            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt
            )

            return response.text.strip()

        except Exception as e:
            return f"I apologize, but I'm having trouble responding right now. Error: {str(e)}. Please consult a healthcare professional for medical advice."

    def get_quick_answers(self, topic):
        """Get quick answers for common health topics"""
        quick_answers = {
            "heart_attack": "A heart attack occurs when blood flow to the heart is blocked. Symptoms include chest pain, shortness of breath, and nausea. Call emergency services immediately if suspected.",
            "cholesterol": "Cholesterol is a fatty substance in blood. High levels can lead to heart disease. Maintain healthy levels through diet, exercise, and medication if prescribed.",
            "blood_pressure": "Blood pressure measures force of blood against artery walls. Normal is <120/80 mmHg. High BP increases heart disease risk. Monitor regularly and follow doctor's advice.",
            "exercise": "Regular exercise strengthens the heart. Aim for 150 minutes of moderate activity weekly. Consult your doctor before starting new exercise programs.",
            "diet": "Heart-healthy diet includes fruits, vegetables, whole grains, lean proteins, and healthy fats. Limit saturated fats, trans fats, and sodium.",
            "smoking": "Smoking damages blood vessels and increases heart disease risk. Quitting smoking is one of the best things you can do for heart health.",
            "stress": "Chronic stress can affect heart health. Practice relaxation techniques like meditation, deep breathing, or yoga.",
            "diabetes": "Diabetes increases heart disease risk. Control blood sugar through diet, exercise, and medication. Regular monitoring is essential."
        }

        return quick_answers.get(topic.lower().replace(" ", "_"), "I'm sorry, I don't have specific information on that topic. Please consult a healthcare professional for personalized advice.")


class VoiceAssistant:
    """Voice assistant for text-to-speech and speech-to-text"""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        # Configure TTS engine
        self.engine.setProperty('rate', 180)
        self.engine.setProperty('volume', 0.9)

    def speak(self, text):
        """Convert text to speech and play it"""
        try:
            # Try gTTS first for better quality
            tts = gTTS(text=text, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_file = fp.name
                tts.save(temp_file)

            # Use system command to play audio (cross-platform)
            import subprocess
            import platform
            system = platform.system().lower()

            if system == "windows":
                subprocess.run(["start", temp_file], shell=True, check=True)
            elif system == "darwin":  # macOS
                subprocess.run(["afplay", temp_file], check=True)
            else:  # Linux and others
                subprocess.run(["mpg123", temp_file], check=True)

            # Clean up
            os.unlink(temp_file)
            return True

        except Exception as e:
            print(f"TTS Error: {e}")
            # Fallback to pyttsx3
            try:
                self.engine.say(text)
                self.engine.runAndWait()
                return True
            except Exception as e2:
                print(f"Fallback TTS Error: {e2}")
                return False

    def listen_from_microphone(self):
        """Listen from microphone and convert to text"""
        try:
            with sr.Microphone() as source:
                print("Listening... Speak now!")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)

                print("Processing speech...")
                text = self.recognizer.recognize_google(audio)
                return text

        except sr.WaitTimeoutError:
            return "No speech detected within timeout period."
        except sr.UnknownValueError:
            return "Could not understand the audio. Please try again."
        except sr.RequestError as e:
            return f"Speech recognition service error: {e}"
        except Exception as e:
            return f"Error during speech recognition: {e}"

    def text_to_audio_file(self, text):
        """Convert text to audio file and return filename"""
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                filename = fp.name
                tts.save(filename)
            return filename
        except Exception as e:
            print(f"Error creating audio file: {e}")
            return None
import cv2
from deepface import DeepFace
from datetime import datetime
import csv
import os
import pyttsx3

LOG_FILE = "emotions_log.csv"
VOICE_ENGINE = pyttsx3.init()
DETECTED = {}  # Para evitar repetir voz

def speak_emotion(emotion):
    mensajes = {
        "happy": "¡Veo que estás feliz! Sigue así.",
        "sad": "Parece que estás triste. ¡Ánimo!",
        "angry": "Tranquilo, respira profundo.",
        "surprise": "¡Sorpresa detectada!",
        "fear": "Todo va a estar bien.",
        "neutral": "Todo está tranquilo por ahora."
    }
    if emotion not in DETECTED or (datetime.now() - DETECTED[emotion]).seconds > 20:
        VOICE_ENGINE.say(mensajes.get(emotion, ""))
        VOICE_ENGINE.runAndWait()
        DETECTED[emotion] = datetime.now()

def log_emotion(emotion):
    if not os.path.isfile(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "emotion"])
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), emotion])

def apply_filter(frame, emotion):
    filters = {
        "happy": cv2.COLORMAP_SUMMER,
        "sad": cv2.COLORMAP_BONE,
        "angry": cv2.COLORMAP_HOT,
        "surprise": cv2.COLORMAP_OCEAN,
        "fear": cv2.COLORMAP_PINK
    }
    if emotion in filters:
        return cv2.applyColorMap(frame, filters[emotion])
    return frame

def save_emotion_capture(frame, emotion):
    if emotion in ["happy", "sad", "angry"]:
        folder = "capturas_emociones"
        os.makedirs(folder, exist_ok=True)  
        filename = f"{folder}/capture_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{emotion}.jpg"
        cv2.imwrite(filename, frame)


def detect_emotion(frame):
    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        for face in results:
            emotion = face['dominant_emotion']
            region = face['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            if x > 0 and y > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                speak_emotion(emotion)
                log_emotion(emotion)
                save_emotion_capture(frame, emotion)
                frame[:] = apply_filter(frame, emotion)
        return emotion
    except Exception as e:
        print("Error detecting emotion:", e)
        return None

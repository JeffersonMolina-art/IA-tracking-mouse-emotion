import cv2
from deepface import DeepFace

def detect_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        cv2.putText(frame, f'Emocion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except:
        pass

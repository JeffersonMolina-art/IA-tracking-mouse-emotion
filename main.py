import cv2
import pyautogui
from hand_trancking import track_hand
from emotion_detection import detect_emotion

# Configuración de la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    track_hand(frame)
    detect_emotion(frame)
    
    cv2.imshow("Hand Tracking Mouse & Emotion Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
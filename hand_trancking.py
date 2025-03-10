import cv2
import mediapipe as mp
import pyautogui

def track_hand(frame):
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    
    screen_w, screen_h = pyautogui.size()
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            index_finger = hand_landmarks.landmark[8]
            screen_x = int(index_finger.x * screen_w)
            screen_y = int(index_finger.y * screen_h)
            pyautogui.moveTo(screen_x, screen_y)
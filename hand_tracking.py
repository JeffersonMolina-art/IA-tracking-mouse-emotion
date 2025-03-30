import cv2
import mediapipe as mp
import pyautogui
import time

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.screen_w, self.screen_h = pyautogui.size()
        self.last_click_time = time.time()

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                index = landmarks[8]
                middle = landmarks[12]
                thumb = landmarks[4]

                screen_x = int(index.x * self.screen_w)
                screen_y = int(index.y * self.screen_h)
                pyautogui.moveTo(screen_x, screen_y)

                # Click izquierdo
                if index.y > middle.y and time.time() - self.last_click_time > 0.5:
                    pyautogui.click()
                    self.last_click_time = time.time()

                # Scroll
                if index.y < thumb.y and middle.y < thumb.y:
                    pyautogui.scroll(10 if index.y < middle.y else -10)

                # Zoom
                if abs(index.x - thumb.x) < 0.03:
                    pyautogui.hotkey('ctrl', '+')
                elif abs(index.x - thumb.x) > 0.1:
                    pyautogui.hotkey('ctrl', '-')

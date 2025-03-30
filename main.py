import cv2
from hand_tracking import HandTracker
from emotion_detection import detect_emotion

def get_emotion_color(emotion):
    colors = {
        "happy": (0, 255, 0),
        "sad": (255, 0, 0),
        "angry": (0, 0, 255),
        "surprise": (255, 255, 0),
        "fear": (255, 0, 255),
        "neutral": (128, 128, 128),
        "disgust": (0, 255, 255)
    }
    return colors.get(emotion, (255, 255, 255)) 

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        tracker.process(frame)
        emotion = detect_emotion(frame)

        color = get_emotion_color(emotion) if emotion else (255, 255, 255)
        frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 10)

        if emotion in ["angry", "sad"]:
            cv2.putText(frame, "Animo Todo estara bien :)",
                        (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Rastreo de Manos + Detección de Emocion", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

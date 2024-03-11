import cv2
import mediapipe as mp
import pyvirtualcam

print('test')

# Initialize Mediapipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize OpenCV
cap = cv2.VideoCapture(0)
width, height = 640, 480
cap.set(3, width)
cap.set(4, height)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
        print(f'Using virtual camera: {cam.device}')
    
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Apply mirror effect
            image = cv2.flip(image, 1)

            # Convert BGR image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process image with Mediapipe Hands
            results = hands.process(image)

            # Draw hand landmarks and bounding boxes on the image
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))

                    # Calculate bounding box coordinates
                    x_min = int(min(hand_landmarks.landmark, key=lambda landmark: landmark.x).x * width)
                    y_min = int(min(hand_landmarks.landmark, key=lambda landmark: landmark.y).y * height)
                    x_max = int(max(hand_landmarks.landmark, key=lambda landmark: landmark.x).x * width)
                    y_max = int(max(hand_landmarks.landmark, key=lambda landmark: landmark.y).y * height)

                    # Draw bounding box
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            cam.send(image)

            # Convert RGB image back to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Show the image
            cv2.imshow('Hand Tracking', image)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Release the capture and destroy the OpenCV windows
cap.release()
cv2.destroyAllWindows()

import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf

# Load the deep learning model
model = tf.keras.models.load_model('deep_learning_model.h5')

# Mediapipe for hand detection
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing_styles = mp.solutions.drawing_styles

# Labels dictionary (adjust as per your dataset)
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
               5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
               15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
               20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
               25: 'Z'}

true_labels = []
predicted_labels = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

            x_, y_, data_hands = [], [], []
            for landmark in hand_landmarks.landmark:
                x, y = landmark.x, landmark.y
                x_.append(x)
                y_.append(y)

            for landmark in hand_landmarks.landmark:
                data_hands.append(landmark.x - min(x_))
                data_hands.append(landmark.y - min(y_))

            # Ensure the data is in the correct format for the deep learning model
            data_hands = np.asarray(data_hands).reshape(1, -1)

            # Change this to the label you want to test
            true_label = 'B'

            # Use the deep learning model for prediction
            prediction = model.predict(data_hands)
            predicted_label = labels_dict[np.argmax(prediction)]

            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

            # Display prediction near the hand
            x1, y1 = int(min(x_) * frame.shape[1]), int(min(y_) * frame.shape[0])
            cv2.putText(frame, predicted_label, (x1, y1), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.putText(frame, "Press 'Esc' to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == 27:  # ASCII code for 'Esc' key
        break

cap.release()
cv2.destroyAllWindows()

# Calculate accuracy
accuracy = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == pred]) / len(true_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')

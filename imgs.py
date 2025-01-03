import os
import cv2

DATA_DIR = './data' 
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 500

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

for class_id in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(class_id))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {class_id}. Press "R" when ready.')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        cv2.putText(frame, 'Ready? Press "R"!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Data Collection', frame)

        if cv2.waitKey(1) & 0xFF == ord('r'):
            break

    print(f"Starting data collection for class {class_id}. Please position yourself.")

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        cv2.putText(frame, f'Collecting: {counter}/{dataset_size}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Data Collection', frame)

        file_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(file_path, frame)

        counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("not done cause u press q")
            break

    print(f"Data collection for class {class_id} completed.")

cap.release()
cv2.destroyAllWindows()

print("done")
# a sign detector app using flask , opencv and tensorflow and mediaPipe

import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import mediapipe as mp
from flask import Response

app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/video')
def video():
    return render_template('video.html')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def generate_frames():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands()  # Initialize Hands within the function to avoid reuse issues
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert the frame to RGB for MediaPipe processing
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Draw hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Encode the frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        # Check if the post request has a file
        if 'file' not in request.files:
            return "No file uploaded", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the image
            processed_path = process_image(file_path)
            return render_template('image.html', uploaded_image=processed_path)

    return render_template('image.html', uploaded_image=None)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def process_image(image_path):
    hands = mp_hands.Hands()
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Save the processed image
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(image_path))
    cv2.imwrite(processed_path, image)
    return processed_path

if __name__ == '__main__':
    app.run(debug=True)
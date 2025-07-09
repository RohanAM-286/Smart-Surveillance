from flask import Flask, render_template, Response
import cv2
import face_recognition
import os
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)
video = cv2.VideoCapture(0)

# Ensure required folders exist
os.makedirs("unknowns", exist_ok=True)
os.makedirs("known_faces", exist_ok=True)

# Load known faces
known_face_encodings = []
known_face_names = []

for file in os.listdir("known_faces"):
    img = face_recognition.load_image_file(f"known_faces/{file}")
    encoding = face_recognition.face_encodings(img)
    if encoding:
        known_face_encodings.append(encoding[0])
        known_face_names.append(os.path.splitext(file)[0])

# Logging detection to CSV
def log_detection(name):
    df = pd.DataFrame([[name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')]], columns=["Name", "Time"])
    if not os.path.exists("log.csv"):
        df.to_csv("log.csv", index=False)
    else:
        df.to_csv("log.csv", mode='a', header=False, index=False)

# Live video stream generator
def generate_frames():
    while True:
        success, frame = video.read()
        if not success:
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect and encode faces
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for encoding, loc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, encoding)
            best_match_index = np.argmin(face_distances)

            face_distance = face_distances[best_match_index]
            if matches[best_match_index] and face_distance < 0.5:
                name = known_face_names[best_match_index]

            log_detection(name)

            # Save image if unknown
            if name == "Unknown":
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                filename = f"unknowns/unknown_{timestamp}.jpg"
                cv2.imwrite(filename, frame)

            # Draw box and label
            top, right, bottom, left = [v * 4 for v in loc]
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Stream to browser
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

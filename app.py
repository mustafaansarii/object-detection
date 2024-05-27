from flask import Flask, render_template, Response, jsonify
import torch
import cv2
import numpy as np
import pyttsx3

app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'yolov5m', force_reload=True, trust_repo=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

engine = pyttsx3.init()

def draw_boxes(img, results):
    detected_objects = []
    for result in results.xyxy[0]:
        xmin, ymin, xmax, ymax = map(int, result[:4])
        label = model.names[int(result[5])]
        confidence = result[4]
        detected_objects.append(label)
        color = (0, 255, 0)  

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img, f"{label} {confidence:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return detected_objects

def speak_detected_objects(detected_objects):
    if detected_objects:
        objects_str = ', '.join(detected_objects)
        text = f"I can see {objects_str}."
        engine.say(text)
        engine.runAndWait()

def generate():
    cap = cv2.VideoCapture(0) 
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (1080, 720))

        results = model(resized_frame)

        detected_objects = draw_boxes(frame, results)

        speak_detected_objects(detected_objects)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_flask_app')
def start_flask_app():
    return jsonify(success=True)

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

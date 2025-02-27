import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import time
import datetime
from pygame import mixer

mixer.init()
sound = mixer.Sound('you are sleeping.wav')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
model = load_model(os.path.join("models", "model.h5"))

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
last_check_time = time.time()
attention_status = "Attentive"
attention_status_list = []
alarm_on = False

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, minNeighbors=3, scaleFactor=1.1, minSize=(25, 25))
    eyes = eye_cascade.detectMultiScale(gray, minNeighbors=1, scaleFactor=1.1)

    cv2.rectangle(frame, (0, height - 50), (350, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    for (x, y, w, h) in eyes:
        eye = frame[y:y + h, x:x + w]
        eye = cv2.resize(eye, (80, 80))
        eye = eye / 255
        eye = eye.reshape(80, 80, 3)
        eye = np.expand_dims(eye, axis=0)
        prediction = model.predict(eye)

        if prediction[0][0] > 0.30:  # Not Attentive
            score += 1
            if score > 50:
                score = 21
        elif prediction[0][1] > 0.70:  # Attentive
            score -= 1
            if score < 0:
                score = 0

    # Update the attention status based on the current score
    if score >= 21:
        attention_status = "Not Attentive"
        if not alarm_on:
            sound.play(-1)  # Play alarm in a loop
            alarm_on = True
    else:
        attention_status = "Attentive"
        if alarm_on:
            sound.stop()  # Stop the alarm
            alarm_on = False

    # Log to file every 10 seconds
    current_time = time.time()
    if current_time - last_check_time >= 10:
        last_check_time = current_time
        attention_status_list.append(attention_status)

        try:
            with open("log_file.txt", "a") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = f"{timestamp} - {attention_status}\n"
                f.write(log_entry)
        except Exception as e:
            print(e)

    cv2.putText(frame, attention_status, (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Score: ' + str(score), (200, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate attentiveness percentage and conclusion at the end of the session
total_checks = len(attention_status_list)
if total_checks > 0:
    attentive_count = attention_status_list.count("Attentive")
    attentiveness_percentage = (attentive_count / total_checks) * 100
    final_conclusion = "User is Active" if attentiveness_percentage > 50 else "User is Not Active"

    print(f"Attentiveness Percentage: {attentiveness_percentage:.2f}%")
    print(f"Final Conclusion: {final_conclusion}")

cap.release()
cv2.destroyAllWindows()
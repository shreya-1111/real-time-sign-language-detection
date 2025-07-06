import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("asl_cnn_model.h5")

# Load labels
with open('asl_labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

IMG_SIZE = 64
cap = cv2.VideoCapture(0)

box_size = 224
box_x = int((640 - box_size) / 2)
box_y = int((480 - box_size) / 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_size, box_y + box_size), (0, 255, 0), 2)

    roi = frame[box_y:box_y+box_size, box_x:box_x+box_size]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE))
    roi_normalized = roi_resized.astype('float32') / 255.0
    roi_reshaped = np.expand_dims(roi_normalized, axis=0)

    prediction = model.predict(roi_reshaped)
    predicted_idx = np.argmax(prediction)
    predicted_class = labels[predicted_idx]
    confidence = np.max(prediction)

    cv2.putText(frame, f"Sign: {predicted_class}", (box_x, box_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Sign Language Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

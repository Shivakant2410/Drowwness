# drowsiness_util.py
import cv2
import torch
import numpy as np
from torchvision import transforms
import mediapipe as mp

def load_model(model_class, model_path, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])

def compute_ear(landmarks, indices):
    def dist(p1, p2):
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
    A = dist(landmarks[indices[1]], landmarks[indices[5]])
    B = dist(landmarks[indices[2]], landmarks[indices[4]])
    C = dist(landmarks[indices[0]], landmarks[indices[3]])
    return (A + B) / (2.0 * C)

def detect_faces_and_drowsiness(frame, model, transform, device):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    sleepy_count = 0
    drowsy_ages = []

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            left_ear = compute_ear(landmarks.landmark, LEFT_EYE_IDX)
            right_ear = compute_ear(landmarks.landmark, RIGHT_EYE_IDX)
            ear = (left_ear + right_ear) / 2.0

            x_coords = [lm.x for lm in landmarks.landmark]
            y_coords = [lm.y for lm in landmarks.landmark]
            x_min, y_min = int(min(x_coords) * w), int(min(y_coords) * h)
            x_max, y_max = int(max(x_coords) * w), int(max(y_coords) * h)

            face_crop = frame[y_min:y_max, x_min:x_max]
            if face_crop.size == 0:
                continue

            input_tensor = transform(face_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                predicted_age = model(input_tensor).item()

            if ear < 0.22:
                sleepy_count += 1
                drowsy_ages.append(int(predicted_age))
                color = (0, 0, 255)
                label = f"Drowsy | Age: {int(predicted_age)}"
            else:
                color = (0, 255, 0)
                label = f"Awake | Age: {int(predicted_age)}"

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, sleepy_count, drowsy_ages

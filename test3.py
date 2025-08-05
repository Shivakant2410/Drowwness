import cv2
import torch
import numpy as np
from torchvision import transforms
from age_model import AgeModel
import mediapipe as mp

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AgeModel().to(device)

# Load trained weights (ensure this matches the model that produced MAE ~5.68)
model.load_state_dict(torch.load("C:/Users/swara/Downloads/age_model.pt", map_location=device))
model.eval()

# Transform for age prediction
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# EAR landmark indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def compute_ear(landmarks, indices):
    def dist(p1, p2):
        return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5
    A = dist(landmarks[indices[1]], landmarks[indices[5]])
    B = dist(landmarks[indices[2]], landmarks[indices[4]])
    C = dist(landmarks[indices[0]], landmarks[indices[3]])
    return (A + B) / (2.0 * C + 1e-6)  # Avoid divide-by-zero

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    sleepy_count = 0
    drowsy_ages = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            try:
                # EAR calculation
                left_ear = compute_ear(face_landmarks.landmark, LEFT_EYE_IDX)
                right_ear = compute_ear(face_landmarks.landmark, RIGHT_EYE_IDX)
                ear = (left_ear + right_ear) / 2.0

                # Get bounding box
                x_coords = [lm.x for lm in face_landmarks.landmark]
                y_coords = [lm.y for lm in face_landmarks.landmark]
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

                # Draw
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            except Exception as e:
                print(f"Error in face loop: {e}")
                continue

    cv2.imshow("Real-Time Age + Drowsiness Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

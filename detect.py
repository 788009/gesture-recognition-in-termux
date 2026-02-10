import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import yaml
import os

# --- 1. Load Configuration ---
def load_config(config_path='config.yaml'):
    # Default configuration as a fallback
    default_config = {
        'model': {
            'path': './model.pth',
            'classes': ['fist', 'five', 'one', 'other', 'two']
        },
        'inference': {
            'stream_url': 'http://127.0.0.1:8080/live.m3u8'
        },
        'ui': {
            'window_name': 'Hand Detection System',
            'display_width': 960
        }
    }
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return default_config

config = load_config()

# Extracting variables from config
CLASSES = config['model']['classes']
MODEL_PATH = config['model']['path']
STREAM_URL = config['inference']['stream_url']
WINDOW_NAME = config['ui']['window_name']
DISPLAY_WIDTH = config['ui']['display_width']

# --- 2. Define Model Structure ---
class GestureMLP_v2(nn.Module):
    def __init__(self, input_size=42, num_classes=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

# --- 3. Landmark Preprocessing ---
def preprocess_landmarks(landmark_list):
    temp_landmark_list = np.array(landmark_list).reshape(-1, 2)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    temp_landmark_list = temp_landmark_list - [base_x, base_y]
    max_value = np.abs(temp_landmark_list).max()
    if max_value != 0:
        temp_landmark_list = temp_landmark_list / max_value
    return temp_landmark_list.flatten()

# --- 4. Initialize MediaPipe and Load Model ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureMLP_v2(num_classes=len(CLASSES)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- 5. Open Window ---
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# --- 6. Real-time Inference ---
cap = cv2.VideoCapture(STREAM_URL)
print(f"Connecting to video stream: {STREAM_URL}...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Stream disconnected or failed to read.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            coords = [[lm.x, lm.y] for lm in hand_lms.landmark]
            processed_input = preprocess_landmarks(coords)
            input_tensor = torch.FloatTensor(processed_input).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.softmax(output, dim=1)
                conf, pred_idx = torch.max(prediction, 1)
                gesture_name = CLASSES[pred_idx.item()]
                confidence = conf.item()

            color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255)
            text = f"{gesture_name} {confidence:.1%}"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Scaling display
    screen_h, screen_w = frame.shape[:2]
    display_h = int(screen_h * (DISPLAY_WIDTH / screen_w))
    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, display_h))

    cv2.imshow(WINDOW_NAME, display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program exited.")
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow import keras
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load your trained CNN+GRU model
def load_model():
    # Load your model here
    model = keras.models.load_model('/Users/tejaswinibharatha/Downloads/Human Activity Recognition using TensorFlow (CNN + LSTM) Code/best_model_300 (1).keras')
    return model

# Load the CNN+GRU model
model = load_model()

# Function to preprocess landmarks for the CNN+GRU model
def preprocess_landmarks(landmarks):
    # Flatten landmarks and normalize or other necessary preprocessing
    return np.array([lm.x for lm in landmarks] + [lm.y for lm in landmarks] + [lm.z for lm in landmarks])

def get_thumb_orientation(landmarks, image_width):
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    thumb_cmc = landmarks[mp_hands.HandLandmark.THUMB_CMC]
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]

    # Normalize by image width to account for different sizes
    wrist_x, thumb_cmc_x, thumb_tip_x = wrist.x * image_width, thumb_cmc.x * image_width, thumb_tip.x * image_width

    # Determine the orientation
    if thumb_tip_x < wrist_x and thumb_cmc_x < wrist_x:
        return "Left"
    elif thumb_tip_x > wrist_x and thumb_cmc_x > wrist_x:
        return "Right"
    else:
        return "Centered"

# Assuming your model expects sequences of 10 time steps
sequence_length = 10
frame_buffer = deque(maxlen=sequence_length)
CONFIDENCE_THRESHOLD = 0.80

# Define global variables
gloss_show = 'Word: none'
labels = {
    0: 'again',
    1: 'book',
    2: 'hello',
    3: 'help',
    4: 'maybe',
    5: 'play',
    6: 'stop',
    7: 'with',
    8: 'cat',
    9: 'dog',
    10: 'color',
    11: 'I',
    12: 'like'
}

# Function to preprocess frame for the ASL model
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224)) / 255.0
    return np.expand_dims(resized_frame, axis=0)

# Function to make ASL prediction
def make_asl_prediction(frame):
    global gloss_show

    processed_frame = preprocess_frame(frame)
    prediction = model.predict(np.expand_dims(processed_frame, axis=0))[0]
    best_pred_idx = np.argmax(prediction)
    best_pred_accuracy = prediction[best_pred_idx]

    highest_prediction = np.max(prediction)
    if highest_prediction < CONFIDENCE_THRESHOLD:
        display_text = "Not a sign"
    else:
        predicted_sign = np.argmax(prediction)
        display_text = f"Predicted sign: {labels.get(predicted_sign, 'Unknown')}"

    if best_pred_accuracy > CONFIDENCE_THRESHOLD:
        gloss = labels.get(best_pred_idx, "Unknown")
        gloss_show = f"Word: {gloss}  {best_pred_accuracy * 100:.2f}%"
    else:
        gloss_show = "Word: none"

    return display_text, gloss_show

def main():
    cap = cv2.VideoCapture(0)
    previous_thumb_orientation = None
    sign_change_detected = False

    st.title("ASL Recognition")
    video_placeholder = st.empty()

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_width = image.shape[1]
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                thumb_orientation = get_thumb_orientation(hand_landmarks.landmark, image_width)

                if previous_thumb_orientation and thumb_orientation != previous_thumb_orientation:
                    sign_change_detected = True 
                    print("Sign changed",thumb_orientation)
                    cv2.putText(image, f"Sign Changed", (image.shape[1] - 200, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                previous_thumb_orientation = thumb_orientation
                cv2.putText(image, f"Thumb Orientation: {thumb_orientation}", (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


                processed_landmarks = preprocess_landmarks(hand_landmarks.landmark)
                frame_buffer.append(processed_landmarks)

                if len(frame_buffer) == sequence_length:
                
                    display_text, gloss_show = make_asl_prediction(image)
                    cv2.putText(image, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, gloss_show, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        video_placeholder.image(image, channels="BGR", use_column_width=True)

if __name__ == '__main__':
    main()

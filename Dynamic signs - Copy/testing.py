import threading
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
gloss_show = 'Word: none'
mp_hands = mp.solutions.hands
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
    10:'color',
    11: 'I',
    12: 'like'
    }

def main():
    dim = (224, 224)
    frames = 10
    channels = 3
    model_path = '/Users/tejaswinibharatha/Downloads/Human Activity Recognition using TensorFlow (CNN + LSTM) Code/saved_models/best_model_attention.keras'
    # model_path = 'C:/Users/yutik/Downloads/best_model.h5'
    threshold = 0.30

#     # define empty buffer
#     frame_buffer = np.empty((0, *dim, channels))

#     print("loading ASL detection model ...")
#     # load model
#     model = load_model(model_path)

#     print("starting video stream ...")
#     # start the video stream
#     cap = cv2.VideoCapture(0)
#     # cap.set(cv2.CAP_PROP_FPS, 25)  # set the FPS to 25

#     x = threading.Thread()
#     # loop over the frames
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             # process the frame
#             frame_res = cv2.resize(frame, dim)
#             frame_res = frame_res / 255.0
#             # append the frame to buffer
#             frame_resh = np.reshape(frame_res, (1, *frame_res.shape))
#             frame_buffer = np.append(frame_buffer, frame_resh, axis=0)
#             # start sign recognition only if the buffer is full
#             if frame_buffer.shape[0] == frames:
#                 # make the prediction
#                 if not x.is_alive():
#                     x = threading.Thread(target=make_prediction, args=(
#                         frame_buffer, model, threshold))
#                     x.start()
#                 else:
#                     pass
#                 # left-shift of the buffer
#                 frame_buffer = frame_buffer[1:frames]
#                 # show label
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 cv2.putText(frame, gloss_show, (20, 450), font, 1, (0, 255, 0),
#                             2, cv2.LINE_AA)
#                 cv2.imshow('frame', frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             break

#     cap.release()


# def make_prediction(frame_buffer, model, threshold):
#     global gloss_show

#     frame_buffer_resh = frame_buffer.reshape(1, *frame_buffer.shape)
#     # model prediction
#     predictions = model.predict(frame_buffer_resh)[0]
#     # get the best prediction
#     best_pred_idx = np.argmax(predictions)
#     print(best_pred_idx)
#     acc_best_pred = predictions[best_pred_idx]
#     print(acc_best_pred)
#     # check mislabeling
#     if acc_best_pred > threshold:
#         gloss = labels[best_pred_idx]
#         gloss_show = "Word: {: <3}  {:.2f}% ".format(gloss,acc_best_pred * 100)
#         print(gloss_show)
#     else:
#         gloss_show = 'Word: none'
    

    # Define the empty buffer more succinctly
    frame_buffer = np.empty((0, *dim, channels))

    print("Loading ASL detection model...")
    # Load the model in a more Pythonic way
    model = load_model(model_path)

    print("Starting video stream...")
    # Simplify video capture start and option setting
    cap = cv2.VideoCapture(1)
    # cap.set(cv2.CAP_PROP_FPS, 25)  # Optional: set FPS to 25 if needed
    # Get the default video FPS and size to ensure compatibility
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # result = cv2.VideoCapture('test.avi',cv2.VideoWriter_fourcc(*'MJPG'),10,dim)
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width,frame_height))

    def process_frame(frame):
        """Resize and normalize frame."""
        frame_resized = cv2.resize(frame, dim) / 255.0
        return np.reshape(frame_resized, (1, *frame_resized.shape))

    def make_prediction():
        global gloss_show

        frame_buffer_reshaped = frame_buffer.reshape(1, *frame_buffer.shape)
        predictions = model.predict(frame_buffer_reshaped)[0]
        best_pred_idx = np.argmax(predictions)
        best_pred_accuracy = predictions[best_pred_idx]

        if best_pred_accuracy > threshold:
            gloss = labels[best_pred_idx]
            gloss_show = f"Word: {gloss}  {best_pred_accuracy * 100:.2f}%"
        else:
            gloss_show = "Word: none"
        print(gloss_show)

    # Initialize an empty thread
    x = threading.Thread()
    try:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process and buffer the frame
            frame_processed = process_frame(frame)
            frame_buffer = np.append(frame_buffer, frame_processed, axis=0)

            # Prediction and buffer management
            if frame_buffer.shape[0] == frames:
                if not x.is_alive():
                    x = threading.Thread(target=make_prediction)
                    x.start()

                # Keep the buffer filled with the most recent frames
                frame_buffer = frame_buffer[1:]

            # Display the frame and prediction
            cv2.putText(frame, gloss_show, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame)   
            out.write(frame)     

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:

        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

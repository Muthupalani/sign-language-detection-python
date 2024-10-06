import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Start video capture from camera
cap = cv2.VideoCapture(0)  # Change to 1 or 2 if you're using an external camera

# Mediapipe setup for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

# Labels for classification
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'L'}  # Ensure your model has the correct labels

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame")
        break

    # Convert to RGB for mediapipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with hand landmarks
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        data_aux = []
        x_, y_ = [], []
        
        # Loop over the detected landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Normalize the landmark coordinates
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            # Ensure we have the correct number of features (84)
            if len(data_aux) == 84:
                # Print the input features for debugging
                print("Input features:", data_aux)
                
                # Make prediction
                prediction = model.predict([np.asarray(data_aux)])
                
                # Print the prediction result for debugging
                print("Raw prediction:", prediction)

                predicted_character = labels_dict[int(prediction[0])]

                # Get bounding box around the hand
                H, W, _ = frame.shape
                x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10

                # Draw rectangle and prediction text
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, predicted_character, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Sign Language Detector', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

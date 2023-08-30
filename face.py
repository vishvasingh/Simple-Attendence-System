import face_recognition
import cv2
import dlib
import numpy as np

# Load sample images and encode them
image_of_person1 = face_recognition.load_image_file('C:/Users/vishv/OneDrive/Documents/code/code/madhav.jpg')
person1_encoding = face_recognition.face_encodings(image_of_person1)[0]

image_of_person2 = face_recognition.load_image_file('C:/Users/vishv/OneDrive/Documents/code/code/richa.jpg')
person2_encoding = face_recognition.face_encodings(image_of_person2)[0]

image_of_person3 = face_recognition.load_image_file('C:/Users/vishv/OneDrive/Documents/code/code/pallavi.jpg')
person3_encoding = face_recognition.face_encodings(image_of_person3)[0]

image_of_person4 = face_recognition.load_image_file("C:/Users/vishv/OneDrive/Documents/code/code/obama.jpeg")
person4_encoding = face_recognition.face_encodings(image_of_person4)[0]

image_of_person5 = face_recognition.load_image_file("C:/Users/vishv/OneDrive/Documents/code/code/biden.jpeg")
person5_encoding = face_recognition.face_encodings(image_of_person5)[0]

# Create a list of known face encodings and corresponding names
known_face_encodings = [person1_encoding, person2_encoding, person3_encoding, person4_encoding, person5_encoding]
known_face_names = ["Madhav", "Richa", "Pallavi", "Barack Obama", "Joe Biden"]

# Initialize the video capture
video_capture = cv2.VideoCapture(0)

# Set desired frame width and height
frame_width = 640
frame_height = 480
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Load the shape predictor for face alignment
shape_predictor_path = "C:/Users/vishv/Downloads/archive/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(shape_predictor_path)

# Variables for optimization
face_detection_interval = 5  # Perform face detection every 5 frames
frame_counter = 0

# Initialize face detection variables
face_locations = []
face_encodings = []

while True:
    # Capture a single frame from the video stream
    ret, frame = video_capture.read()

    if not ret:
        break

    frame_counter += 1

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame_counter % face_detection_interval == 0:
        # Apply histogram equalization to enhance contrast
        equalized_frame = cv2.equalizeHist(gray_frame)

        # Convert the equalized frame back to color (if needed)
        equalized_frame_color = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2BGR)
        # Convert the frame from BGR color (OpenCV default) to RGB color
        rgb_frame = cv2.cvtColor(equalized_frame_color, cv2.COLOR_BGR2RGB)

        # Find all faces and their encodings in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Iterate over the detected faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding against known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the index of the first matching known face
        if True in matches:
            matched_index = matches.index(True)
            name = known_face_names[matched_index]

        # Extract the face region of interest
        face_roi = gray_frame[top:bottom, left:right]

        # Detect facial landmarks for the current face
        landmarks = predictor(face_roi, dlib.rectangle(0, 0, face_roi.shape[1], face_roi.shape[0]))
        landmarks_points = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Define the desired eye corner points for alignment
        desired_left_eye = (0.35, 0.35)
        desired_right_eye = (0.65, 0.35)

        # Compute the transformation matrix for alignment
        src_pts = landmarks_points[:3]
        dst_pts = np.array([
            [desired_left_eye[0] * face_roi.shape[1], desired_left_eye[1] * face_roi.shape[0]],
            [desired_right_eye[0] * face_roi.shape[1], desired_right_eye[1] * face_roi.shape[0]],
            [desired_right_eye[0] * face_roi.shape[1], desired_left_eye[1] * face_roi.shape[0]]
        ])
        transformation_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        # Apply the transformation to align the face
        aligned_face = cv2.warpAffine(face_roi, transformation_matrix, (face_roi.shape[1], face_roi.shape[0]))
        # Resize the aligned face to a fixed size (e.g., 150x150)
        aligned_face = cv2.resize(aligned_face, (200, 200))

        # Display the aligned face (for debugging purposes)
        cv2.imshow("Aligned Face", aligned_face)

        # Draw a rectangle around the face and label it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
video_capture.release()
cv2.destroyAllWindows()

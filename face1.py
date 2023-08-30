import face_recognition
import cv2

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
known_face_names = ["Madhav", "Richa", "Pallavi", "Barack Obama",
                    "Joe Biden"]

# Initialize the video capture
video_capture = cv2.VideoCapture(0)
known_faces = [(enc, name) for enc, name in zip(known_face_encodings, known_face_names)]
while True:
    # Capture a single frame from the video stream
    ret, frame = video_capture.read()

    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

# Simple-Attendance-System
face recognition  and OpenCV-based attendance system

Download the Frontal Face Cascade File from the below GitHub Repo-
https://github.com/anaustinbeing/haar-cascade-files
Upload the images as much as you want to create automatic dataset generation and encoding generation
"
image_of_person1 = face_recognition.load_image_file('PAth/to/image.jpg')
person1_encoding = face_recognition.face_encodings(image_of_person1)[0]

"
Maintain  the sequence and change the known faces names list according to the encoding and images you are uploading in order.
"
known_face_encodings = [person1_encoding, person2_encoding, person3_encoding, person4_encoding, person5_encoding]
known_face_names = ["Madhav", "Richa", "Pallavi", "Barack Obama", "Joe Biden"]

"
If you don't have an attendance file it will create an attendence.csv file.
0 in video capture tells  to capture on default camera change according to your camera number

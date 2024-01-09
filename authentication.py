import cv2

# Load pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize camera
cap = cv2.VideoCapture(0)
# Function to perform authentication
def authenticate():
    saved_face = None
    authenticated = False
    while not authenticated:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Display the captured frame
            cv2.imshow('Face Authentication', frame)
            key = cv2.waitKey(1)
            if key == ord('s') and saved_face is None:
                saved_face = frame[y:y+h, x:x+w]
                cv2.imwrite('saved_face.jpg', saved_face)
                print("Saved detected face as 'saved_face.jpg'")
            if key == ord('c') and saved_face is not None:
                current_face = frame[y:y+h, x:x+w]
                # Perform face comparison here (e.g., using face recognition libraries)
                # For demonstration purposes, just show a message indicating successful authentication
                if compare_faces(saved_face, current_face):
                    print("Authentication Successful!")
                    authenticated = True
                    break
                else:
                    print("Authentication Failed!")
            # Perform authentication logic 
            # For demo purposes, I add a key press to simulate authentication
            if key == ord('a'):
                authenticated = True
                print("Authentication Successful! just for testing purpose")
                break
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
def compare_faces(face1, face2):
    # This is a placeholder function for face comparison
    # In reality, you'd use a face recognition library or algorithm to compare faces
    # Here, just comparing the sizes of the saved and current face images
    return face1.shape == face2.shape
# Start authentication process
authenticate()

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
import torch
import cv2

from picamera2 import Picamera2
from model import FERModel
from lcd import lcd_init, lcd_string, LCD_LINE_1, LCD_LINE_2


# labels relative to FER2013
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# init lcd screen
lcd_init()
lcd_string("Starting...", LCD_LINE_1)
lcd_string("", LCD_LINE_2)

# init model
model = FERModel(len(labels))
model.load_state_dict(torch.load("weights.pt", map_location=torch.device('cpu')))
model.eval()


if __name__ == "__main__":
    with Picamera2() as camera:
        # camera configuration and start
        camera_config = camera.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
        camera.configure(camera_config)
        camera.start()

        while True:
            # capture frame and grayscale
            frame = camera.capture_array("main")
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces
            face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.15, minNeighbors=5)

            for (x, y, w, h) in faces:
                # crop face
                face = gray_frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (48, 48))

                face_normalized = face_resized / 255.0
                face_reshaped = torch.tensor(face_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
               
                # predict emotion from face
                with torch.no_grad():
                    prediction = model(face_reshaped)
                    emotion = labels[torch.argmax(prediction).item()]

                lcd_string(emotion, LCD_LINE_1)

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

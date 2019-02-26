import cv2


# Find faces in frame
def detect_faces(f_cascade, colored_img, scaleFactor=1.1):
    img_copy = colored_img.copy()

    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)

    # Draw box and print out coordinates
    for (x, y, w, h) in faces:
        print(f"X: {x}, Y: {y}")
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return img_copy

cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

cv2.namedWindow("ur face")

# Use webcam as image source
cam = cv2.VideoCapture(0)
if cam.isOpened():
    rval, frame = cam.read()
else:
    rval = False

while rval:
    frame = detect_faces(cascade, frame)
    cv2.imshow("ur face", frame)
    rval, frame = cam.read()
    key = cv2.waitKey(20)
    if key == 27:
        break

cv2.destroyAllWindows()
cam.release()

import cv2
import mediapipe as mp

def detect_face(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform face detection
    results = face_detection.process(image_rgb)

    # Extract bounding box coordinates
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            return bbox
    else:
        print("No face detected in the image.")
        return None

if __name__ == "__main__":
    image_path = input("Enter the path to the image: ")
    face_bbox = detect_face("WIN_20220413_14_00_37_Pro.jpg")
    # ok yea, the heatmap can't see the face.
    if face_bbox:
        print("Face bounding box coordinates (xmin, ymin, width, height):", face_bbox)

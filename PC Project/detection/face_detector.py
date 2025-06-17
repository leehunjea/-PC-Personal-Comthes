import dlib

class FaceDetector:
    def __init__(self, predictor_path="C:/Users/AI-LHJ/Desktop/PC Project/detection/shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect_face(self, image):
        faces = self.detector(image)
        if len(faces) == 0:
            raise ValueError("No face detected.")
        landmarks = self.predictor(image, faces[0])
        return faces[0], landmarks

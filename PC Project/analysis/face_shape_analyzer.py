# analysis/face_shape_analyzer.py
import dlib
import numpy as np

class FaceShapeAnalyzer:
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def analyze(self, image):
        faces = self.detector(image)
        if len(faces) == 0:
            return "Unknown"
        landmarks = self.predictor(image, faces[0])
        # 예시: 얼굴형을 간단히 분류 (실제 구현은 더 정교하게)
        jaw_width = landmarks.part(16).x - landmarks.part(0).x
        face_height = landmarks.part(8).y - landmarks.part(27).y
        ratio = jaw_width / face_height
        if ratio > 1.5:
            return "Round"
        elif ratio < 1.2:
            return "Oval"
        else:
            return "Square"

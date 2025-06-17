import dlib
import numpy as np
import cv2 # OpenCV는 ROI 시각화 또는 이미지 처리에 사용될 수 있음

# dlib 모델 파일 경로 (models 폴더 안에 있다고 가정)
DLIB_LANDMARK_MODEL_PATH = "C:/Users/AI-LHJ/Desktop/PC Project/analysis/PersonalColorAnalyzerModel/personalcolor_ai/models/shape_predictor_68_face_landmarks.dat"

class FaceDetector:
    def __init__(self, landmark_model_path=DLIB_LANDMARK_MODEL_PATH):
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(landmark_model_path)
        except RuntimeError as e:
            print(f"Error loading dlib models: {e}")
            print(f"Ensure '{landmark_model_path}' exists.")
            self.detector = None
            self.predictor = None

    def detect_main_face_and_landmarks(self, image_bgr):
        """
        이미지에서 가장 큰 얼굴 하나와 해당 얼굴의 랜드마크를 감지합니다.
        image_bgr: OpenCV BGR 이미지 (numpy array)
        반환: (감지된 얼굴 영역 객체, 랜드마크 객체) 또는 (None, None)
        """
        if self.detector is None or self.predictor is None:
            print("Dlib models not loaded. Face detection cannot proceed.")
            return None, None

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if not faces:
            return None, None

        # 가장 큰 얼굴 선택 (일반적으로 가장 중요한 얼굴)
        main_face = max(faces, key=lambda rect: rect.width() * rect.height())
        landmarks = self.predictor(gray, main_face)
        return main_face, landmarks

    def _get_cheek_regions(self, landmarks, image_shape):
        """
        랜드마크로부터 양쪽 뺨 영역의 좌표를 추출합니다.
        논문이나 일반적인 기준에 따라 영역을 정의합니다. (여기서는 예시 영역)
        반환: (왼쪽 뺨 좌표_np_array, 오른쪽 뺨 좌표_np_array)
        """
        # 왼쪽 뺨 (예시: 랜드마크 1, 2, 3, 4, 31, 48번을 잇는 다각형)
        # dlib 랜드마크 인덱스는 0부터 시작
        cheek_left_indices = [1, 2, 3, 4, 31, 48] 
        # 오른쪽 뺨 (예시: 랜드마크 15, 14, 13, 12, 35, 54번을 잇는 다각형)
        cheek_right_indices = [15, 14, 13, 12, 35, 54]

        cheek_points_left = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in cheek_left_indices], dtype=np.int32)
        cheek_points_right = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in cheek_right_indices], dtype=np.int32)
        
        return cheek_points_left, cheek_points_right

    def _get_eye_regions(self, landmarks):
        """
        랜드마크로부터 양쪽 눈동자 영역의 좌표를 추출합니다. (여기서는 눈 전체 영역)
        반환: (왼쪽 눈 좌표_np_array, 오른쪽 눈 좌표_np_array)
        """
        # 왼쪽 눈 (랜드마크 36-41)
        eye_left_indices = list(range(36, 42))
        # 오른쪽 눈 (랜드마크 42-47)
        eye_right_indices = list(range(42, 48))

        eye_points_left = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_left_indices], dtype=np.int32)
        eye_points_right = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_right_indices], dtype=np.int32)

        return eye_points_left, eye_points_right

    def _get_hair_region(self, landmarks, image_shape):
        """
        랜드마크로부터 머리카락 영역의 좌표를 추출합니다. (이마 위쪽 영역으로 단순화)
        이 부분은 정확한 추출이 매우 어렵고, 다양한 방법론이 필요합니다.
        여기서는 이마 위쪽 사각형 영역으로 매우 단순화합니다.
        반환: 머리카락 영역 좌표_np_array
        """
        # 이마 영역 (예시: 랜드마크 19, 20, 21, 22, 23, 24번 위쪽)
        forehead_top_y = min(landmarks.part(i).y for i in range(19, 25))
        forehead_left_x = landmarks.part(17).x
        forehead_right_x = landmarks.part(26).x

        # 머리카락 영역을 이마 위로 단순하게 가정
        hair_top_y = max(0, forehead_top_y - int((landmarks.part(8).y - forehead_top_y) * 0.8)) # 이마 높이의 80%만큼 위로
        hair_bottom_y = forehead_top_y
        hair_left_x = max(0, forehead_left_x - 10)
        hair_right_x = min(image_shape[1], forehead_right_x + 10)

        if hair_top_y >= hair_bottom_y or hair_left_x >= hair_right_x: # 영역이 유효하지 않으면 None 반환
            return None

        hair_points = np.array([
            (hair_left_x, hair_top_y),
            (hair_right_x, hair_top_y),
            (hair_right_x, hair_bottom_y),
            (hair_left_x, hair_bottom_y)
        ], dtype=np.int32)
        return hair_points

    def extract_facial_features_roi(self, image_bgr, landmarks):
        """
        주어진 랜드마크를 사용하여 피부(뺨), 눈, 머리카락 ROI 좌표를 반환합니다.
        """
        if landmarks is None:
            return None, None, None, None, None

        image_shape = image_bgr.shape
        cheek_points_left, cheek_points_right = self._get_cheek_regions(landmarks, image_shape)
        eye_points_left, eye_points_right = self._get_eye_regions(landmarks)
        hair_points = self._get_hair_region(landmarks, image_shape)

        return cheek_points_left, cheek_points_right, eye_points_left, eye_points_right, hair_points
# personalcolor_ai/core.py

import numpy as np
from personalcolor_ai.utils import preprocess_image, get_roi_pixels, visualize_landmarks_and_roi, convert_to_lab
from personalcolor_ai.detect_face import FaceDetector
from personalcolor_ai.color_extract import ColorExtractor
from personalcolor_ai.tone_analysis import ColorAnalyzer, SeasonClassifier

class PersonalColorAI:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.color_extractor = ColorExtractor(n_clusters=1)
        self.color_analyzer = ColorAnalyzer()
        self.season_classifier = SeasonClassifier()

    def analyze(self, image_path, visualize=False):
        try:
            image_bgr = preprocess_image(image_path)
            if image_bgr is None:
                return {"error": "이미지 로드 또는 전처리 실패"}

            face_rect, landmarks = self.face_detector.detect_main_face_and_landmarks(image_bgr)
            if landmarks is None:
                return {"error": "얼굴 또는 랜드마크 감지 실패"}

            cheek_l_pts, cheek_r_pts, eye_l_pts, eye_r_pts, hair_pts = \
                self.face_detector.extract_facial_features_roi(image_bgr, landmarks)

            if visualize:
                visualize_landmarks_and_roi(image_bgr, landmarks, cheek_l_pts, cheek_r_pts, eye_l_pts, eye_r_pts, hair_pts)

            skin_pixels = []
            if cheek_l_pts is not None:
                skin_pixels.extend(get_roi_pixels(image_bgr, cheek_l_pts))
            if cheek_r_pts is not None:
                skin_pixels.extend(get_roi_pixels(image_bgr, cheek_r_pts))
            skin_pixels_bgr = np.array(skin_pixels) if skin_pixels else np.array([])

            eye_pixels = []
            if eye_l_pts is not None:
                eye_pixels.extend(get_roi_pixels(image_bgr, eye_l_pts))
            if eye_r_pts is not None:
                eye_pixels.extend(get_roi_pixels(image_bgr, eye_r_pts))
            eye_pixels_bgr = np.array(eye_pixels) if eye_pixels else np.array([])

            hair_pixels_bgr = get_roi_pixels(image_bgr, hair_pts) if hair_pts is not None else np.array([])

            skin_roi_pixels_lab = None
            if skin_pixels_bgr.size > 0:
                try:
                    reshaped = skin_pixels_bgr.astype(np.uint8).reshape(1, -1, 3)
                    lab_image = convert_to_lab(reshaped)
                    skin_roi_pixels_lab = lab_image.reshape(-1, 3)
                except Exception as e:
                    print(f"피부 ROI LAB 변환 오류: {e}")
                    skin_roi_pixels_lab = None

            roi_map = {
                'skin_pixels': skin_pixels_bgr,
                'eye_pixels': eye_pixels_bgr,
                'hair_pixels': hair_pixels_bgr
            }

            feature_colors_bgr = self.color_extractor.extract_feature_colors(image_bgr, roi_map)
            if feature_colors_bgr.get('skin') is None:
                return {"error": "피부색 추출 실패"}

            color_attributes = self.color_analyzer.analyze_color_attributes(feature_colors_bgr, skin_roi_pixels_lab)
            if 'error' in color_attributes:
                return {"error": color_attributes['error']}

            season_type = self.season_classifier.classify(color_attributes)

            return {
                "extracted_colors_bgr": feature_colors_bgr,
                "color_attributes": color_attributes,
                "personal_color_season": season_type,
                "image_shape_processed": image_bgr.shape
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

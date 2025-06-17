import argparse
import cv2 # 이미지 로딩 및 시각화용
import numpy as np
import os # sys.path 조작을 위해 (VSCode에서 직접 실행 시 필요할 수 있음)
import sys # sys.path 조작을 위해
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# VSCode 등에서 F5(디버그 실행) 또는 재생 버튼으로 직접 실행 시
# 'personalcolor_ai' 패키지를 찾지 못하는 문제를 해결하기 위함.
# launch.json에서 "module"을 사용하면 이 부분이 필요 없을 수 있음.
# 현재 파일의 디렉토리 (personalcolor_ai)의 부모 디렉토리(PersonalColorAnalyzerModel)를 sys.path에 추가
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# if parent_dir not in sys.path:
#    sys.path.insert(0, parent_dir)


from personalcolor_ai.utils import preprocess_image, get_roi_pixels, visualize_landmarks_and_roi, convert_to_lab
from personalcolor_ai.detect_face import FaceDetector
from personalcolor_ai.color_extract import ColorExtractor
from personalcolor_ai.tone_analysis import ColorAnalyzer, SeasonClassifier

class PersonalColorAI:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.color_extractor = ColorExtractor(n_clusters=1) # 각 ROI에서 하나의 대표색상 추출
        self.color_analyzer = ColorAnalyzer()
        self.season_classifier = SeasonClassifier()

    def analyze(self, image_path, visualize=False):
        """
        이미지 경로를 받아 퍼스널컬러를 분석합니다.
        """
        try:
            # 1. 이미지 로드 및 전처리
            image_bgr = preprocess_image(image_path) # BGR 포맷으로 로드됨
            if image_bgr is None: # preprocess_image 내부에서 예외처리하지만, 한번 더 확인
                return {"error": "이미지 로드 또는 전처리 실패"}

            # 2. 얼굴 감지 및 랜드마크 추출
            face_rect, landmarks = self.face_detector.detect_main_face_and_landmarks(image_bgr)
            if landmarks is None:
                return {"error": "얼굴 또는 랜드마크 감지 실패"}

            # 3. 주요 특징 ROI 좌표 추출 (피부(뺨), 눈, 머리카락)
            cheek_l_pts, cheek_r_pts, eye_l_pts, eye_r_pts, hair_pts = \
                self.face_detector.extract_facial_features_roi(image_bgr, landmarks)
            
            if visualize: # 시각화 옵션이 켜져있으면 랜드마크와 ROI 영역 표시
                visualize_landmarks_and_roi(image_bgr, landmarks, cheek_l_pts, cheek_r_pts, eye_l_pts, eye_r_pts, hair_pts)

            # 4. 각 ROI에서 픽셀 데이터 추출 (BGR 형식)
            skin_pixels_list_collected = []
            if cheek_l_pts is not None and len(cheek_l_pts) > 0 :
                skin_pixels_list_collected.extend(get_roi_pixels(image_bgr, cheek_l_pts))
            if cheek_r_pts is not None and len(cheek_r_pts) > 0:
                skin_pixels_list_collected.extend(get_roi_pixels(image_bgr, cheek_r_pts))
            
            skin_pixels_bgr = np.array(skin_pixels_list_collected) if skin_pixels_list_collected else np.array([])

            eye_pixels_list_collected = []
            if eye_l_pts is not None and len(eye_l_pts) > 0:
                eye_pixels_list_collected.extend(get_roi_pixels(image_bgr, eye_l_pts))
            if eye_r_pts is not None and len(eye_r_pts) > 0:
                eye_pixels_list_collected.extend(get_roi_pixels(image_bgr, eye_r_pts))
            eye_pixels_bgr = np.array(eye_pixels_list_collected) if eye_pixels_list_collected else np.array([])

            hair_pixels_bgr = get_roi_pixels(image_bgr, hair_pts) if hair_pts is not None and len(hair_pts) > 0 else np.array([])
            
            # 4.1 피부 ROI 픽셀들을 LAB로 변환 (ColorAnalyzer의 청탁도 분석용)
            skin_roi_pixels_lab = None
            if skin_pixels_bgr.size > 0: # skin_pixels_bgr이 비어있지 않은지 확인
                # skin_pixels_bgr은 (N, 3) 형태의 BGR 픽셀 배열
                # cvtColor는 이미지 형태 (H, W, C) 또는 (1, N, C) 등을 기대
                # 따라서 (1, num_pixels, 3) 형태로 reshape
                try:
                    reshaped_skin_bgr = skin_pixels_bgr.astype(np.uint8).reshape(1, -1, 3)
                    lab_image_skin_roi = convert_to_lab(reshaped_skin_bgr) # utils.convert_to_lab 사용
                    skin_roi_pixels_lab = lab_image_skin_roi.reshape(-1, 3) # 다시 (N, 3) 형태로
                except Exception as e:
                    print(f"피부 ROI LAB 변환 중 오류: {e}")
                    skin_roi_pixels_lab = None # 오류 발생 시 None으로 설정

            roi_map_for_extraction = {
                'skin_pixels': skin_pixels_bgr,
                'eye_pixels': eye_pixels_bgr,
                'hair_pixels': hair_pixels_bgr
            }
            
            # 5. 대표 색상 추출 (BGR)
            feature_colors_bgr = self.color_extractor.extract_feature_colors(image_bgr, roi_map_for_extraction)
            if feature_colors_bgr.get('skin') is None:
                 return {"error": "피부색 추출 실패. ROI가 너무 작거나 없습니다."}

            # 6. 색상 속성 분석 (웜/쿨, 명도, 채도, 청탁 등)
            # 피부 ROI의 LAB 픽셀 정보를 ColorAnalyzer에 전달
            color_attributes = self.color_analyzer.analyze_color_attributes(feature_colors_bgr, skin_roi_pixels_lab)
            if 'error' in color_attributes:
                return {"error": color_attributes['error']}

            # 7. 계절 분류
            season_type = self.season_classifier.classify(color_attributes)

            return {
                "extracted_colors_bgr": feature_colors_bgr,
                "color_attributes": color_attributes,
                "personal_color_season": season_type,
                "image_shape_processed": image_bgr.shape # 디버깅용 추가 정보
            }

        except FileNotFoundError as e:
            print(f"오류: {e}")
            return {"error": str(e)}
        except RuntimeError as e: # dlib 모델 로드 실패 등
            print(f"런타임 오류: {e}")
            return {"error": str(e)}
        except Exception as e:
            print(f"분석 중 예상치 못한 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"일반 분석 오류: {str(e)}"}


def main():
    parser = argparse.ArgumentParser(description="AI 기반 퍼스널컬러 진단 시스템")
    parser.add_argument("image_path", type=str, help="분석할 이미지 파일 경로")
    parser.add_argument("--visualize", "-v", action='store_true', help="랜드마크 및 ROI 시각화 여부")
    args = parser.parse_args()

    analyzer = PersonalColorAI()
    result = analyzer.analyze(args.image_path, visualize=args.visualize)

    print("\n--- 퍼스널컬러 진단 결과 ---")
    if "error" in result:
        print(f"오류: {result['error']}")
    else:
        print(f"입력 이미지 처리 후 크기: {result.get('image_shape_processed', 'N/A')}")
        print(f"\n추출된 주요 색상 (BGR):")
        skin_color = result['extracted_colors_bgr'].get('skin')
        eye_color = result['extracted_colors_bgr'].get('eye')
        hair_color = result['extracted_colors_bgr'].get('hair')

        if skin_color is not None:
            print(f"  - 피부: {skin_color}")
        else:
            print(f"  - 피부: 추출 실패")
        if eye_color is not None:
            print(f"  - 눈: {eye_color}")
        else:
            print(f"  - 눈: 추출 실패")
        if hair_color is not None:
            print(f"  - 머리카락: {hair_color}")
        else:
            print(f"  - 머리카락: 추출 실패")
        
        print("\n색상 속성 분석:")
        for key, value in result['color_attributes'].items():
            # 매우 긴 raw 데이터는 생략하거나 요약해서 출력할 수 있음
            if isinstance(value, np.ndarray) and value.size > 10:
                 print(f"  - {key}: Array data (size: {value.size})")
            else:
                print(f"  - {key}: {value}")
        
        print(f"\n최종 퍼스널컬러 유형 (예상): {result['personal_color_season']}")

if __name__ == "__main__":
    # VSCode에서 launch.json의 "module": "personalcolor_ai.main"으로 실행하는 것이 권장됩니다.
    # 터미널에서 직접 실행 시:
    # personalcolor_ai 폴더의 부모 디렉토리에서 python -m personalcolor_ai.main <이미지경로> 로 실행
    main()
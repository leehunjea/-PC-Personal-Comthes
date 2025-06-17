import cv2
import numpy as np

def preprocess_image(image_path, target_size=(512, 512)):
    """
    이미지를 로드하고 기본적인 전처리를 수행합니다.
    - 조명 보정 (예시: CLAHE)
    - 화이트 밸런스 조정 (예시: 그레이 월드)
    - 리사이즈
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    # 1. 조명 보정 (CLAHE 예시)
    image = correct_lighting(image)
    
    # 2. 화이트 밸런스 조정 (그레이 월드 예시)
    image = adjust_white_balance(image)
    
    # 3. 이미지 리사이즈
    image = cv2.resize(image, target_size)

    return image

def convert_to_lab(image_bgr):
    """BGR 이미지를 LAB 색상 공간으로 변환합니다."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)

def convert_to_hsv(image_bgr):
    """BGR 이미지를 HSV 색상 공간으로 변환합니다."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

def get_roi_pixels(image, points):
    """
    주어진 좌표(points)로 ROI를 만들고 해당 영역의 픽셀(BGR)을 반환합니다.
    points는 [[x1,y1], [x2,y2], ...] 형태의 numpy 배열 또는 리스트이어야 합니다.
    반환: (N, 3) 형태의 BGR 픽셀 numpy 배열
    """
    if points is None or len(points) == 0:
        return np.array([])

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # cv2.fillPoly는 점들의 리스트를 요소로 갖는 리스트를 기대합니다. (예: [np.array([[x,y],...])] )
    # points가 이미 numpy 배열이라면 np.array(points, dtype=np.int32)로 변환합니다.
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], (255))
    
    # 마스크된 영역의 픽셀들만 추출 (검은색 배경 제외)
    masked_pixels = image[mask == 255]
    return masked_pixels


def visualize_landmarks_and_roi(image, landmarks, cheek_points_left, cheek_points_right, eye_points_left, eye_points_right, hair_points):
    """얼굴 랜드마크와 추출된 ROI 영역을 시각화합니다 (테스트용)."""
    vis_image = image.copy()

    # 랜드마크 그리기
    if landmarks is not None:
        for i in range(landmarks.num_parts):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(vis_image, (x, y), 2, (0, 255, 0), -1)

    # ROI 영역 그리기
    if cheek_points_left is not None and len(cheek_points_left) > 0:
        cv2.polylines(vis_image, [np.array(cheek_points_left, dtype=np.int32)], True, (255, 0, 0), 1) # 파란색: 왼쪽 뺨
    if cheek_points_right is not None and len(cheek_points_right) > 0:
        cv2.polylines(vis_image, [np.array(cheek_points_right, dtype=np.int32)], True, (255, 0, 255), 1) # 마젠타: 오른쪽 뺨
    if eye_points_left is not None and len(eye_points_left) > 0:
        cv2.polylines(vis_image, [np.array(eye_points_left, dtype=np.int32)], True, (0, 255, 255), 1) # 청록색: 왼쪽 눈
    if eye_points_right is not None and len(eye_points_right) > 0:
        cv2.polylines(vis_image, [np.array(eye_points_right, dtype=np.int32)], True, (0, 255, 255), 1) # 청록색: 오른쪽 눈
    if hair_points is not None and len(hair_points) > 0:
         cv2.polylines(vis_image, [np.array(hair_points, dtype=np.int32)], True, (0, 0, 255), 1) # 빨간색: 머리카락 (상단)


    cv2.imshow("Landmarks and ROI", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def correct_lighting(image_bgr):
    """
    이미지의 조명을 보정합니다. (예시: LAB 색공간 L채널에 CLAHE 적용)
    더 정교한 알고리즘은 연구 및 테스트가 필요합니다.
    """
    # print("조명 보정 시도 (CLAHE)")
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    corrected_lab = cv2.merge((cl, a_channel, b_channel))
    corrected_bgr = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR)
    return corrected_bgr

def adjust_white_balance(image_bgr):
    """
    이미지의 화이트 밸런스를 조정합니다. (예시: 그레이 월드 알고리즘)
    더 정교한 알고리즘은 연구 및 테스트가 필요합니다.
    """
    # print("화이트 밸런스 조정 시도 (그레이 월드)")
    result = image_bgr.copy()
    b, g, r = cv2.split(result)
    
    # 0으로 나누는 것을 방지하기 위해 작은 값(epsilon) 추가
    epsilon = 1e-5 
    avg_b = np.mean(b) + epsilon
    avg_g = np.mean(g) + epsilon
    avg_r = np.mean(r) + epsilon
    
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    
    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r
    
    # 스케일링 값이 너무 크거나 작아지는 것을 방지 (예: 0.5 ~ 2.0 범위로 제한)
    # scale_b = np.clip(scale_b, 0.5, 2.0)
    # scale_g = np.clip(scale_g, 0.5, 2.0)
    # scale_r = np.clip(scale_r, 0.5, 2.0)

    b_corrected = np.clip(b * scale_b, 0, 255).astype(np.uint8)
    g_corrected = np.clip(g * scale_g, 0, 255).astype(np.uint8)
    r_corrected = np.clip(r * scale_r, 0, 255).astype(np.uint8)
    
    return cv2.merge((b_corrected, g_corrected, r_corrected))
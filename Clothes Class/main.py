import os
import glob
import shutil
import cv2
from ultralytics import YOLO

def safe_imread(img_path):
    """안전한 이미지 읽기 함수"""
    if not os.path.exists(img_path):
        return None
    
    try:
        # 원본 cv2.imread 직접 사용 (ultralytics 패치 우회)
        import cv2 as cv2_original
        img = cv2_original.imread(img_path)
        return img
    except:
        return None

def is_valid_clothing_image(results, img_shape):
    """의류 이미지가 유효한지 검사하는 함수"""
    height, width = img_shape[:2]
    person_boxes = []
    clothing_boxes = []
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            if cls == 0:  # person 클래스
                person_boxes.append(box)
            elif cls in [27, 28, 29, 30, 31, 32, 33]:  # 의류 관련 클래스 (예: 가방, 넥타이, 가방, 하의, 상의 등)
                clothing_boxes.append(box)
    
    # 1. 사람이 두 명 이상인 경우
    if len(person_boxes) > 1:
        return False
    
    # 2. 의류가 여러 개인 경우
    if len(clothing_boxes) > 1:
        return False
    
    # 3. 의류의 형태가 불완전한 경우 체크
    for box in clothing_boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        box_width = x2 - x1
        box_height = y2 - y1
        
        # 이미지 크기에 비해 너무 작은 경우
        if box_width < width * 0.1 or box_height < height * 0.1:
            return False
        
        # 너무 비정상적인 비율인 경우
        aspect_ratio = box_width / box_height
        if aspect_ratio < 0.2 or aspect_ratio > 5:
            return False
    
    return True

# 모델 로드
model = YOLO('C:/Users/AI-LHJ/Desktop/ClothesCC/best.pt')  # 기본 YOLOv8 모델 사용

root = 'C:/Users/AI-LHJ/Desktop/images'
style_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

for style in style_dirs:
    style_path = os.path.join(root, style)
    sub_dirs = [d for d in os.listdir(style_path) if os.path.isdir(os.path.join(style_path, d))]
    
    for sub in sub_dirs:
        sub_path = os.path.join(style_path, sub)
        invalid_dir = os.path.join(style_path, '학습불가_이미지', sub)
        os.makedirs(invalid_dir, exist_ok=True)
        
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            for img_path in glob.glob(os.path.join(sub_path, ext)):
                # 안전한 이미지 읽기
                img = safe_imread(img_path)
                if img is None:
                    print(f"이미지 읽기 실패: {img_path}")
                    # 읽기 실패한 파일도 학습불가로 이동
                    out_path = os.path.join(invalid_dir, os.path.basename(img_path))
                    shutil.move(img_path, out_path)
                    continue
                
                # YOLO 검출
                results = model(img)
                
                # 의류 이미지 유효성 검사
                if not is_valid_clothing_image(results, img.shape):
                    # 유효하지 않은 이미지는 학습불가 폴더로 이동
                    out_path = os.path.join(invalid_dir, os.path.basename(img_path))
                    shutil.move(img_path, out_path)
                    print(f"이동: {img_path} -> {out_path}")

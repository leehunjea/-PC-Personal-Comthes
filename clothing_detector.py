import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

class ClothingDetector:
    def __init__(self):
        # YOLOv8 모델 로드
        self.model = YOLO('yolov8n.pt')
        
        # 의류 관련 클래스 ID
        self.clothing_classes = {
            0: 'person',
            15: 't-shirt',
            16: 'shirt',
            17: 'sweater',
            18: 'jacket',
            19: 'vest',
            20: 'dress',
            21: 'blouse',
            22: 'tank top',
            23: 'polo shirt',
            24: 'hoodie'
        }

    def detect_clothing(self, image_path):
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"이미지를 로드할 수 없습니다: {image_path}")
            return None, None

        # YOLO로 객체 감지
        results = self.model(image)
        
        # 결과 시각화
        annotated_image = results[0].plot()
        
        # 감지된 객체 정보 추출
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in self.clothing_classes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        'class': self.clothing_classes[cls],
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2)
                    })
        
        return annotated_image, detections

    def visualize_detection(self, image_path, output_dir):
        # 결과 저장 디렉토리 생성
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 이미지 처리
        annotated_image, detections = self.detect_clothing(image_path)
        if annotated_image is None:
            return
        
        # 결과 시각화
        plt.figure(figsize=(15, 10))
        
        # 원본 이미지
        plt.subplot(1, 2, 1)
        original_image = cv2.imread(str(image_path))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        plt.imshow(original_image)
        plt.title('원본 이미지')
        plt.axis('off')
        
        # 감지 결과
        plt.subplot(1, 2, 2)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        plt.imshow(annotated_image)
        plt.title('의류 감지 결과')
        plt.axis('off')
        
        # 결과 저장
        output_path = output_dir / f"{Path(image_path).stem}_detection.png"
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        # 감지된 의류 정보 출력
        print(f"\n감지된 의류:")
        for det in detections:
            print(f"- {det['class']}: {det['confidence']:.2f}")
        
        return output_path

def process_style_directories(base_path):
    detector = ClothingDetector()
    base_path = Path(base_path)
    
    # 결과를 저장할 디렉토리 생성
    output_dir = base_path.parent / 'detection_results'
    output_dir.mkdir(exist_ok=True)
    
    # 스타일 폴더 순회
    for style_dir in base_path.iterdir():
        if not style_dir.is_dir() or style_dir.name.startswith('.'):
            continue
            
        print(f"\n처리 중인 스타일: {style_dir.name}")
        
        # 001_reg (상의) 폴더 확인
        top_dir = style_dir / '001_reg'
        if not top_dir.exists():
            print(f"경고: {style_dir.name}에 001_reg 폴더가 없습니다.")
            continue
        
        # 스타일별 결과 디렉토리 생성
        style_output_dir = output_dir / style_dir.name
        style_output_dir.mkdir(exist_ok=True)
        
        # 이미지 처리 (처음 5개만 테스트)
        for i, image_path in enumerate(top_dir.glob('*.*')):
            if i >= 5:  # 처음 5개 이미지만 테스트
                break
                
            if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            print(f"\n이미지 처리 중: {image_path}")
            detector.visualize_detection(image_path, style_output_dir)

if __name__ == "__main__":
    base_path = "images"
    process_style_directories(base_path) 
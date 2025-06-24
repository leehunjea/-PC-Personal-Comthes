from pathlib import Path
from PIL import Image
import os

def check_images(base_path):
    base_path = Path(base_path)
    total_images = 0
    valid_images = 0
    invalid_images = []
    
    print("\n=== 이미지 검사 시작 ===\n")
    
    # 스타일 폴더 순회
    for style_dir in base_path.iterdir():
        if not style_dir.is_dir() or style_dir.name.startswith('.'):
            continue
            
        print(f"\n스타일 '{style_dir.name}' 검사 중...")
        style_total = 0
        style_valid = 0
        
        # 001_reg (상의) 폴더 확인
        top_dir = style_dir / '001_reg'
        if not top_dir.exists():
            print(f"경고: {style_dir.name}에 001_reg 폴더가 없습니다.")
            continue
            
        # 이미지 파일 검사
        for image_path in top_dir.glob('*.*'):
            if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                style_total += 1
                total_images += 1
                
                try:
                    # 이미지 열기 시도
                    with Image.open(image_path) as img:
                        # 이미지 크기 확인
                        width, height = img.size
                        if width * height > 100000000:  # 1억 픽셀 제한
                            print(f"경고: 이미지가 너무 큼 ({image_path})")
                            invalid_images.append(str(image_path))
                            continue
                            
                        # 이미지 모드 확인
                        if img.mode not in ['RGB', 'L']:
                            print(f"경고: 지원하지 않는 이미지 모드 ({image_path})")
                            invalid_images.append(str(image_path))
                            continue
                            
                        # 이미지 데이터 로드 시도
                        img.load()
                        style_valid += 1
                        valid_images += 1
                        
                except Exception as e:
                    print(f"오류: {image_path} - {str(e)}")
                    invalid_images.append(str(image_path))
        
        print(f"총 이미지: {style_total}")
        print(f"유효한 이미지: {style_valid}")
        print(f"무효한 이미지: {style_total - style_valid}")
    
    print("\n=== 검사 결과 요약 ===")
    print(f"전체 이미지 수: {total_images}")
    print(f"유효한 이미지 수: {valid_images}")
    print(f"무효한 이미지 수: {total_images - valid_images}")
    
    if invalid_images:
        print("\n무효한 이미지 목록:")
        for img in invalid_images:
            print(f"- {img}")

if __name__ == "__main__":
    base_path = "images"
    check_images(base_path) 
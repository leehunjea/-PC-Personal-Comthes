# personalcolor_ai/main.py

import argparse
from personalcolor_ai.core import PersonalColorAI

def main():
    parser = argparse.ArgumentParser(description="AI 기반 퍼스널컬러 진단 시스템")
    parser.add_argument("image_path", type=str, help="분석할 이미지 파일 경로")
    parser.add_argument("--visualize", "-v", action="store_true", help="ROI 시각화 여부")
    args = parser.parse_args()

    analyzer = PersonalColorAI()
    result = analyzer.analyze(args.image_path, visualize=args.visualize)

    print("\n--- 퍼스널컬러 진단 결과 ---")
    if "error" in result:
        print(f"오류: {result['error']}")
        return

    print(f"\n입력 이미지 크기: {result['image_shape_processed']}")
    print(f"\n[주요 색상 (BGR)]")
    for part in ['skin', 'eye', 'hair']:
        color = result['extracted_colors_bgr'].get(part)
        print(f"  - {part}: {color if color is not None else '추출 실패'}")

    print("\n[색상 속성]")
    for key, value in result['color_attributes'].items():
        print(f"  - {key}: {value}")

    print(f"\n[예측된 퍼스널컬러 시즌]: {result['personal_color_season']}")

if __name__ == "__main__":
    main()

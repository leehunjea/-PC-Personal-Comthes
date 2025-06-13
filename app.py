from flask import Flask, render_template, request
from face_color_extract import process_image, get_personal_color_16_hard, analyze_features
import cv2
import numpy as np
from base64 import b64encode

app = Flask(__name__)

season_description = {
                "봄클리어": "봄클리어는 맑고 밝은 웜톤으로, 따뜻하고 생기 있는 컬러가 잘 어울립니다. 코랄, 연두, 밝은 베이지, 골드가 추천됩니다.",
                "봄라이트": "봄라이트는 밝고 부드러운 웜톤으로, 연한 살구, 민트, 연노랑 등 부드러운 파스텔톤이 잘 어울립니다.",
                "봄트루": "봄트루는 따뜻하고 화사한 컬러가 잘 어울리는 전형적인 봄 웜톤입니다. 피치, 라이트 옐로우, 코랄 오렌지가 추천 컬러입니다.",
                "봄소프트": "봄소프트는 부드러운 웜톤으로, 톤다운된 연베이지, 살구, 모카 브라운 등이 잘 어울립니다.",
                "여름라이트": "여름라이트는 맑고 밝은 쿨톤입니다. 라이트핑크, 베이비블루, 라일락, 연보라 등 파스텔톤이 잘 어울립니다.",
                "여름트루": "여름트루는 투명하고 청순한 쿨톤으로, 로즈핑크, 푸시아, 청록, 플럼 등 차분한 색감이 잘 어울립니다.",
                "여름클리어": "여름클리어는 맑고 선명한 쿨톤입니다. 크리스탈핑크, 시원한 블루, 밝은 라벤더 계열을 추천합니다.",
                "여름소프트": "여름소프트는 톤다운된 쿨톤으로, 회보라, 스카이블루, 차분한 분홍, 그레이시한 컬러가 어울립니다.",
                "가을딥": "가을딥은 깊고 선명한 웜톤입니다. 카키, 머스타드, 벽돌, 다크오렌지 계열이 추천 컬러입니다.",
                "가을트루": "가을트루는 전형적인 웜톤으로, 카멜, 브라운, 올리브, 코랄, 녹턴 계열이 잘 어울립니다.",
                "가을라이트": "가을라이트는 밝고 부드러운 웜톤으로, 살구, 라이트브라운, 소프트카키 계열이 추천 컬러입니다.",
                "가을소프트": "가을소프트는 톤다운된 웜톤으로, 모카, 카키그레이, 딥살몬 계열이 잘 어울립니다.",
                "겨울딥": "겨울딥은 어둡고 선명한 쿨톤입니다. 블랙, 와인, 딥네이비, 코발트블루가 잘 어울립니다.",
                "겨울트루": "겨울트루는 강렬한 대비의 쿨톤으로, 퓨어화이트, 블랙, 비비드핑크, 코발트블루가 추천 컬러입니다.",
                "겨울클리어": "겨울클리어는 맑고 쨍한 쿨톤입니다. 선명한 블루, 푸시아, 강렬한 아이시핑크가 어울립니다.",
                "겨울소프트": "겨울소프트는 톤다운된 쿨톤으로, 미디엄그레이, 쿨브라운, 푸른 회색이 잘 어울립니다."
                }

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    img_base64 = None
    error_message = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            npimg = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            features = process_image(image)
            # 인식 실패 조건
            required_keys = ['skin_h_mean', 'skin_s_mean', 'skin_v_mean', 'skin_r_mean', 'skin_b_mean']
            if (not features) or (any(features.get(k) is None for k in required_keys)):
                error_message = "얼굴이 인식되지 않았거나, 피부 컬러 추출에 실패했습니다.<br>다시 촬영해 주세요."
            else:
                import pandas as pd
                feature_df = pd.DataFrame([features])
                analyzed = analyze_features(feature_df.iloc[0])
                season = get_personal_color_16_hard(pd.Series({**features, **analyzed}))
                result = {
                    'season': season,
                    'skin_h': features['skin_h_mean'],
                    'skin_s': features['skin_s_mean'],
                    'skin_v': features['skin_v_mean'],
                    'skin_r': features['skin_r_mean'],
                    'skin_b': features['skin_b_mean']
                }
                _, img_encoded = cv2.imencode('.png', image)
                img_base64 = b64encode(img_encoded).decode()
    return render_template('index.html', result=result, img_base64=img_base64, season_description=season_description, error_message=error_message)


if __name__ == '__main__':
    app.run(debug=True)
import cv2, dlib, numpy as np
import torch
from PIL import Image
from torchvision import transforms

import os
import cv2
import dlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image

# ========= [BiSeNet 관련] ==========
import sys
BISENET_PATH = "C:/Users/AI-LHJ/Desktop/Task/face-parsing.PyTorch"  # 본인 설치 경로로 수정!
sys.path.append(BISENET_PATH)
from model import BiSeNet

BISENET_WEIGHTS = os.path.join(BISENET_PATH, "79999_iter.pth")

# BiSeNet 모델 로딩 (최초 1회만)
bisenet_model = None
def load_bisenet_model():
    global bisenet_model
    if bisenet_model is None:
        n_classes = 19
        bisenet_model = BiSeNet(n_classes=n_classes)
        bisenet_model.cuda()
        bisenet_model.load_state_dict(torch.load(BISENET_WEIGHTS))
        bisenet_model.eval()
    return bisenet_model

def get_hair_mask_bisenet(image_bgr):
    model = load_bisenet_model()
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    to_tensor = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = to_tensor(pil_img).unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(img_tensor)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    # 머리카락: class 17
    hair_mask = (parsing == 17).astype(np.uint8)
    hair_mask = cv2.resize(hair_mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return hair_mask

def extract_hair_color_by_segmentation(image):
    hair_mask = get_hair_mask_bisenet(image)
    if hair_mask.sum() == 0:
        return [0,0,0], [0,0,0]
    pixels = image[hair_mask == 1]
    if len(pixels) == 0:
        return [0,0,0], [0,0,0]
    mean_rgb = np.mean(pixels, axis=0)
    mean_hsv = np.mean(cv2.cvtColor(pixels.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1,3), axis=0)
    return mean_rgb, mean_hsv

# ========= [dlib 관련] ==========
predictor_path = "C:/Users/AI-LHJ/Desktop/Task/shape_predictor_68_face_landmarks.dat"  # 본인 경로로 수정
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

FACIAL_LANDMARKS_IDXS = {
    "jaw": (0, 17),
    "right_eyebrow": (17, 22),
    "left_eyebrow": (22, 27),
    "nose": (27, 36),
    "right_eye": (36, 42),
    "left_eye": (42, 48),
    "mouth_outer": (48, 60),
    "mouth_inner": (60, 68),
}
REGIONS = {
    'skin': list(range(0, 17)) + list(range(17, 22)) + list(range(22, 27)),  # jaw + eyebrow
    'lips': list(range(48, 60)),
    'right_eye': list(range(36, 42)),
    'left_eye': list(range(42, 48)),
    'nose': list(range(27, 36)),
}

def extract_mean_color(image, landmarks, idxs):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array([landmarks[i] for i in idxs if i < len(landmarks)])
    if len(points) > 2:
        cv2.fillPoly(mask, [points], 255)
        region = cv2.bitwise_and(image, image, mask=mask)
        pixels = region[mask == 255]
        if len(pixels) > 0:
            mean_rgb = np.mean(pixels, axis=0)
            mean_hsv = np.mean(cv2.cvtColor(pixels.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1,3), axis=0)
            return mean_rgb, mean_hsv
    return [0,0,0], [0,0,0]

def get_landmarks(image):
    dets = detector(image, 1)
    if len(dets) == 0:
        return None
    shape = predictor(image, dets[0])
    landmarks = np.array([[pt.x, pt.y] for pt in shape.parts()])
    return landmarks

def process_image(image):
    if image is None:
        return None
    landmarks = get_landmarks(image)
    if landmarks is None:
        return None

    features = {}

    # mean 컬럼명에 맞게 수정
    mean_skin_rgb, mean_skin_hsv = extract_mean_color(image, landmarks, REGIONS['skin'])
    features.update({
        "skin_r_mean": mean_skin_rgb[2], "skin_g_mean": mean_skin_rgb[1], "skin_b_mean": mean_skin_rgb[0],
        "skin_h_mean": mean_skin_hsv[0], "skin_s_mean": mean_skin_hsv[1], "skin_v_mean": mean_skin_hsv[2],
    })

    mean_lips_rgb, mean_lips_hsv = extract_mean_color(image, landmarks, REGIONS['lips'])
    features.update({
        "lips_r_mean": mean_lips_rgb[2], "lips_g_mean": mean_lips_rgb[1], "lips_b_mean": mean_lips_rgb[0],
        "lips_h_mean": mean_lips_hsv[0], "lips_s_mean": mean_lips_hsv[1], "lips_v_mean": mean_lips_hsv[2],
    })

    mean_reye_rgb, mean_reye_hsv = extract_mean_color(image, landmarks, REGIONS['right_eye'])
    mean_leye_rgb, mean_leye_hsv = extract_mean_color(image, landmarks, REGIONS['left_eye'])
    features.update({
        "reye_r_mean": mean_reye_rgb[2], "reye_g_mean": mean_reye_rgb[1], "reye_b_mean": mean_reye_rgb[0],
        "reye_h_mean": mean_reye_hsv[0], "reye_s_mean": mean_reye_hsv[1], "reye_v_mean": mean_reye_hsv[2],
        "leye_r_mean": mean_leye_rgb[2], "leye_g_mean": mean_leye_rgb[1], "leye_b_mean": mean_leye_rgb[0],
        "leye_h_mean": mean_leye_hsv[0], "leye_s_mean": mean_leye_hsv[1], "leye_v_mean": mean_leye_hsv[2],
    })

    mean_nose_rgb, mean_nose_hsv = extract_mean_color(image, landmarks, REGIONS['nose'])
    features.update({
        "nose_r_mean": mean_nose_rgb[2], "nose_g_mean": mean_nose_rgb[1], "nose_b_mean": mean_nose_rgb[0],
        "nose_h_mean": mean_nose_hsv[0], "nose_s_mean": mean_nose_hsv[1], "nose_v_mean": mean_nose_hsv[2],
    })

    mean_hair_rgb, mean_hair_hsv = extract_hair_color_by_segmentation(image)
    features.update({
        "hair_r_mean": mean_hair_rgb[2], "hair_g_mean": mean_hair_rgb[1], "hair_b_mean": mean_hair_rgb[0],
        "hair_h_mean": mean_hair_hsv[0], "hair_s_mean": mean_hair_hsv[1], "hair_v_mean": mean_hair_hsv[2],
    })

    return features

# (1) 부위별 쿨/웜, 밝기, 채도 판정 함수
def get_tone_info(r, b, v, s):
    temp = r - b
    # 웜/쿨
    if temp >= 40:
        tone = '웜'
    elif temp <= 20:
        tone = '쿨'
    else:
        tone = '중성'
    # 밝기
    if v > 185:
        lightness = '밝음'
    elif v < 140:
        lightness = '어두움'
    else:
        lightness = '중간'
    # 채도
    if s > 120:
        saturation = '선명'
    elif s < 70:
        saturation = '저채도'
    else:
        saturation = '중간'
    return tone, lightness, saturation

# (2) 부위별 분석 정보
def analyze_features(row):
    skin_tone, skin_light, skin_sat = get_tone_info(row['skin_r_mean'], row['skin_b_mean'], row['skin_v_mean'], row['skin_s_mean'])
    hair_tone, hair_light, hair_sat = get_tone_info(row['hair_r_mean'], row['hair_b_mean'], row['hair_v_mean'], row['hair_s_mean'])
    lips_tone, lips_light, lips_sat = get_tone_info(row['lips_r_mean'], row['lips_b_mean'], row['lips_v_mean'], row['lips_s_mean'])
    eye_r = (row['reye_r_mean'] + row['leye_r_mean']) / 2
    eye_b = (row['reye_b_mean'] + row['leye_b_mean']) / 2
    eye_v = (row['reye_v_mean'] + row['leye_v_mean']) / 2
    eye_s = (row['reye_s_mean'] + row['leye_s_mean']) / 2
    eye_tone, eye_light, eye_sat = get_tone_info(eye_r, eye_b, eye_v, eye_s)
    return {
        'skin_tone': skin_tone, 'skin_light': skin_light, 'skin_sat': skin_sat,
        'hair_tone': hair_tone, 'hair_light': hair_light, 'hair_sat': hair_sat,
        'lips_tone': lips_tone, 'lips_light': lips_light, 'lips_sat': lips_sat,
        'eye_tone': eye_tone, 'eye_light': eye_light, 'eye_sat': eye_sat
    }

# (3) 시즌별 대표 임계값 (가중 평균값, 실험 기반 추천)
SEASON_REF = {
    #      v      s     temp
    '봄클리어':   (200, 135,  90),
    '봄트루':     (190, 125,  70),
    '봄라이트':   (200, 110,  55),
    '봄소프트':   (185,  90,  45),
    '여름클리어': (200, 135,  10),
    '여름트루':   (195, 125,  25),
    '여름라이트': (200, 110,  25),
    '여름소프트': (185,  90,  15),
    '가을딥':     (150, 120,  85),
    '가을트루':   (160, 110,  70),
    '가을라이트': (175, 100,  60),
    '가을소프트': (155,  80,  55),
    '겨울딥':     (135, 120,  10),
    '겨울트루':   (140, 110,  20),
    '겨울라이트': (165,  95,  25),
    '겨울소프트': (145,  85,  10),
}


# (4) 시즌 판별 함수
def get_personal_color_16_hard(row):
    skin_v = row['skin_v_mean']; skin_s = row['skin_s_mean']; skin_temp = row['skin_r_mean'] - row['skin_b_mean']
    hair_v = row['hair_v_mean']; hair_s = row['hair_s_mean']; hair_temp = row['hair_r_mean'] - row['hair_b_mean']
    lips_v = row['lips_v_mean']; lips_s = row['lips_s_mean']; lips_temp = row['lips_r_mean'] - row['lips_b_mean']
    eye_v = (row['reye_v_mean'] + row['leye_v_mean']) / 2
    eye_s = (row['reye_s_mean'] + row['leye_s_mean']) / 2
    eye_temp = ((row['reye_r_mean'] + row['leye_r_mean'])/2) - ((row['reye_b_mean'] + row['leye_b_mean'])/2)

    # v, s, temp의 가중합 (가중치는 여기서 쉽게 조정)
    # v, s, temp의 가중합 (피부 영향력 강화)
    v = 0.85 * skin_v + 0.05 * hair_v + 0.05 * lips_v + 0.05 * eye_v
    s = 0.7  * skin_s + 0.1 * hair_s + 0.1 * lips_s + 0.1 * eye_s
    temp = 0.85 * skin_temp + 0.05 * hair_temp + 0.05 * lips_temp + 0.05 * eye_temp


    # 시즌별 조건(기존 hard 조건식)
    season_scores = {}
    # 봄
    season_scores['봄클리어']  = int(v > 185 and temp > 70 and s > 120)
    season_scores['봄트루']    = int(v > 185 and temp > 40 and s > 120)
    season_scores['봄라이트']  = int(v > 185 and temp > 40 and 90 < s <= 120)
    season_scores['봄소프트']  = int(v > 185 and temp > 40 and s <= 90)
    # 여름
    season_scores['여름클리어'] = int(v > 185 and temp <= 40 and s > 120 and temp <= 20)
    season_scores['여름트루']   = int(v > 185 and temp <= 40 and s > 120 and not (temp <= 20))
    season_scores['여름라이트'] = int(v > 185 and temp <= 40 and 90 < s <= 120)
    season_scores['여름소프트'] = int(v > 185 and temp <= 40 and s <= 90)
    # 가을
    season_scores['가을딥']    = int(v <= 185 and temp > 40 and v <= 150 and s > 120 and temp > 70)
    season_scores['가을트루']  = int(v <= 185 and temp > 40 and v <= 150 and s > 120)
    season_scores['가을라이트'] = int(v <= 185 and temp > 40 and s > 90)
    season_scores['가을소프트'] = int(v <= 185 and temp > 40 and s <= 90)
    # 겨울
    season_scores['겨울딥']    = int(v <= 185 and temp <= 40 and v <= 150 and s > 120 and temp <= 20)
    season_scores['겨울트루']  = int(v <= 185 and temp <= 40 and v <= 150 and s > 120)
    season_scores['겨울라이트'] = int(v <= 185 and temp <= 40 and s > 90)
    season_scores['겨울소프트'] = int(v <= 185 and temp <= 40 and s <= 90)

    # (a) 조건식 일치(=1)인 시즌 우선
    sorted_season = [k for k, v in sorted(season_scores.items(), key=lambda item: -item[1])]
    for season in sorted_season:
        if season_scores[season]:
            return season

    # (b) 모든 조건이 0이면, 대표 임계값과의 차이로 "가장 가까운 시즌"을 리턴
    min_diff = float('inf')
    best_season = None
    for season, (ref_v, ref_s, ref_temp) in SEASON_REF.items():
        diff = abs(v-ref_v) + abs(s-ref_s) + abs(temp-ref_temp)
        if diff < min_diff:
            min_diff = diff
            best_season = season
    return best_season
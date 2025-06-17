# 수정된 ColorAnalyzer 및 SeasonClassifier
import numpy as np
import cv2
import json
import os

from personalcolor_ai.utils import convert_to_lab, convert_to_hsv

class ColorAnalyzer:
    def __init__(self):
        self.wc_weights = {'skin': 0.7, 'hair': 0.2, 'eye': 0.1}

    def _calculate_warm_cool_value(self, lab_color):
        if lab_color is None:
            return 0
        _, _, b_val = lab_color
        return b_val - 128

    def _quantize_attribute(self, value, thresholds, labels):
        for i, t in enumerate(thresholds):
            if value <= t:
                return labels[i]
        return labels[-1]

    def analyze_color_attributes(self, feature_colors_bgr, skin_roi_pixels_lab=None):
        attributes = {}
        skin_bgr = feature_colors_bgr.get('skin')
        hair_bgr = feature_colors_bgr.get('hair')
        eye_bgr = feature_colors_bgr.get('eye')

        if skin_bgr is None:
            return {"error": "Skin color missing"}

        skin_lab = convert_to_lab(np.uint8([[skin_bgr]]))[0][0]
        skin_hsv = convert_to_hsv(np.uint8([[skin_bgr]]))[0][0]

        hair_lab = convert_to_lab(np.uint8([[hair_bgr]]))[0][0] if hair_bgr is not None else None
        eye_lab = convert_to_lab(np.uint8([[eye_bgr]]))[0][0] if eye_bgr is not None else None

        wc_skin_val = self._calculate_warm_cool_value(skin_lab)
        wc_hair_val = self._calculate_warm_cool_value(hair_lab) if hair_lab is not None else 0
        wc_eye_val = self._calculate_warm_cool_value(eye_lab) if eye_lab is not None else 0

        final_wc_score = wc_skin_val * self.wc_weights['skin'] + \
                           wc_hair_val * self.wc_weights['hair'] + \
                           wc_eye_val * self.wc_weights['eye']
        attributes['warm_cool_raw_score'] = final_wc_score

        wc_type = "W" if skin_lab[2] > 132 else "C"
        attributes['color_type'] = wc_type

        wc_level_abs = abs(skin_lab[2] - 128)
        wc_level_thresholds = [10, 20, 30, 40]
        wc_level_labels = [1, 2, 3, 4, 5]
        attributes['color_level'] = self._quantize_attribute(wc_level_abs, wc_level_thresholds, wc_level_labels)
        attributes['color'] = f"{attributes['color_level']}{wc_type}"

        lightness_val = skin_lab[0]
        attributes['lightness_raw'] = lightness_val
        lightness_thresholds = [90, 130, 160, 190, 220]
        lightness_labels = [1, 2, 3, 4, 5, 6]
        attributes['lightness_level'] = self._quantize_attribute(lightness_val, lightness_thresholds, lightness_labels)
        attributes['lightness'] = f"{attributes['lightness_level']}L"

        saturation_val = skin_hsv[1]
        attributes['saturation_raw'] = saturation_val
        saturation_thresholds = [30, 60, 90, 120]
        saturation_labels = [1, 2, 3, 4, 5]
        attributes['saturation_level'] = self._quantize_attribute(saturation_val, saturation_thresholds, saturation_labels)
        attributes['saturation'] = f"{attributes['saturation_level']}S"

        clarity_score = None
        if skin_roi_pixels_lab is not None and len(skin_roi_pixels_lab) > 0:
            l_std_dev = np.std(skin_roi_pixels_lab[:, 0])
            clarity_score = (100 - l_std_dev) * 0.5 + saturation_val * 0.5
            attributes['clarity_raw_l_std'] = l_std_dev
        else:
            clarity_score = saturation_val
        attributes['clarity_raw_score'] = clarity_score

        clarity_type = "C" if clarity_score > 200 else "T"
        attributes['clear_dull_type'] = clarity_type

        clarity_level_abs = abs(clarity_score - 200)
        clarity_level_thresholds = [40, 80, 120, 160]
        clarity_level_labels = [1, 2, 3, 4, 5]
        attributes['clear_dull_level'] = self._quantize_attribute(clarity_level_abs, clarity_level_thresholds, clarity_level_labels)
        attributes['clear_dull'] = f"{attributes['clear_dull_level']}{clarity_type}"

        return attributes


class SeasonClassifier:
    def __init__(self, matrix_filename="pccs_matrix.json"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.matrix_file_path = os.path.join(base_dir, "config", matrix_filename)
        self.matrix = self._load_season_matrix_from_file()

    def _load_season_matrix_from_file(self):
        try:
            with open(self.matrix_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"매트릭스 로드 오류: {e}")
            return {}

    def _check_condition(self, attr_dict, cond):
        if not cond:
            return True
        if "type" in cond and attr_dict.get("type") != cond["type"]:
            return False
        if "level_range" in cond:
            if not (cond["level_range"][0] <= attr_dict.get("level", -1) <= cond["level_range"][1]):
                return False
        if "level_or_value" in cond and attr_dict.get("level") != cond["level_or_value"]:
            return False
        return True

    def _get_attr_dict(self, attributes, key):
        if key == "color":
            return {"type": attributes.get("color_type"), "level": attributes.get("color_level")}
        if key == "lightness":
            return {"level": attributes.get("lightness_level")}
        if key == "clear_dull":
            return {"type": attributes.get("clear_dull_type"), "level": attributes.get("clear_dull_level")}
        if key == "saturation":
            return {"level": attributes.get("saturation_level")}
        return {}

    def classify(self, attributes):
        if 'error' in attributes:
            return "Error in color attributes"

        scored = []
        for season, conds in self.matrix.items():
            total = 0
            match = 0
            for key in conds:
                attr = self._get_attr_dict(attributes, key)
                if self._check_condition(attr, conds[key]):
                    match += 1
                total += 1
            scored.append((season, match / total if total > 0 else 0))

        best = max(scored, key=lambda x: x[1])
        return best[0] if best[1] >= 0.6 else "분류 미정 (조건 불충분)"

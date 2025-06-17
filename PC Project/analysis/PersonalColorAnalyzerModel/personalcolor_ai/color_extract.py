import numpy as np
from sklearn.cluster import KMeans
import cv2 # 주로 색상 변환에 사용

class ColorExtractor:
    def __init__(self, n_clusters=1):
        """
        n_clusters: K-Means에서 찾을 대표 색상 개수 (보통 1개로 주요 색상 추출)
        """
        self.n_clusters = n_clusters

    def extract_dominant_color_from_pixels(self, pixels_bgr):
        """
        주어진 픽셀 데이터(BGR)에서 K-Means를 사용하여 대표 색상을 추출합니다.
        pixels_bgr: N x 3 형태의 numpy 배열 (BGR 순서)
        반환: 대표 색상 BGR 값 (numpy array) 또는 None
        """
        if pixels_bgr is None or len(pixels_bgr) == 0:
            return None
        
        # 픽셀 수가 클러스터 수보다 적으면 첫 번째 픽셀을 반환하거나 에러 처리
        if len(pixels_bgr) < self.n_clusters:
            # print(f"Warning: Number of pixels ({len(pixels_bgr)}) is less than n_clusters ({self.n_clusters}). Using mean color.")
            if len(pixels_bgr) > 0:
                return np.mean(pixels_bgr, axis=0).astype(int)
            return None

        try:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=10)
            kmeans.fit(pixels_bgr)
            dominant_color = kmeans.cluster_centers_[0].astype(int)
            return dominant_color # BGR
        except Exception as e:
            print(f"Error during K-Means clustering: {e}")
            # K-Means 실패 시 평균 색상 반환 시도
            if len(pixels_bgr) > 0:
                return np.mean(pixels_bgr, axis=0).astype(int)
            return None


    def extract_feature_colors(self, image_bgr, roi_map):
        """
        이미지와 ROI 픽셀들로부터 각 부위의 대표 색상을 추출합니다.
        image_bgr: 전체 BGR 이미지
        roi_map: {'skin_pixels': skin_pixels_bgr, 'eye_pixels': eye_pixels_bgr, 'hair_pixels': hair_pixels_bgr} 형태의 딕셔너리
        반환: {'skin': dominant_skin_color_bgr, 'eye': dominant_eye_color_bgr, 'hair': dominant_hair_color_bgr}
        """
        feature_colors_bgr = {}

        skin_pixels_bgr = roi_map.get('skin_pixels')
        eye_pixels_bgr = roi_map.get('eye_pixels')
        hair_pixels_bgr = roi_map.get('hair_pixels')

        feature_colors_bgr['skin'] = self.extract_dominant_color_from_pixels(skin_pixels_bgr)
        feature_colors_bgr['eye'] = self.extract_dominant_color_from_pixels(eye_pixels_bgr)
        feature_colors_bgr['hair'] = self.extract_dominant_color_from_pixels(hair_pixels_bgr)
        
        return feature_colors_bgr
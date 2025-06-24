import os
import shutil
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tqdm import tqdm

class ImageClassifier:
    def __init__(self, input_folder, output_folder, n_clusters=10):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.n_clusters = n_clusters
        self.model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        self.image_paths = []
        self.features = []
        
    def load_images(self):
        """폴더에서 이미지 파일들을 로드"""
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        
        for filename in os.listdir(self.input_folder):
            if filename.lower().endswith(supported_formats):
                self.image_paths.append(os.path.join(self.input_folder, filename))
        
        print(f"총 {len(self.image_paths)}개의 이미지를 발견했습니다.")
        return len(self.image_paths)
    
    def extract_features(self):
        """VGG16 모델을 사용하여 이미지 특징 추출"""
        print("이미지 특징을 추출하는 중...")
        
        for img_path in tqdm(self.image_paths):
            try:
                # 이미지 로드 및 전처리
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                
                # 특징 추출
                features = self.model.predict(img_array, verbose=0)
                self.features.append(features.flatten())
                
            except Exception as e:
                print(f"이미지 처리 오류 {img_path}: {e}")
                # 오류 발생한 이미지는 제외
                self.image_paths.remove(img_path)
        
        self.features = np.array(self.features)
        print(f"특징 추출 완료: {self.features.shape}")
    
    def reduce_dimensions(self, n_components=50):
        """PCA를 사용하여 차원 축소"""
        print("차원 축소 중...")
        pca = PCA(n_components=n_components)
        self.features = pca.fit_transform(self.features)
        print(f"차원 축소 완료: {self.features.shape}")
    
    def cluster_images(self):
        """K-Means를 사용하여 이미지 클러스터링"""
        print(f"{self.n_clusters}개 그룹으로 클러스터링 중...")
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels = kmeans.fit_predict(self.features)
        
        print("클러스터링 완료!")
        return self.labels
    
    def create_output_folders(self):
        """출력 폴더 생성"""
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        
        os.makedirs(self.output_folder)
        
        for i in range(self.n_clusters):
            cluster_folder = os.path.join(self.output_folder, f"cluster_{i}")
            os.makedirs(cluster_folder)
    
    def save_clustered_images(self):
        """클러스터링 결과에 따라 이미지를 폴더별로 저장"""
        print("분류된 이미지를 폴더에 저장 중...")
        
        self.create_output_folders()
        
        for img_path, label in tqdm(zip(self.image_paths, self.labels)):
            filename = os.path.basename(img_path)
            dest_folder = os.path.join(self.output_folder, f"cluster_{label}")
            dest_path = os.path.join(dest_folder, filename)
            
            shutil.copy2(img_path, dest_path)
        
        print("이미지 저장 완료!")
    
    def show_cluster_summary(self):
        """클러스터링 결과 요약 출력"""
        print("\n=== 클러스터링 결과 요약 ===")
        for i in range(self.n_clusters):
            count = np.sum(self.labels == i)
            percentage = (count / len(self.labels)) * 100
            print(f"클러스터 {i}: {count}개 이미지 ({percentage:.1f}%)")
    
    def run_classification(self):
        """전체 분류 과정 실행"""
        # 1. 이미지 로드
        if self.load_images() == 0:
            print("이미지를 찾을 수 없습니다.")
            return
        
        # 2. 특징 추출
        self.extract_features()
        
        if len(self.features) == 0:
            print("특징 추출에 실패했습니다.")
            return
        
        # 3. 차원 축소
        self.reduce_dimensions()
        
        # 4. 클러스터링
        self.cluster_images()
        
        # 5. 결과 저장
        self.save_clustered_images()
        
        # 6. 결과 요약
        self.show_cluster_summary()

# 사용 예시
if __name__ == "__main__":
    # 설정
    INPUT_FOLDER = "C:/Users/AI-LHJ/Desktop/Task/images/로맨틱/로맨틱_001_reg_분류결과/그룹_16"      # 원본 이미지가 있는 폴더
    OUTPUT_FOLDER = "C:/Users/AI-LHJ/Desktop/Task/images/로맨틱/로맨틱_001_reg_분류결과/그룹_16/분루" # 분류된 이미지를 저장할 폴더
    N_CLUSTERS = 20                     # 분류할 그룹 수
    
    # 분류기 생성 및 실행
    classifier = ImageClassifier(INPUT_FOLDER, OUTPUT_FOLDER, N_CLUSTERS)
    classifier.run_classification()
    
    print("\n이미지 분류가 완료되었습니다!")
    print(f"결과는 '{OUTPUT_FOLDER}' 폴더에서 확인하세요.")

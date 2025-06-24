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
import gc

class MultiCategoryImageClassifier:
    def __init__(self, base_folder, n_clusters=10):
        self.base_folder = base_folder
        self.n_clusters = n_clusters
        self.model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        
    def get_category_folders(self):
        """카테고리 폴더 목록 가져오기"""
        category_folders = []
        
        for item in os.listdir(self.base_folder):
            item_path = os.path.join(self.base_folder, item)
            if os.path.isdir(item_path):
                # 하위 폴더들 확인
                subfolders = [f for f in os.listdir(item_path) 
                             if os.path.isdir(os.path.join(item_path, f))]
                
                if subfolders:  # 하위 폴더가 있는 경우
                    for subfolder in subfolders:
                        subfolder_path = os.path.join(item_path, subfolder)
                        category_folders.append({
                            'name': f"{item}_{subfolder}",
                            'path': subfolder_path,
                            'parent': item_path
                        })
                else:  # 하위 폴더가 없는 경우 (직접 이미지가 있는 폴더)
                    category_folders.append({
                        'name': item,
                        'path': item_path,
                        'parent': self.base_folder
                    })
        
        return category_folders
    
    def load_images_from_folder(self, folder_path):
        """특정 폴더에서 이미지 파일들을 로드"""
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        image_paths = []
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(supported_formats):
                image_paths.append(os.path.join(folder_path, filename))
        
        return image_paths
    
    def extract_features_batch(self, image_paths, batch_size=16):
        """배치 단위로 특징 추출 (메모리 효율성 향상)"""
        features = []
        valid_paths = []
        
        print(f"총 {len(image_paths)}개의 이미지에서 특징을 추출하는 중...")
        
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i+batch_size]
            batch_features = []
            batch_valid_paths = []
            
            for img_path in batch_paths:
                try:
                    # 이미지 로드 및 전처리
                    img = image.load_img(img_path, target_size=(224, 224))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)
                    
                    # 특징 추출
                    feature = self.model.predict(img_array, verbose=0)
                    batch_features.append(feature.flatten())
                    batch_valid_paths.append(img_path)
                    
                except Exception as e:
                    print(f"이미지 처리 오류 {img_path}: {e}")
                    continue
            
            features.extend(batch_features)
            valid_paths.extend(batch_valid_paths)
            
            # 메모리 정리
            gc.collect()
        
        return np.array(features), valid_paths
    
    def classify_category(self, category_info):
        """단일 카테고리 분류"""
        print(f"\n=== {category_info['name']} 카테고리 분류 시작 ===")
        
        # 이미지 로드
        image_paths = self.load_images_from_folder(category_info['path'])
        
        if len(image_paths) == 0:
            print(f"{category_info['name']}: 이미지를 찾을 수 없습니다.")
            return
        
        print(f"{category_info['name']}: {len(image_paths)}개 이미지 발견")
        
        # 특징 추출
        features, valid_paths = self.extract_features_batch(image_paths)
        
        if len(features) == 0:
            print(f"{category_info['name']}: 특징 추출에 실패했습니다.")
            return
        
        # 클러스터 수 조정 (이미지 수가 적으면 클러스터 수 줄이기)
        actual_clusters = min(self.n_clusters, len(features))
        
        # 차원 축소
        print(f"{category_info['name']}: 차원 축소 중...")
        n_components = min(50, len(features) - 1)
        pca = PCA(n_components=n_components)
        features_reduced = pca.fit_transform(features)
        
        # 클러스터링
        print(f"{category_info['name']}: {actual_clusters}개 그룹으로 클러스터링 중...")
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_reduced)
        
        # 결과 저장
        self.save_classified_images(category_info, valid_paths, labels, actual_clusters)
        
        # 결과 요약
        self.show_category_summary(category_info['name'], labels, actual_clusters)
    
    def save_classified_images(self, category_info, image_paths, labels, n_clusters):
        """분류된 이미지를 원본 폴더 내에 저장"""
        print(f"{category_info['name']}: 분류된 이미지를 저장 중...")
        
        # 출력 폴더 생성 (원본 폴더 내에)
        output_base = os.path.join(category_info['parent'], f"{category_info['name']}_분류결과")
        
        if os.path.exists(output_base):
            shutil.rmtree(output_base)
        
        os.makedirs(output_base)
        
        # 클러스터별 폴더 생성
        for i in range(n_clusters):
            cluster_folder = os.path.join(output_base, f"그룹_{i+1}")
            os.makedirs(cluster_folder)
        
        # 이미지 복사
        for img_path, label in tqdm(zip(image_paths, labels), desc="이미지 저장"):
            filename = os.path.basename(img_path)
            dest_folder = os.path.join(output_base, f"그룹_{label+1}")
            dest_path = os.path.join(dest_folder, filename)
            
            shutil.copy2(img_path, dest_path)
        
        print(f"{category_info['name']}: 저장 완료 -> {output_base}")
    
    def show_category_summary(self, category_name, labels, n_clusters):
        """카테고리별 클러스터링 결과 요약"""
        print(f"\n=== {category_name} 분류 결과 요약 ===")
        for i in range(n_clusters):
            count = np.sum(labels == i)
            percentage = (count / len(labels)) * 100
            print(f"그룹 {i+1}: {count}개 이미지 ({percentage:.1f}%)")
    
    def run_all_classifications(self):
        """모든 카테고리 분류 실행"""
        print("카테고리 폴더를 스캔하는 중...")
        categories = self.get_category_folders()
        
        if not categories:
            print("분류할 카테고리를 찾을 수 없습니다.")
            return
        
        print(f"총 {len(categories)}개의 카테고리를 발견했습니다:")
        for cat in categories:
            print(f"- {cat['name']}: {cat['path']}")
        
        # 각 카테고리별로 분류 실행
        for category in categories:
            try:
                self.classify_category(category)
            except Exception as e:
                print(f"{category['name']} 분류 중 오류 발생: {e}")
                continue
        
        print("\n=== 전체 분류 작업 완료 ===")

# 사용 예시
if __name__ == "__main__":
    # 설정
    BASE_FOLDER = "C:/Users/AI-LHJ/Desktop/task/images"  # 메인 이미지 폴더
    N_CLUSTERS = 20  # 각 카테고리별 분류할 그룹 수
    
    # 분류기 생성 및 실행
    classifier = MultiCategoryImageClassifier(BASE_FOLDER, N_CLUSTERS)
    classifier.run_all_classifications()
    
    print("\n모든 이미지 분류가 완료되었습니다!")
    print("각 카테고리 폴더 내에서 '_분류결과' 폴더를 확인하세요.")

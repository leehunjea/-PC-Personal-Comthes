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
import tensorflow as tf
from tensorflow.keras import backend as K
import psutil
import logging
import json
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiCategoryImageClassifier:
    def __init__(self, base_folder, n_clusters=10, batch_size=8, skip_completed=True):
        self.base_folder = base_folder
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.skip_completed = skip_completed
        self.model = None
        self.progress_file = os.path.join(base_folder, "classification_progress.json")
        self.completed_categories = self._load_progress()
        self._initialize_model()
        
    def _load_progress(self):
        """이전 작업 진행상황 로드"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    completed = set(progress_data.get('completed_categories', []))
                    logger.info(f"이전 진행상황 로드: {len(completed)}개 카테고리 완료됨")
                    return completed
            except Exception as e:
                logger.warning(f"진행상황 파일 로드 실패: {e}")
        return set()
    
    def _save_progress(self, category_name):
        """작업 진행상황 저장"""
        self.completed_categories.add(category_name)
        progress_data = {
            'completed_categories': list(self.completed_categories),
            'last_updated': datetime.now().isoformat(),
            'total_completed': len(self.completed_categories)
        }
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
            logger.info(f"진행상황 저장: {category_name} 완료")
        except Exception as e:
            logger.warning(f"진행상황 저장 실패: {e}")
    
    def _is_category_completed(self, category_info):
        """카테고리가 이미 분류되었는지 확인"""
        # 1. 진행상황 파일에서 확인
        if self.skip_completed and category_info['name'] in self.completed_categories:
            return True, "진행상황 파일에 기록됨"
        
        # 2. 분류결과 폴더 존재 여부 확인
        result_folder = os.path.join(category_info['parent'], f"{category_info['name']}_분류결과")
        if os.path.exists(result_folder):
            # 분류결과 폴더 내에 그룹 폴더들이 있는지 확인
            group_folders = [f for f in os.listdir(result_folder) 
                           if f.startswith('그룹_') and os.path.isdir(os.path.join(result_folder, f))]
            
            if group_folders:
                # 그룹 폴더들에 이미지가 있는지 확인
                total_images = 0
                for group_folder in group_folders:
                    group_path = os.path.join(result_folder, group_folder)
                    images_in_group = len([f for f in os.listdir(group_path) 
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))])
                    total_images += images_in_group
                
                if total_images > 0:
                    # 진행상황에도 추가
                    self.completed_categories.add(category_info['name'])
                    return True, f"분류결과 폴더 존재 ({len(group_folders)}개 그룹, {total_images}개 이미지)"
        
        return False, "미완료"
        
    def _initialize_model(self):
        """모델 초기화 및 메모리 최적화 설정"""
        # GPU 메모리 증가 허용 설정
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                logger.warning(f"GPU 메모리 설정 실패: {e}")
        
        # 모델 로드 (한 번만)
        self.model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        logger.info("VGG16 모델 로드 완료")
        
    def _clear_memory(self):
        """메모리 정리 함수"""
        K.clear_session()
        gc.collect()
        
        if hasattr(tf.config.experimental, 'reset_memory_stats'):
            try:
                tf.config.experimental.reset_memory_stats('GPU:0')
            except:
                pass
    
    def _log_memory_usage(self, step=""):
        """메모리 사용량 로깅"""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"{step} - 메모리 사용량: {memory_mb:.1f} MB")
        
    def get_category_folders(self):
        """카테고리 폴더 목록 가져오기"""
        category_folders = []
        
        for item in os.listdir(self.base_folder):
            item_path = os.path.join(self.base_folder, item)
            if os.path.isdir(item_path):
                subfolders = [f for f in os.listdir(item_path) 
                             if os.path.isdir(os.path.join(item_path, f))]
                
                if subfolders:
                    for subfolder in subfolders:
                        subfolder_path = os.path.join(item_path, subfolder)
                        category_folders.append({
                            'name': f"{item}_{subfolder}",
                            'path': subfolder_path,
                            'parent': item_path
                        })
                else:
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
    
    def extract_features_batch(self, image_paths):
        """메모리 최적화된 배치 특징 추출"""
        features = []
        valid_paths = []
        
        logger.info(f"총 {len(image_paths)}개의 이미지에서 특징을 추출하는 중...")
        self._log_memory_usage("특징 추출 시작")
        
        for i in tqdm(range(0, len(image_paths), self.batch_size)):
            batch_paths = image_paths[i:i+self.batch_size]
            batch_images = []
            batch_valid_paths = []
            
            # 배치 이미지 로드
            for img_path in batch_paths:
                try:
                    img = image.load_img(img_path, target_size=(224, 224))
                    img_array = image.img_to_array(img)
                    img_array = preprocess_input(img_array)
                    batch_images.append(img_array)
                    batch_valid_paths.append(img_path)
                except Exception as e:
                    logger.warning(f"이미지 로드 실패 {img_path}: {e}")
                    continue
            
            if batch_images:
                try:
                    # 배치를 numpy 배열로 변환
                    batch_array = np.array(batch_images)
                    
                    # 특징 추출
                    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                        batch_features = self.model.predict(batch_array, verbose=0, batch_size=self.batch_size)
                    
                    # 결과 저장
                    for j, feature in enumerate(batch_features):
                        features.append(feature.flatten())
                        valid_paths.append(batch_valid_paths[j])
                    
                    # 배치 메모리 정리
                    del batch_array, batch_features, batch_images
                    
                except Exception as e:
                    logger.error(f"배치 특징 추출 실패: {e}")
                    continue
            
            # 주기적 메모리 정리
            if i % (self.batch_size * 5) == 0:
                gc.collect()
        
        self._log_memory_usage("특징 추출 완료")
        return np.array(features), valid_paths
    
    def classify_category(self, category_info):
        """메모리 최적화된 단일 카테고리 분류"""
        logger.info(f"=== {category_info['name']} 카테고리 분류 시작 ===")
        self._log_memory_usage(f"{category_info['name']} 시작")
        
        try:
            # 이미지 로드
            image_paths = self.load_images_from_folder(category_info['path'])
            
            if len(image_paths) == 0:
                logger.warning(f"{category_info['name']}: 이미지를 찾을 수 없습니다.")
                return
            
            logger.info(f"{category_info['name']}: {len(image_paths)}개 이미지 발견")
            
            # 특징 추출
            features, valid_paths = self.extract_features_batch(image_paths)
            
            if len(features) == 0:
                logger.warning(f"{category_info['name']}: 특징 추출에 실패했습니다.")
                return
            
            # 클러스터 수 조정
            actual_clusters = min(self.n_clusters, len(features))
            
            # 차원 축소
            logger.info(f"{category_info['name']}: 차원 축소 중...")
            n_components = min(50, len(features) - 1)
            pca = PCA(n_components=n_components)
            features_reduced = pca.fit_transform(features)
            
            # 클러스터링
            logger.info(f"{category_info['name']}: {actual_clusters}개 그룹으로 클러스터링 중...")
            kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_reduced)
            
            # 결과 저장
            self.save_classified_images(category_info, valid_paths, labels, actual_clusters)
            
            # 결과 요약
            self.show_category_summary(category_info['name'], labels, actual_clusters)
            
            # 진행상황 저장
            self._save_progress(category_info['name'])
            
            # 메모리 정리
            del features, features_reduced, pca, kmeans, labels
            
        except Exception as e:
            logger.error(f"{category_info['name']} 분류 중 오류 발생: {e}")
        finally:
            # 카테고리별 메모리 정리
            self._clear_memory()
            self._log_memory_usage(f"{category_info['name']} 완료")
    
    def save_classified_images(self, category_info, image_paths, labels, n_clusters):
        """분류된 이미지를 원본 폴더 내에 저장"""
        logger.info(f"{category_info['name']}: 분류된 이미지를 저장 중...")
        
        # 출력 폴더 생성
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
            
            try:
                shutil.copy2(img_path, dest_path)
            except Exception as e:
                logger.warning(f"이미지 복사 실패 {img_path}: {e}")
        
        logger.info(f"{category_info['name']}: 저장 완료 -> {output_base}")
    
    def show_category_summary(self, category_name, labels, n_clusters):
        """카테고리별 클러스터링 결과 요약"""
        logger.info(f"=== {category_name} 분류 결과 요약 ===")
        for i in range(n_clusters):
            count = np.sum(labels == i)
            percentage = (count / len(labels)) * 100
            logger.info(f"그룹 {i+1}: {count}개 이미지 ({percentage:.1f}%)")
    
    def run_all_classifications(self):
        """모든 카테고리 분류 실행 (완료된 카테고리 스킵)"""
        logger.info("카테고리 폴더를 스캔하는 중...")
        self._log_memory_usage("전체 작업 시작")
        
        categories = self.get_category_folders()
        
        if not categories:
            logger.warning("분류할 카테고리를 찾을 수 없습니다.")
            return
        
        # 카테고리 상태 확인
        pending_categories = []
        skipped_count = 0
        
        logger.info(f"총 {len(categories)}개의 카테고리를 발견했습니다.")
        logger.info("카테고리 상태 확인 중...")
        
        for cat in categories:
            is_completed, reason = self._is_category_completed(cat)
            if is_completed:
                logger.info(f"⏭️  SKIP: {cat['name']} - {reason}")
                skipped_count += 1
            else:
                logger.info(f"📋 TODO: {cat['name']} - {reason}")
                pending_categories.append(cat)
        
        logger.info(f"\n📊 상태 요약:")
        logger.info(f"   - 완료됨: {skipped_count}개")
        logger.info(f"   - 처리 대기: {len(pending_categories)}개")
        
        if not pending_categories:
            logger.info("🎉 모든 카테고리가 이미 분류 완료되었습니다!")
            return
        
        # 대기 중인 카테고리만 처리
        logger.info(f"\n🚀 {len(pending_categories)}개 카테고리 분류를 시작합니다...")
        
        for idx, category in enumerate(pending_categories):
            try:
                logger.info(f"📈 진행률: {idx+1}/{len(pending_categories)} (전체: {skipped_count + idx + 1}/{len(categories)})")
                self.classify_category(category)
                
                # 카테고리 간 메모리 정리
                if idx % 3 == 0:
                    self._clear_memory()
                    
            except Exception as e:
                logger.error(f"{category['name']} 분류 중 오류 발생: {e}")
                continue
        
        logger.info("🎉 전체 분류 작업 완료!")
        logger.info(f"📊 최종 결과: {len(self.completed_categories)}개 카테고리 완료")
        self._log_memory_usage("전체 작업 완료")
    
    def reset_progress(self):
        """진행상황 초기화 (모든 카테고리를 다시 처리하고 싶을 때)"""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
            logger.info("진행상황이 초기화되었습니다.")
        self.completed_categories = set()
    
    def show_progress_status(self):
        """현재 진행상황 표시"""
        categories = self.get_category_folders()
        completed_count = 0
        
        logger.info("=== 현재 진행상황 ===")
        for cat in categories:
            is_completed, reason = self._is_category_completed(cat)
            status = "✅ 완료" if is_completed else "⏳ 대기"
            logger.info(f"{status}: {cat['name']} - {reason}")
            if is_completed:
                completed_count += 1
        
        logger.info(f"\n📊 전체 진행률: {completed_count}/{len(categories)} ({completed_count/len(categories)*100:.1f}%)")
    
    def __del__(self):
        """소멸자에서 메모리 정리"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            self._clear_memory()
        except:
            pass


# 사용 예시
if __name__ == "__main__":
    # 설정
    BASE_FOLDER = "C:/Users/AI-LHJ/Desktop/task/images/로맨틱/로맨틱_001_reg_뷴류결과/그룹_16"
    N_CLUSTERS = 20
    BATCH_SIZE = 4
    
    try:
        # 분류기 생성
        classifier = MultiCategoryImageClassifier(
            BASE_FOLDER, 
            N_CLUSTERS, 
            batch_size=BATCH_SIZE,
            skip_completed=True  # 완료된 카테고리 스킵 활성화
        )
        
        # 현재 진행상황 확인 (선택사항)
        classifier.show_progress_status()
        
        # 진행상황 초기화 (필요시)
        # classifier.reset_progress()
        
        # 분류 실행 (완료된 카테고리는 자동으로 스킵됨)
        classifier.run_all_classifications()
        
        logger.info("✅ 모든 이미지 분류가 완료되었습니다!")
        logger.info("📁 각 카테고리 폴더 내에서 '_분류결과' 폴더를 확인하세요.")
        
    except Exception as e:
        logger.error(f"❌ 전체 프로세스 오류: {e}")
    finally:
        # 최종 메모리 정리
        K.clear_session()
        gc.collect()

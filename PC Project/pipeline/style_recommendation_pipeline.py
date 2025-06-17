# pipeline/style_recommendation_pipeline.py
from preprocessing.image_preprocessor import ImagePreprocessor
from detection.face_detector import FaceDetector
from analysis.color_extractor import ColorExtractor
from analysis.personal_color_analyzer import PersonalColorAnalyzer
from analysis.face_shape_analyzer import FaceShapeAnalyzer
from classification.fashion_style_classifier import FashionStyleClassifier
from recommendation.recommendation_generator import RecommendationGenerator

class StyleRecommendationPipeline:
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.face_detector = FaceDetector()
        self.color_extractor = ColorExtractor()
        self.personal_color_analyzer = PersonalColorAnalyzer()
        self.face_shape_analyzer = FaceShapeAnalyzer()
        self.fashion_style_classifier = FashionStyleClassifier()
        self.recommendation_generator = RecommendationGenerator()

    def process(self, image_data: bytes):
        img = self.preprocessor.preprocess(image_data)
        # 얼굴형 분석
        face_shape = self.face_shape_analyzer.analyze(img)
        # 얼굴 감지 및 특징 추출
        face, landmarks = self.face_detector.detect_face(img)
        # 색상 추출
        colors = self.color_extractor.extract_dominant_colors(img)
        features = colors.mean(axis=0)
        # 퍼스널컬러 분석
        personal_color = self.personal_color_analyzer.analyze(features)
        # 의류 스타일 분류
        fashion_style = self.fashion_style_classifier.classify(img)
        # 추천 생성
        recommendations = self.recommendation_generator.generate(personal_color, fashion_style)
        # 결과 통합
        return {
            "personal_color": personal_color,
            "fashion_style": fashion_style,
            "face_shape": face_shape,
            **recommendations
        }

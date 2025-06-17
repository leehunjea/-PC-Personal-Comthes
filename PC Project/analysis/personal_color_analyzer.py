import joblib
import numpy as np

class PersonalColorAnalyzer:
    def __init__(self, model_path="C:/Users/AI-LHJ/Desktop/PC Project/analysis/personal_color.h5"):
        self.model = joblib.load(model_path)

    def analyze(self, features: np.ndarray) -> str:
        season = self.model.predict([features])[0]
        season_mapping = {0: "봄", 1: "여름", 2: "가을", 3: "겨울"}
        return season_mapping[season]

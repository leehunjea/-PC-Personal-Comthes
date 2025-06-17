class RecommendationGenerator:
    def __init__(self):
        self.color_recommendations = {
            "봄": ["밝은 노랑", "복숭아색"],
            "여름": ["라벤더", "파스텔 블루"],
            "가을": ["카멜", "올리브 그린"],
            "겨울": ["검정", "흰색"]
        }
        self.style_recommendations = {
            "캐주얼": ["청바지", "티셔츠"],
            "포멀": ["슈트", "블라우스"],
            # ...
        }

    def generate(self, personal_color, fashion_style):
        return {
            "personal_color": personal_color,
            "recommended_colors": self.color_recommendations.get(personal_color, []),
            "fashion_style": fashion_style,
            "recommended_items": self.style_recommendations.get(fashion_style, [])
        }

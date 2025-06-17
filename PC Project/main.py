from pipeline.style_recommendation_pipeline import StyleRecommendationPipeline

if __name__ == "__main__":
    with open("C:/Users/AI-LHJ/Desktop/Task/TEST3.jpg", "rb") as f:
        image_data = f.read()
    pipeline = StyleRecommendationPipeline()
    result = pipeline.process(image_data)
    print(result)

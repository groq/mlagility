# labels: test_group::monthly author::model-attribution-challenge name::opt-350m downloads::442 license::other task::Natural_Language_Processing sub_task::Text_Generation
from transformers import pipeline

generator = pipeline('text-generation', model="facebook/opt-350m")
generator("Hello, I'm am conscious and")


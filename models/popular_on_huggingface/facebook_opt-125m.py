# labels: test_group::monthly,daily author::facebook name::opt-125m downloads::228,909 license::other task::Natural_Language_Processing sub_task::Text_Generation
from transformers import pipeline

generator = pipeline('text-generation', model="facebook/opt-125m")
generator("Hello, I'm am conscious and")


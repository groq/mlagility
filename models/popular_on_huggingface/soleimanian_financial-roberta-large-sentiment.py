# labels: test_group::monthly author::soleimanian name::financial-roberta-large-sentiment downloads::261 license::apache-2.0 task::Natural_Language_Processing sub_task::Text_Classification

from transformers import pipeline
sentiment_analysis = pipeline("sentiment-analysis",model="soleimanian/financial-roberta-large-sentiment")
print(sentiment_analysis("In fiscal 2021, we generated a net yield of approximately 4.19% on our investments, compared to approximately 5.10% in fiscal 2020."))
  
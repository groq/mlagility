# labels: test_group::monthly author::ml4pubmed name::BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section downloads::356 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline

model_tag = "ml4pubmed/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section"
classifier = pipeline(
              'text-classification', 
              model=model_tag, 
            )
            
prompt = """
Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.
"""

classifier(
    prompt,
) # classify the sentence

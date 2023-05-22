# labels: test_group::monthly author::raynardj name::ner-gene-dna-rna-jnlpba-pubmed downloads::10,415 license::apache-2.0 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import pipeline

PRETRAINED = "raynardj/ner-gene-dna-rna-jnlpba-pubmed"
ner = pipeline(task="ner",model=PRETRAINED, tokenizer=PRETRAINED)
ner("Your text", aggregation_strategy="first")

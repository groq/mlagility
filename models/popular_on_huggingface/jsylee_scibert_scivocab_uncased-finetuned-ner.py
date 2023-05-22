# labels: test_group::monthly author::jsylee name::scibert_scivocab_uncased-finetuned-ner downloads::645 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import (AutoModelForTokenClassification, 
                          AutoTokenizer, 
                          pipeline,
                          )

model_checkpoint = "jsylee/scibert_scivocab_uncased-finetuned-ner"
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=5,
                                                        id2label={0: 'O', 1: 'B-DRUG', 2: 'I-DRUG', 3: 'B-EFFECT', 4: 'I-EFFECT'} 
                                                        )                                                        
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

model_pipeline = pipeline(task="ner", model=model, tokenizer=tokenizer)

print( model_pipeline ("Abortion, miscarriage or uterine hemorrhage associated with misoprostol (Cytotec), a labor-inducing drug."))

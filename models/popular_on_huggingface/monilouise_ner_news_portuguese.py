# labels: test_group::monthly author::monilouise name::ner_news_portuguese downloads::463 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import BertForTokenClassification, DistilBertTokenizerFast, pipeline
model = BertForTokenClassification.from_pretrained('monilouise/ner_pt_br')
tokenizer = DistilBertTokenizerFast.from_pretrained('neuralmind/bert-base-portuguese-cased'
                                                    , model_max_length=512
                                                    , do_lower_case=False
                                                    )
nlp = pipeline('ner', model=model, tokenizer=tokenizer, grouped_entities=True)
result = nlp("O Tribunal de Contas da União é localizado em Brasília e foi fundado por Rui Barbosa.")

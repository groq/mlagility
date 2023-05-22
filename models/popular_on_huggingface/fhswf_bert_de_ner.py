# labels: test_group::monthly author::fhswf name::bert_de_ner downloads::4,217 license::cc-by-sa-4.0 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import pipeline

classifier = pipeline('ner', model="fhswf/bert_de_ner")
classifier('Von der Organisation „medico international“ hieß es, die EU entziehe sich seit vielen Jahren der Verantwortung für die Menschen an ihren Außengrenzen.')



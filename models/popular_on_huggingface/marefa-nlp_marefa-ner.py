# labels: test_group::monthly author::marefa-nlp name::marefa-ner downloads::548 task::Natural_Language_Processing sub_task::Token_Classification
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

custom_labels = ["O", "B-job", "I-job", "B-nationality", "B-person", "I-person", "B-location","B-time", "I-time", "B-event", "I-event", "B-organization", "I-organization", "I-location", "I-nationality", "B-product", "I-product", "B-artwork", "I-artwork"]

def _extract_ner(text: str, model: AutoModelForTokenClassification,
                 tokenizer: AutoTokenizer, start_token: str="▁"):
    tokenized_sentence = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    tokenized_sentences = tokenized_sentence['input_ids'].numpy()

    with torch.no_grad():
        output = model(**tokenized_sentence)

    last_hidden_states = output[0].numpy()
    label_indices = np.argmax(last_hidden_states[0], axis=1)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_sentences[0])
    special_tags = set(tokenizer.special_tokens_map.values())

    grouped_tokens = []
    for token, label_idx in zip(tokens, label_indices):
        if token not in special_tags:
            if not token.startswith(start_token) and len(token.replace(start_token,"").strip()) > 0:
                grouped_tokens[-1]["token"] += token
            else:
                grouped_tokens.append({"token": token, "label": custom_labels[label_idx]})

    # extract entities
    ents = []
    prev_label = "O"
    for token in grouped_tokens:
        label = token["label"].replace("I-","").replace("B-","")
        if token["label"] != "O":
            
            if label != prev_label:
                ents.append({"token": [token["token"]], "label": label})
            else:
                ents[-1]["token"].append(token["token"])
            
        prev_label = label
    
    # group tokens
    ents = [{"token": "".join(rec["token"]).replace(start_token," ").strip(), "label": rec["label"]}  for rec in ents ]

    return ents

model_cp = "marefa-nlp/marefa-ner"

tokenizer = AutoTokenizer.from_pretrained(model_cp)
model = AutoModelForTokenClassification.from_pretrained(model_cp, num_labels=len(custom_labels))

samples = [
    "تلقى تعليمه في الكتاب ثم انضم الى الأزهر عام 1873م. تعلم على يد السيد جمال الدين الأفغاني والشيخ محمد عبده",
    "بعد عودته إلى القاهرة، التحق نجيب الريحاني فرقة جورج أبيض، الذي كان قد ضمَّ - قُبيل ذلك - فرقته إلى فرقة سلامة حجازي . و منها ذاع صيته",
    "في استاد القاهرة، قام حفل افتتاح بطولة كأس الأمم الأفريقية بحضور رئيس الجمهورية و رئيس الاتحاد الدولي لكرة القدم",
    "من فضلك أرسل هذا البريد الى صديقي جلال الدين في تمام الساعة الخامسة صباحا في يوم الثلاثاء القادم",
    "امبارح اتفرجت على مباراة مانشستر يونايتد مع ريال مدريد في غياب الدون كرستيانو رونالدو",
    "لا تنسى تصحيني الساعة سبعة, و ضيف في الجدول اني احضر مباراة نادي النصر غدا",
]

# [optional]
samples = [ " ".join(word_tokenize(sample.strip())) for sample in samples if sample.strip() != "" ]

for sample in samples:
    ents = _extract_ner(text=sample, model=model, tokenizer=tokenizer, start_token="▁")

    print(sample)
    for ent in ents:
        print("\t",ent["token"],"==>",ent["label"])
    print("========\n")

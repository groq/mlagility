# labels: test_group::monthly author::liandarizkia name::SA01-IndoBert downloads::944 task::Natural_Language_Processing sub_task::Text_Classification
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

pretrained= "liandarizkia/SA01-IndoBert"

model = AutoModelForSequenceClassification.from_pretrained(pretrained)
tokenizer = AutoTokenizer.from_pretrained(pretrained)

sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

label_index = {'LABEL_0': 'negative', 'LABEL_1': 'positive', 'LABEL_2': 'neutral'}

text = """Aku baru sebulan udah pengen lepas rasanya. Udah gak peduli uang yang keluar sayang. Pokoknya gak nyaman, setiap hari sedih terus. Akhirnya aku cerita ke dokterku kalau aku dah gak kuat aku bilang kalau bakal bertahan 2 atau 3 bulan dari pemasangan behel. Setelah itu aku minta buat beneran lepas aja. Pokoknya jangan ragu buat cerita ke dokter"""

result = sentiment_analysis(text)
status = label_index[result[0]['label']]
score = result[0]['score']
print(f'Text: {text} | Label : {status} ({score * 100:.3f}%)')

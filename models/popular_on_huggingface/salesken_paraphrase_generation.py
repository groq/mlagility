# labels: test_group::monthly author::salesken name::paraphrase_generation downloads::302 license::apache-2.0 task::Natural_Language_Processing sub_task::Text_Generation
from transformers import AutoTokenizer, AutoModelWithLMHead 

import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
else :
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained("salesken/paraphrase_generation")  
model = AutoModelWithLMHead.from_pretrained("salesken/paraphrase_generation").to(device)

input_query="every moment is a fresh beginning"
query= input_query + " ~~ "

input_ids = tokenizer.encode(query.lower(), return_tensors='pt').to(device)
sample_outputs = model.generate(input_ids,
                                do_sample=True,
                                num_beams=1, 
                                max_length=128,
                                temperature=0.9,
                                top_p= 0.99,
                                top_k = 30,
                                num_return_sequences=40)
paraphrases = []
for i in range(len(sample_outputs)):
    r = tokenizer.decode(sample_outputs[i], skip_special_tokens=True).split('||')[0]
    r = r.split(' ~~ ')[1]
    if r not in paraphrases:
        paraphrases.append(r)

print(paraphrases)


# labels: test_group::monthly author::salesken name::text_generate downloads::205 task::Natural_Language_Processing sub_task::Text_Generation
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
else :
    device = "cpu"
    
tokenizer = AutoTokenizer.from_pretrained("salesken/text_generate")
model = AutoModelWithLMHead.from_pretrained("salesken/text_generate").to(device)

input_query="tough challenges make you stronger.  "
input_ids = tokenizer.encode(input_query.lower(), return_tensors='pt').to(device)

sample_outputs = model.generate(input_ids,
                                do_sample=True,
                                num_beams=1, 
                                max_length=1024,
                                temperature=0.99,
                                top_k = 10,
                                num_return_sequences=1)

for i in range(len(sample_outputs)):
    print(tokenizer.decode(sample_outputs[i], skip_special_tokens=True))

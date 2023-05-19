# labels: test_group::monthly author::salesken name::natural_rephrase downloads::203 license::apache-2.0 task::Natural_Language_Processing sub_task::Text_Generation
from transformers import AutoTokenizer, AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained("salesken/natural_rephrase")
model = AutoModelWithLMHead.from_pretrained("salesken/natural_rephrase")


Input_query="Hey Siri, Send message to mom to say thank you for the delicious dinner yesterday"
query= Input_query + " ~~ "
input_ids = tokenizer.encode(query.lower(), return_tensors='pt')
sample_outputs = model.generate(input_ids,
                            do_sample=True,
                            num_beams=1, 
                            max_length=len(Input_query),
                            temperature=0.2,
                            top_k = 10,
                            num_return_sequences=1)
for i in range(len(sample_outputs)):
    result = tokenizer.decode(sample_outputs[i], skip_special_tokens=True).split('||')[0].split('~~')[1]
    print(result)

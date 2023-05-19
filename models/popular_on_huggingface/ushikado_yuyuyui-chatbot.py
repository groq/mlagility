# labels: test_group::monthly author::ushikado name::yuyuyui-chatbot downloads::483 task::Natural_Language_Processing sub_task::Text_Generation
from transformers import T5Tokenizer, AutoModelForCausalLM

tokenizer = T5Tokenizer.from_pretrained("ushikado/yuyuyui-chatbot")
model = AutoModelForCausalLM.from_pretrained("ushikado/yuyuyui-chatbot")

query_text = "<某>神樹様について教えてください。</s><上里 ひなた>"
input_tensor = tokenizer.encode(query_text, add_special_tokens=False, return_tensors="pt")
output_list = model.generate(input_tensor, max_length=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
output_text = tokenizer.decode(output_list[0])
print(output_text)
"""
<某> 神樹様について教えてください。</s> <上里 ひなた> 造反神は、神樹様の分裂を煽り出して、神樹様の中の一体感を高める存在です。</s>
"""

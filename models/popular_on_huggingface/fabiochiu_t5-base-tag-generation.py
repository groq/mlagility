# labels: test_group::monthly author::fabiochiu name::t5-base-tag-generation downloads::189 license::apache-2.0 task::Natural_Language_Processing sub_task::Text2Text_Generation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
nltk.download('punkt')

tokenizer = AutoTokenizer.from_pretrained("fabiochiu/t5-base-tag-generation")
model = AutoModelForSeq2SeqLM.from_pretrained("fabiochiu/t5-base-tag-generation")

text = """
Python is a high-level, interpreted, general-purpose programming language. Its
design philosophy emphasizes code readability with the use of significant
indentation. Python is dynamically-typed and garbage-collected.
"""

inputs = tokenizer([text], max_length=512, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10,
                        max_length=64)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
tags = list(set(decoded_output.strip().split(", ")))

print(tags)
# ['Programming', 'Code', 'Software Development', 'Programming Languages',
#  'Software', 'Developer', 'Python', 'Software Engineering', 'Science',
#  'Engineering', 'Technology', 'Computer Science', 'Coding', 'Digital', 'Tech',
#  'Python Programming']

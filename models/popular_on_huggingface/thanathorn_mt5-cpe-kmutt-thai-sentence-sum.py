# labels: test_group::monthly author::thanathorn name::mt5-cpe-kmutt-thai-sentence-sum downloads::744 task::Natural_Language_Processing sub_task::Summarization
from simpletransformers.t5 import T5Model, T5Args
from torch import cuda

model = T5Model("t5", "thanathorn/mt5-cpe-kmutt-thai-sentence-sum", use_cuda=cuda.is_available())

sentence = "simplify: ถ้าพูดถึงขนมหวานในตำนานที่ชื่นใจที่สุดแล้วละก็ต้องไม่พ้น น้ำแข็งใส แน่เพราะว่าเป็นอะไรที่ชื่นใจสุด"
prediction = model.predict([sentence])
print(prediction[0])

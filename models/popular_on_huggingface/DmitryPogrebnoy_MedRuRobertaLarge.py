# labels: test_group::monthly author::DmitryPogrebnoy name::MedRuRobertaLarge downloads::194 license::apache-2.0 task::Natural_Language_Processing sub_task::Fill-Mask
from transformers import pipeline
pipeline = pipeline('fill-mask', model='DmitryPogrebnoy/MedRuRobertaLarge')
pipeline("У пациента <mask> боль в грудине.")


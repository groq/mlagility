# labels: test_group::monthly author::uw-madison name::nystromformer-512 downloads::659 task::Fill-Mask
from transformers import pipeline
unmasker = pipeline('fill-mask', model='uw-madison/nystromformer-512')
unmasker("Paris is the [MASK] of France.")



# Getting image from docker hub
from httpd

RUN apt-get update && apt-get install -y --no-install-recommends python3-dev python3-setuptools python3-wheel python3-pip 
ENV PYTHONPATH "${PYTHONPATH}:/usr/bin/python3"
RUN pip install onnxruntime

# CMD ["pip", "list"]
# CMD ["lsb_release", "-a"]
# docker build --no-cache -t ort .
# docker run -d --rm -v ~:/app --name ort_mlas ort
# docker exec ort_mlas ls /usr/bin/python3 /app/runonnxmodel.py --onnx-file /app/bert_d59172a2-op14-opt-f16.onnx --outputs-file out.txt --errors-file errors.txt --iterations 10
# docker cp ort_mlas:/usr/local/apache2/out_docker.txt .
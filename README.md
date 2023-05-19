# The MLAgility Project

[![MLAgility tests](https://github.com/groq/mlagility/actions/workflows/test_mlagility.yml/badge.svg)](https://github.com/groq/mlagility/tree/main/test "Check out our tests")
[![onnxflow tests](https://github.com/groq/mlagility/actions/workflows/test_onnxflow.yml/badge.svg)](https://github.com/groq/mlagility/tree/main/test "Check out our tests")
[![MLAgility GPU tests](https://github.com/groq/mlagility/actions/workflows/test_gpu_mlagility.yml/badge.svg)](https://github.com/groq/mlagility/tree/main/test "Check out our tests")
[![OS - Linux](https://img.shields.io/badge/OS-Linux-blue?logo=linux&logoColor=white)](https://github.com/groq/mlagility/blob/main/docs/install.md "Check out our instructions")
[![Made with Python](https://img.shields.io/badge/Python-3.8,3.10-blue?logo=python&logoColor=white)](https://github.com/groq/mlagility/blob/main/docs/install.md "Check out our instructions")
[![License](https://img.shields.io/badge/License-MIT-blue)](https://github.com/groq/mlagility/blob/main/LICENSE "Check out our license")


MLAgility offers a complementary approach to MLPerf by examining the capability of vendors to provide turnkey solutions to a corpus of hundreds of off-the-shelf models. All of the model scripts and benchmarking code are published as open source software. The performance data is available at our [Huggingface Space](https://huggingface.co/spaces/Groq/mlagility).


## Benchmarking Tool

Our _benchit_ CLI allows you to benchmark Pytorch models without changing a single line of code. The demo below shows BERT-Base being benchmarked on both Nvidia A100 and Intel Xeon. For more information, check out our [Tutorials](https://github.com/groq/mlagility/blob/main/examples/cli/readme.md) and [Tools User Guide](https://github.com/groq/mlagility/blob/main/docs/tools_user_guide.md).

<img src="https://github.com/groq/mlagility/raw/main/docs/img/mlagility.gif"  width="800"/>

You can reproduce this demo by trying out the [Just Benchmark BERT](https://github.com/groq/mlagility/blob/main/examples/cli/readme.md#just-benchmark-bert) tutorial.

## 1000+ Models

[![Transformers](https://img.shields.io/github/directory-file-count/groq/mlagility/models/transformers?label=transformers)](https://github.com/groq/mlagility/tree/main/models/transformers "Transformer models")
[![Diffusers](https://img.shields.io/github/directory-file-count/groq/mlagility/models/diffusers?label=diffusers)](https://github.com/groq/mlagility/tree/main/models/diffusers "Diffusion models")
[![popular_on_huggingface](https://img.shields.io/github/directory-file-count/groq/mlagility/models/popular_on_huggingface?label=popular_on_huggingface)](https://github.com/groq/mlagility/tree/main/models/popular_on_huggingface "Popular Models on Huggingface")
[![torch_hub](https://img.shields.io/github/directory-file-count/groq/mlagility/models/torch_hub?label=torch_hub)](https://github.com/groq/mlagility/tree/main/models/torch_hub "Models from Torch Hub")
[![torchvision](https://img.shields.io/github/directory-file-count/groq/mlagility/models/torchvision?label=torchvision)](https://github.com/groq/mlagility/tree/main/models/torchvision "Models from Torch Vision")
[![timm](https://img.shields.io/github/directory-file-count/groq/mlagility/models/timm?label=timm)](https://github.com/groq/mlagility/tree/main/models/timm "Pytorch Image Models")

This repository is home to a diverse corpus of hundreds of models. We are actively working on increasing the number of models on our model library. You can see the set of models in each category by clicking on the corresponding badge.

## Benchmarking results

We are also working on making MLAgility results publicly available at our [Huggingface Space](https://huggingface.co/spaces/Groq/mlagility). Check it out!

<img src="https://github.com/groq/mlagility/raw/main/docs/img/dashboard.gif"  width="800"/>

## How everything fits together

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/block_diagram_light.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/img/block_diagram_dark.png">
  <img src="docs/img/block_diagram_dark.png" alt="Architecture block diagram" width="700" height="500" />
</picture>

The diagram above illustrates the MLAgility repository's structure. Simply put, the MLAgility models are leveraged by our benchmarking tool, benchit, to produce benchmarking outcomes showcased on our Hugging Face Spaces page. 
## Installation

Please refer to our [mlagility installation guide](https://github.com/groq/mlagility/blob/main/docs/install.md) to get instructions on how to install mlagility.

## Contributing

We are actively seeking collaborators from across the industry. If you would like to contribute to this project, please check out our [contribution guide](https://github.com/groq/mlagility/blob/main/docs/contribute.md).

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/groq/mlagility/blob/main/LICENSE) file for details.

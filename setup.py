from setuptools import setup

with open("src/mlagility/version.py", encoding="utf-8") as fp:
    version = fp.read().split('"')[1]

setup(
    name="mlagility",
    version=version,
    description="MLAgility Benchmark and Tools",
    url="https://github.com/groq/mlagility",
    author="Jeremy Fowers, Daniel Holanda, Ramakrishnan Sivakumar, Victoria Godsoe",
    author_email="jfowers@groq.com, dhnoronha@groq.com, rsivakumar@groq.com, vgodsoe@groq.com",
    license="MIT",
    package_dir={"": "src", "mlagility_models": "models"},
    packages=[
        "mlagility",
        "mlagility.analysis",
        "mlagility.api",
        "mlagility.cli",
        "mlagility.common",
        "onnxflow",
        "onnxflow.common",
        "onnxflow.justbuildit",
        "onnxflow.model",
        "mlagility_models",
        "mlagility_models.diffusers",
        "mlagility_models.graph_convolutions",
        "mlagility_models.llm",
        "mlagility_models.llm_layer",
        "mlagility_models.popular_on_huggingface",
        "mlagility_models.selftest",
        "mlagility_models.timm",
        "mlagility_models.torch_hub",
        "mlagility_models.torchvision",
        "mlagility_models.transformers",
    ],
    install_requires=[
        "invoke>=2.0.0",
        "onnx==1.14.0",
        "onnxmltools==1.11.2",
        "hummingbird-ml==0.4.4",
        "scikit-learn==1.1.1",
        "xgboost==1.6.1",
        "lightgbm==3.3.5",
        "onnxruntime==1.15.1",
        "paramiko==2.11.0",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "pyyaml>=5.4",
        "packaging>=20.9",
        "pandas>=1.5.3",
        "fasteners",
    ],
    extras_require={
        "tensorflow": [
            "tensorflow-cpu>=2.8.1,<2.12",
            "tf2onnx>=1.12.0",
        ],
        "groq": [
            "groqflow==3.1.0",
        ],
    },
    classifiers=[],
    entry_points={
        "console_scripts": [
            "benchit=mlagility:benchitcli",
        ]
    },
    python_requires=">=3.8, <3.11",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data={
        "mlagility.api": ["Dockerfile"],
        "mlagility_models": ["requirements.txt", "readme.md"],
    },
)

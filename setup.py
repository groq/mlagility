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
    package_dir={"": "src"},
    packages=["mlagility", "onnxflow"],
    install_requires=[
        "invoke>=2.0.0",
        "onnx>=1.11.0",
        "onnxmltools==1.10.0",
        "hummingbird-ml==0.4.4",
        "scikit-learn==1.1.1",
        "xgboost==1.6.1",
        "onnxruntime>=1.10.0",
        "paramiko==2.11.0",
        "torch>=1.12.1",
        "protobuf>=3.17.3",
        "pyyaml>=5.4",
        "typeguard>=2.3.13",
        "packaging>=21.3",
    ],
    extras_require={
        "tensorflow": [
            "tensorflow-cpu>=2.8.1,<2.12",
            "tf2onnx>=1.12.0",
        ],
        "groq": [
            "groqflow==2.5.2",
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
)

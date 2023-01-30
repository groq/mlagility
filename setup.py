from setuptools import setup, find_packages

with open("mlabench/version.py", encoding="utf-8") as fp:
    version = fp.read().split('"')[1]

setup(
    name="mlabench",
    version=version,
    description="MLAgility Benchmarking Tools",
    url="https://github.com/groq/mlabench",
    author="Groq",
    author_email="sales@groq.com",
    license="MIT",
    packages=find_packages(
        exclude=["*.__pycache__.*"],
    ),
    install_requires=[
        "transformers",
        "groqflow>=2.5.2",
    ],
    classifiers=[],
    entry_points={
        "console_scripts": [
            "benchit=mlabench:benchitcli",
            "autogroq=mlabench:autogroq",
        ]
    },
)

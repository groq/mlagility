from setuptools import setup, find_packages

version_number = ""
with open("VERSION_NUMBER", encoding="utf-8") as f:
    version_number = f.readline().strip()

setup(
    name="mlabench",
    version=version_number,
    description="MLAgility Benchmarking Tools",
    url="https://github.com/groq/mlabench",
    author="Groq",
    author_email="sales@groq.com",
    license="Groq License Agreement",
    packages=find_packages(
        exclude=["*.__pycache__.*"],
    ),
    install_requires=[
        "transformers",
        # "groqflow>=2.5.0",
    ],
    classifiers=[],
    entry_points={
        "console_scripts": [
            "benchit=mlabench:benchitcli",
            "autogroq=mlabench:autogroq",
        ]
    },
)

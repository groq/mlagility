from setuptools import setup

with open("src/mlagility/version.py", encoding="utf-8") as fp:
    version = fp.read().split('"')[1]

setup(
    name="mlagility",
    version=version,
    description="MLAgility Benchmark and Tools",
    url="https://github.com/groq/mlagility",
    author="Groq",
    author_email="sales@groq.com",
    license="MIT",
    package_dir={"": "src"},
    packages=["mlagility"],
    install_requires=[
        "transformers",
        "groqflow>=2.5.2",
        "invoke>=2.0.0"
    ],
    classifiers=[],
    entry_points={
        "console_scripts": [
            "benchit=mlagility:benchitcli",
        ]
    },
    python_requires="==3.8.*",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)

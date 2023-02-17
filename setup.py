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
    packages=["mlagility"],
    install_requires=[
        "groqflow @ https://test-files.pythonhosted.org/packages/a0/96/2eba6c53f93e9aa1b9de3f80f01ac23db7256b62d31d932d9814a0367ed4/groqflow-3.0.1.tar.gz",
        "invoke>=2.0.0",
    ],
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

# The MLAgility Project

MLAgility is on a mission to create and benchmark an enormous corpus of deep learning models. All of the model scripts, benchmarking code will be published as open source software and the performance data will be available on the open web as a dashboard on a Huggingface Space (coming soon).

This repository is home to both:
- A diverse corpus of hundreds of models, in `models/`
  - See the [benchmark readme](https://github.com/groq/mlagility/blob/main/models/readme.md) to learn more
- Automated benchmarking tools, the `benchit` CLI and API, for CPUs, GPUs, and GroqChip processors, in `src/`
  - See the [benchit CLI/API user guide](https://github.com/groq/mlagility/blob/main/docs/benchit_user_guide.md) to learn more

## Contributing

MLAgility is open source software under the MIT license and we are welcoming contributions. While MLAgility is funded by Groq, we aspire for the tools and benchmarks to be as vendor-neutral as possible.

One of the easiest ways to contribute is to add a model to the benchmark. To do, simply add a `.py` file to `models/` that instantiates and calls a `PyTorch` or `Keras` model. The automated benchmarking infrastructure will do the rest!

### Issues

Please file any bugs or feature requests you have as an [Issue](https://github.com/groq/mlagility/issues) and we will take a look.

### Pull Requests

You will have to sign a [CLA](https://github.com/groq/mlagility/blob/main/cla.md) to submit a PR.

Please have a discussion with the team before making major changes.

### Testing

Tests are defined in `tests/` and run automatically on each PR, as defined in our [testing action](https://github.com/groq/mlagility/blob/main/.github/workflows/test.yml). This action performs both linting and unit testing and must succeed before code can be merged.

We don't have any fancy testing framework set up yet. If you want to run tests locally:
- Activate a `conda` environment that has `mlagility` (this package) installed
- Run `conda install pylint` if you haven't already (other pylint installs will give you a lot of import warnings)
- Run `pylint src --rcfile .pylintrc` from the repo root
- Run `python test.py` for each test script in `test/`

### Versioning

We use semantic versioning, as described in [versioning.md](https://github.com/groq/mlagility/blob/main/docs/versioning.md).

### Release

We have an [action](https://github.com/groq/mlagility/blob/main/.github/workflows/publish-to-test-pypi.yml) for automatically deploying `mlagility` packages to PyPI.
- An `mlagility` package is deployed to Test PyPI on every commit (to any branch), as long as it has a unique version number
- An `mlagility` package is deployed to PyPI on every push to `main`, as long as it has:
  - A unique version number
  - The git commit has been tagged: run `git tag vX.X.X` to tag a commit with version X.X.X, then `git push --tags` to push that tag to GitHub

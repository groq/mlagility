# MlAgility Contribution Guide

MLAgility is open source software under the MIT license and we are welcoming contributions. While MLAgility is funded by Groq, we aspire for the tools and benchmarks to be as vendor-neutral as possible.

One of the easiest ways to contribute is to add a model to the benchmark. To do so, simply add a `.py` file to the `models/` directory that instantiates and calls a supported type of model (see [Tools User Guide](https://github.com/groq/mlagility/blob/main/docs/tools_user_guide.md) to learn more). The automated benchmarking infrastructure will do the rest!

Please read about our [repo and code organization](https://github.com/groq/mlagility/blob/main/docs/code.md) before getting started.

## Issues

Please file any bugs or feature requests you have as an [Issue](https://github.com/groq/mlagility/issues) and we will take a look.

## Pull Requests

Contribute code by creating a pull request (PR). You will have to sign a standard [CLA](https://github.com/groq/mlagility/blob/main/cla.md) before you can merge the PR. Your PR will be reviewed by one of the [repo maintainers](https://github.com/groq/mlagility/blob/main/CODEOWNERS).

Please have a discussion with the team before making major changes. The best way to start such a discussion is to file an [Issue](https://github.com/groq/mlagility/issues) and seek a response from one of the [repo maintainers](https://github.com/groq/mlagility/blob/main/CODEOWNERS).

## Testing

Tests are defined in `tests/` and run automatically on each PR, as defined in our [testing action](https://github.com/groq/mlagility/blob/main/.github/workflows/test.yml). This action performs both linting and unit testing and must succeed before code can be merged.

We don't have any fancy testing framework set up yet. If you want to run tests locally:
- Activate a `conda` environment that has `mlagility` (this package) installed.
- Run `conda install pylint` if you haven't already (other pylint installs will give you a lot of import warnings).
- Run `pylint src --rcfile .pylintrc` from the repo root.
- Run `python test.py` for each test script in `test/`.

## Versioning

We use semantic versioning, as described in [versioning.md](https://github.com/groq/mlagility/blob/main/docs/versioning.md).

## Release

We have an [action](https://github.com/groq/mlagility/blob/main/.github/workflows/publish-to-test-pypi.yml) for automatically deploying `mlagility` packages to PyPI.
- An `mlagility` package is deployed to PyPI on every push to `main`, as long as it has:
  - A unique version number (which is defined in `src/mlagility/version.py`).
  - The git commit has been tagged: run `git tag vX.X.X` to tag a commit with version X.X.X, then `git push --tags` to push that tag to GitHub.
<table>
<tr>
<td>⚠️</td>
<td>
<strong>NOTE:</strong> Welcome to MLAgility. Our team is hard at work to bring you the best experience possible. However, as with any work in progress, there may be bugs, unfinished features, and other issues that need to be addressed. We appreciate your patience and understanding as we work to make this project the best it can be. Stay tuned for updates and improvements in the near future.
Feel free to open issues with any early feedback.
</td>
<td>⚠️</td>
</tr>
</table>

----
# The MLAgility Project

MLAgility is on a mission to create and benchmark an enormous corpus of deep learning models. All of the model scripts and benchmarking code will be published as open source software. The performance data will be available on the open web as a dashboard on a Huggingface Space (coming soon).

This repository is home to both:
- A diverse corpus of hundreds of models, in `models/`
  - See the [benchmark readme](https://github.com/groq/mlagility/blob/main/models/readme.md) to learn more
- Automated benchmarking tools, the `benchit` CLI and API, for CPUs, GPUs, and GroqChip processors, in `src/`
  - See the [Tools User Guide](https://github.com/groq/mlagility/blob/main/docs/tools_user_guide.md) to learn more

## Installation

Please refer to our [mlagility installation guide](https://github.com/groq/mlagility/blob/main/docs/install.md) to get instructions on how to install mlagility.

## Contributing

MLAgility is open source software under the MIT license and we are welcoming contributions. While MLAgility is funded by Groq, we aspire for the tools and benchmarks to be as vendor-neutral as possible.

One of the easiest ways to contribute is to add a model to the benchmark. To do so, simply add a `.py` file to the `models/` directory that instantiates and calls a supported type of model (see [Tools User Guide](https://github.com/groq/mlagility/blob/main/docs/tools_user_guide.md) to learn more). The automated benchmarking infrastructure will do the rest!

Please read about our [repo and code organization](https://github.com/groq/mlagility/blob/main/docs/code.md) before getting started.

### Issues

Please file any bugs or feature requests you have as an [Issue](https://github.com/groq/mlagility/issues) and we will take a look.

### Pull Requests

Contribute code by creating a pull request (PR). You will have to sign a [CLA](https://github.com/groq/mlagility/blob/main/cla.md) before you can merge the PR. Your PR will be reviewed by one of the [repo maintainers](https://github.com/groq/mlagility/blob/main/CODEOWNERS).

Please have a discussion with the team before making major changes. The best way to start such a discussion is to file an [Issue](https://github.com/groq/mlagility/issues) and seek a response from one of the [repo maintainers](https://github.com/groq/mlagility/blob/main/CODEOWNERS).

### Testing

Tests are defined in `tests/` and run automatically on each PR, as defined in our [testing action](https://github.com/groq/mlagility/blob/main/.github/workflows/test.yml). This action performs both linting and unit testing and must succeed before code can be merged.

We don't have any fancy testing framework set up yet. If you want to run tests locally:
- Activate a `conda` environment that has `mlagility` (this package) installed.
- Run `conda install pylint` if you haven't already (other pylint installs will give you a lot of import warnings).
- Run `pylint src --rcfile .pylintrc` from the repo root.
- Run `python test.py` for each test script in `test/`.

### Versioning

We use semantic versioning, as described in [versioning.md](https://github.com/groq/mlagility/blob/main/docs/versioning.md).

### Release

We have an [action](https://github.com/groq/mlagility/blob/main/.github/workflows/publish-to-test-pypi.yml) for automatically deploying `mlagility` packages to PyPI.
- An `mlagility` package is deployed to Test PyPI on every commit (to any branch), as long as it has a unique version number.
- An `mlagility` package is deployed to PyPI on every push to `main`, as long as it has:
  - A unique version number (which is defined in `src/mlagility/version.py`).
  - The git commit has been tagged: run `git tag vX.X.X` to tag a commit with version X.X.X, then `git push --tags` to push that tag to GitHub.

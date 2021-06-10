# Contributing

Tarteel-ML is an open-source project, which means you can help us make it better!
Check out the Issues tab to see open issues.
You're welcome to start with those issues that are tagged with `Good First Issue`,
tackle other issues, or create your own issues.

## Getting started
Thank you for considering contributing to Tarteel-ML! Here are step-by-step instructions.

### Installing Dependencies

1. Before starting, you will need to install a few dependencies. We use the
[Anaconda Python distribution](https://www.anaconda.com/) for dependency management, and
our instructions assume you use it to. You can download it at this
[link](https://www.anaconda.com/download/).

2. Once you have installed Anaconda and verified it is being used, clone and `cd` into the
Tarteel-ML repository


3. We highly recommend creating a specific env for this repo by running the following commands to install all dependencies.
   ```commandline
   conda env create --file requirements.txt
   ```

4. After this, activate the `tarteel` environment.
   ```commandline
   source activate tarteel-ml
   ```

You should now be ready to contribute to Tarteel-ML! When you are done, remember to deactivate the environment.
```commandline
source deactivate tarteel-ml
```


### Adding New Dependencies
Use the `pip install <library-name>` command to add any new dependencies and ensure that the environment
resolves. Pull requests with new dependencies that break the existing environment for others will be
rejected.

After adding your new dependencies in Anaconda, add it (with the version number) in `requirements.txt`.

### Conventions

#### Pull Requests
- Whenever submitting a new PR, create a new branch named using the convention `<username>/<issue>`.

- Make sure to include descriptive and clear commit messages, while also referencing any issues your
PR addresses.

- Your pull request will be reviewed by the maintainers of this repository, and upon
approval, will be merged into the master branch.

#### Documentation
Tarteel-ML requires that your code be well-commented and that you explain clearly what your changes
are doing. Insufficiently commented code may be rejected if it is unclear to reviewers. Take a look
at existing code to see what is expected!

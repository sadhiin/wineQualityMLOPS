# Wine Quality Prediction ML

This repository contains a machine learning project for predicting the quality of wines. It uses a dataset of wine attributes to build a predictive model and offers a set of instructions to set up the environment, install dependencies, and get started with the project.

## Getting Started

Follow the steps below to set up the environment and get this project up and running on your local machine.

### Prerequisites

- [conda](https://docs.conda.io/en/latest/miniconda.html)
- [git](https://git-scm.com/)

### Environment Setup

1. Create a conda environment for this project:

```bash
conda create -n winequality_mlops python=3.10 -y
```

2. Activate the environment:

```bash
conda activate winequality_mlops
```

3. Create a `requirements.txt` file:

```bash
touch requirements.txt
```

4. Install the required packages:

```bash
pip install -r requirements.txt
```

### Initializing Git and DVC

This project uses Git and [DVC (Data Version Control)](https://dvc.org/) for managing code and data. Follow these steps to initialize them:

1. Initialize a Git repository:

```bash
git init
```

2. Initialize a DVC repository:

```bash
dvc init
```

# DVC + Testing(automated)
### Adding Data

The dataset used for this project is stored in the `data_given` directory. To add the dataset using DVC, run the following command:

```bash
dvc add data_given/winequality.csv
```

### Commit Changes

After adding the data and initializing Git and DVC, commit the changes to the repository:

```bash
git add .
git commit -m "Initial commit"
```

## Run trianing
```bash
python src/train_and_evaluate.py --config=params.yaml
```

## Result monitor
### Show different metrics
```bash
dvc metrics show
```
### Check matrics differents
```bash
dvc metrics diff
```
## Tox
Tox command
```bash
tox
```
for rebuilding
```bash
tox -r
```

## Testing and Setup
pytest command
```bash
pytest -v
```

setup commands -
```bash
pip install -e .
```

Build your own package commands-
```bash
python setup.py sdist bdist_wheel
```

# MLFlow + DVC


## License

This project is licensed under the MIT License - see the [LICENSE](#) file for details.

## Acknowledgments

- This project is based on a dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).
- Special thanks to the contributors and maintainers of the libraries and tools used in this project.

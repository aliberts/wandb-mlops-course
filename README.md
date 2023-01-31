# Weights & Biases MLOPs Course Material
<p align="center">
	<a href="https://www.python.org/downloads/release/python-3100/">
		<img src="https://img.shields.io/badge/Python-3.10-blue"
			 alt="Python Version">
	</a>
	<a href="https://github.com/psf/black">
		<img src="https://img.shields.io/badge/Code%20style-Black-000000.svg"
			 alt="Code Style">
	</a>
	<a href="https://wandb.ai/aliberts/mlops-course-001">
		<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28-gray.svg"
			 alt="Weights & Biases"
			 height="20">
	</a>
</p>


This repo contains all the material I used while attending [Weights & Biases' MLOPs course](https://www.wandb.courses/courses/effective-mlops-model-development).

The dataset used for this course is the BDD simple 1k, which is a small subset of the original [BDD 100k dataset](https://www.bdd100k.com/)


## Dashboard
Checkout the project on [Weights & Biases](https://wandb.ai/aliberts/mlops-course-001).

## Installation

#### Step 1
Ensure your gpu driver & cuda are properly setup for pytorch to use it (the name of your device should appear):
```bash
nvidia-smi
```

#### Step 2
If you don't have it already, install [poetry](https://python-poetry.org/):
```bash
make setup-poetry
```

#### Step 3
Setup the environment:
```bash
conda create --yes --name wandb python=3.10
conda activate wandb
git clone git@github.com:aliberts/wandb-mlops-course.git
cd wandb-mlops-course
poetry install
```

#### Step 4
Download the `bdd1k` dataset:
```bash
make dataset
```
This will download and extract the archive into `artifacts/` and then delete the original `.zip` archive.
You can also download it manually [here](https://storage.googleapis.com/wandb_course/bdd_simple_1k.zip). If you do, you'll need to update the `dataset.dir` option in [src/config.py](src/config.py)

#### Optional
Make your commands shorter with this `alias`:
```bash
alias py='poetry run python'
```

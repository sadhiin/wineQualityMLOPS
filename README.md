Create env

```bash
conda create -n winequality_mlops python=3.10 -y
```

Activate Environment
```bash
conda activate winequality_mlops
```

Create `requirements.txt` file
```bash
touch requirements.txt
```

Create `README.md` file
```bash
touch README.md
```
Install requriments
```bash
pip install -r requirmnets.txt
```

```bash
git init && dvc init

dvc add ata_given/winequality.csv

git add . && git commit -m "first commit"
```


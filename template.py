import os

dirs = [
    os.path.join('data','raw'),
    os.path.join('data','processed'),
    os.path.join('data','raw'),
    'notebooks',
    'data_given',
    'saved_models',
    'src'
]
for dir_ in dirs:
    os.makedirs(dir_, exist_ok=True)
    with open(os.path.join(dir_, '.gitignore'), 'w') as f:
        pass

files = [
    'dvc.yaml',
    'params.yaml',
    '.gitignore',
    os.path.join('src', '__init__.py'),
    os.path.join('src', 'get_data.py')
]

for file_ in files:
    with open(file_, 'w') as f:
        pass
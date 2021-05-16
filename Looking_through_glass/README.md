# Looking through glass
## paper_link : https://arxiv.org/abs/2101.01508

Description : This repository has codes both for running on big dataset (as used in the paper) and sample dataset (as provided in the repository). Please execute the relevant files to get the results. A stepwise procedure to setup the virtual enviornment has been described below:

Run the following commands in the current folder to generate results from sample data

### Create virtual environment
```sh
python -m venv glass
```

### Activate virtual environment
```sh
source glass/bin/activate
```

### Install required modules
```sh
pip install -r requirements.txt
```
### For downloading ChemDataExtractor files and models, run
```sh
cde data download
```
### Run the following command for results
```sh
python panini_test.py
```
### Run the following command for sample results
```sh
python sample_results.py
```

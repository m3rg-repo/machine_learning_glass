# Looking through glass: Knowledge discovery from materials science literature using natural language processing
## paper_link : https://www.sciencedirect.com/science/article/pii/S2666389921001239

Description : This repository has codes both for running on big dataset (as used in the paper) and sample dataset (as provided in the repository). Please execute the relevant files to get the results. A stepwise procedure to setup the virtual environment has been described below:

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

------------
### Cite as:
If you find this useful in your research, please consider citing:
```
@article{VENUGOPAL2021100290,
title = {Looking through glass: Knowledge discovery from materials science literature using natural language processing},
journal = {Patterns},
volume = {2},
number = {7},
pages = {100290},
year = {2021},
issn = {2666-3899},
doi = {https://doi.org/10.1016/j.patter.2021.100290},
url = {https://www.sciencedirect.com/science/article/pii/S2666389921001239},
author = {Vineeth Venugopal and Sourav Sahoo and Mohd Zaki and Manish Agarwal and Nitya Nand Gosvami and N. M. Anoop Krishnan}
}
```

# PASNet
### PASNet is a pathway-based sparse deep neural network.
# Get Started
## Example Datasets
To get started, you need to download example datasets from URLs as below:

[Train data](http://datax.kennesaw.edu/PASNet/train.csv) 

[Validation data](http://datax.kennesaw.edu/PASNet/validation.csv)

[Pathway Mask data](http://datax.kennesaw.edu/PASNet/pathway_mask.csv)

## Empirical Search for Hyperparameters 
Run_EmpiricalSearch.py: to find the optimal pair of hyperparmaters for PASNet before performing cross validation. PASNet is trained with the inputs from train.csv. Hyperparameters are optimized by emipirical search with validation.csv.
## 5-fold Cross Validation
Run.py: to train and evaluate the model performance based on 10 times 5-fold cross validation.

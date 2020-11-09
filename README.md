# PASNet
### PASNet is a pathway-based sparse deep neural network. The [PASNet](https://doi.org/10.1186/s12859-018-2500-z) model has the following contributions:
* ### Interpretable neural network on the biological pathway level
* ### Training the neural netowrk with high-dimension, low-sample size data 
* ### Automatically optimizing the sparse neural network
* ### Better classification performance

### Reference
```
@article{hao2018pasnet:,
  author = {Hao, Jie and Kim, Youngsoon and Kim, Tae-Kyung and Kang, Mingon},
  year = {2018},
  title = {PASNet: pathway-associated sparse deep neural network for prognosis prediction from high-throughput data},
  journal = {BMC Bioinformatics},
  doi = {10.1186/s12859-018-2500-z},
  volume = {19},
  month = {12},
  pages = {510},
  number = {1},
  url = {https://doi.org/10.1186/s12859-018-2500-z},
}
```
# Get Started
## Example Datasets
To get started, you need to download example datasets from URLs as below:

[Train data](http://dataxlab.org/PASNet/train.csv) 

[Validation data](http://dataxlab.org/PASNet/validation.csv)

[Pathway Mask data](http://dataxlab.org/PASNet/pathway_mask.csv)

## Empirical Search for Hyperparameters 
Run_EmpiricalSearch.py: to find the optimal pair of hyperparmaters for PASNet before performing cross validation. PASNet is trained with the inputs from train.csv. Hyperparameters are optimized by emipirical search with validation.csv.
## 5-fold Cross Validation
Run.py: to train and evaluate the model performance based on 10 times 5-fold cross validation.

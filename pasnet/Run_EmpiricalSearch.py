from DataLoader import load_data, load_pathway
from Train import trainPASNet
import torch
import numpy as np



''' PASNet Settings'''
In_Nodes = 4359 ###number of genes
Pathway_Nodes = 574 ###number of pathways
Hidden_Nodes = 100 ###number of hidden nodes
Out_Nodes = 2 ###one is for LTS, and the other is for non-LTS
''' Initial Settings for Empirical Search '''
Learning_Rates = [0.05, 0.01, 0.007, 0.005, 0.001, 0.0007, 0.0005, 0.0001]
L2_Lambdas = [3e-4, 5e-4, 7e-4, 1e-3, 3e-3, 5e-3]
Dropout_Rates = [0.8, 0.7] ###sub-network setup
nEpochs = 5000 ###for empirical search

''' load data and pathway '''
dtype = torch.FloatTensor
pathway_mask = load_pathway("data/gbm_binary_pathway_mask_reactome_574.csv", dtype)
###loaded data were split for empirical search only
x_train, y_train = load_data("data/std_train.csv", dtype)
x_valid, y_valid = load_data("data/std_valid.csv", dtype)

opt_l2 = 0
opt_lr = 0
opt_loss = torch.Tensor([float("Inf")])
###if gpu is being used
if torch.cuda.is_available():
	opt_loss = opt_loss.cuda()
###

##grid search the optimal hyperparameters using train and validation data
for lr in Learning_Rates:
	for l2 in L2_Lambdas:
		pred_tr, pred_val, loss_tr, loss_val = trainPASNet(x_train, y_train, x_valid, y_valid, pathway_mask, \
																In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, \
																lr, l2, nEpochs, Dropout_Rates, optimizer = "Adam")
		if loss_val < opt_loss:
			opt_l2 = l2
			opt_lr = lr
			opt_loss = loss_val
		
		print("L2: ", l2, "LR: ", lr, "Loss in Validation: ", loss_val)
		print("Optimal L2: ", opt_l2, "Optimal LR: ", opt_lr)




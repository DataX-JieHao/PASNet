from DataLoader import load_data, load_pathway
from Train import trainPASNet

import torch
import numpy as np


dtype = torch.FloatTensor
''' Net Settings'''
In_Nodes = 4359 ###number of genes
Pathway_Nodes = 574 ###number of pathways
Hidden_Nodes = 100 ###number of hidden nodes
Out_Nodes = 2 ###number of hidden nodes in the last hidden layer
''' Initialize '''
nEpochs = 10000 ###for training
Dropout_Rates = [0.8, 0.7] ###sub-network setup
''' load data and pathway '''
pathway_mask = load_pathway("data/gbm_binary_pathway_mask_reactome_574.csv", dtype)


opt_lr = 1e-4
opt_l2 = 3e-4
test_auc = []
for replicate in range(10):
	for fold in range(5):
		print("replicate: ", replicate, "fold: ", fold)
		x_train, y_train = load_data("data/std_train_"+str(replicate)+"_"+str(fold)+".csv", dtype)
		x_test, y_test = load_data("data/std_test_"+str(replicate)+"_"+str(fold)+".csv", dtype)
		loss_train, loss_test, auc_tr, auc_te = trainPASNet(x_train, y_train, x_test, y_test, pathway_mask, \
															In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, \
															opt_lr, opt_l2, nEpochs, Dropout_Rates, optimizer = "Adam")
		print ("Loss in Train: ", loss_train, "Loss in Test: ", loss_test, "AUC in Train: ", auc_tr, "AUC in Test: ", auc_te)
		test_auc.append(auc_te)

np.savetxt("pasnet_auc.txt", test_auc, delimiter = ",")
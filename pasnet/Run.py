from DataLoader import load_data, load_pathway
from Train import trainPASNet
from EvalFunc import auc, f1

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

N = 10 # number of repeated times
K = 5 # number of folds
opt_lr = 1e-4
opt_l2 = 3e-4
test_auc = []
test_f1 = []
for replicate in range(N):
	for fold in range(K):
		print("replicate: ", replicate, "fold: ", fold)
		x_train, y_train = load_data("data/std_train_"+str(replicate)+"_"+str(fold)+".csv", dtype)
		x_test, y_test = load_data("data/std_test_"+str(replicate)+"_"+str(fold)+".csv", dtype)
		pred_train, pred_test, loss_train, loss_test = trainPASNet(x_train, y_train, x_test, y_test, pathway_mask, \
															In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, \
															opt_lr, opt_l2, nEpochs, Dropout_Rates, optimizer = "Adam")
		###if gpu is being used, transferring back to cpu
		if torch.cuda.is_available():
			pred_test = pred_test.cpu().detach()
		###
		np.savetxt("PASNet_pred_"+str(replicate)+"_"+str(fold)+".txt", pred_test.numpy(), delimiter = ",")
		auc_te = auc(y_test, pred_test)
		f1_te = f1(y_test, pred_test)
		print("AUC in Test: ", auc_te, "F1 in Test: ", f1_te)
		test_auc.append(auc_te)
		test_f1.append(f1_te)
		
np.savetxt("PASNet_AUC.txt", test_auc, delimiter = ",")
np.savetxt("PASNet_F1.txt", test_f1, delimiter = ",")

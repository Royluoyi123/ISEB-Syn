import numpy as np
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random
random.seed(0)
from prettytable import PrettyTable

import os

from argument import config2string
from utils import create_batch_mask, get_roc_score
from data import Dataclass
from torch_geometric.data import DataLoader


class embedder:

    def __init__(self, args, train_df, test_df, gene_exp_dict, repeat, fold):
        self.args = args
        self.config_str = config2string(args)
        print("\n[Config] {}\n".format(self.config_str))
        


        # Model Checkpoint Path


        # Select GPU device
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)

        self.train_dataset = Dataclass(train_df, gene_exp_dict)
        self.test_dataset = Dataclass(test_df, gene_exp_dict)

        self.train_loader = DataLoader(self.train_dataset, batch_size = args.batch_size, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size = 128)
        self.explain_loader = DataLoader(self.train_dataset, batch_size = 1)

        self.is_early_stop = False

        self.best_train_auc = -1.0
        self.best_test_auc = -1.0


        self.test_auc = -1.0
        self.test_aupr = -1.0
        self.test_acc = -1.0
        self.test_kappa = -1.0

    def evaluate(self, epoch, final = False):
        
        train_outputs, train_labels = [], []
        test_outputs, test_labels = [], []

        mol_indexs, outputs = [], []

        for bc, samples in enumerate(self.train_loader):

            masks = create_batch_mask(samples)
            output,_,_,_,_,_ = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device), samples[2].to(self.device)], bottle=True)
            
            train_outputs.append(output.reshape(-1).detach().cpu().numpy())
            train_labels.append(samples[3].reshape(-1).detach().cpu().numpy())
        
        train_outputs = np.hstack(train_outputs)
        train_labels = np.hstack(train_labels)

        self.train_auc_score, train_aupr_score, train_acc_score, self.train_kappa_score , self.train_bacc_score, self.train_f1_score, self.train_mcc_score= get_roc_score(train_outputs, train_labels)


        train_res = PrettyTable()
        train_res.field_names = ["epoch", "train_AUC","train_AUPR","train_ACC","train_KAPPA","train_bacc","train_f1","train_mcc"]
        train_res.add_row(
            [epoch,self.train_auc_score,train_aupr_score,train_acc_score,self.train_kappa_score,self.train_bacc_score,self.train_f1_score,self.train_mcc_score]
            )
        print(train_res)



        for bc, samples in enumerate(self.test_loader):
                
            masks = create_batch_mask(samples)
            output,_,_,_,_,_= self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device), samples[2].to(self.device)],bottle= True)
            
            test_outputs.append(output.reshape(-1).detach().cpu().numpy())
            test_labels.append(samples[3].reshape(-1).detach().cpu().numpy())

        test_outputs = np.hstack(test_outputs)
        test_labels = np.hstack(test_labels)

        

        test_auc_score, test_aupr_score, test_acc_score, test_kappa_score, test_bacc_score, test_f1_score , test_mcc_score = get_roc_score(test_outputs, test_labels)
        test_res = PrettyTable()
        test_res.field_names = ["epoch", "test_AUC","test_AUPR","test_ACC","test_KAPPA","test_BACC","test_F1","test_MCC"]
        test_res.add_row(
            [epoch,test_auc_score,test_aupr_score,test_acc_score,test_kappa_score,test_bacc_score,test_f1_score,test_mcc_score]
            )
        print(test_res)


        # Save ROC score
        if test_auc_score > self.best_test_auc :

            self.patience = 0
            
            # Save train score
            self.best_test_auc = test_auc_score

            # Save test score
            self.test_auc = test_auc_score
            self.test_aupr = test_aupr_score
            self.test_acc = test_acc_score
            self.test_kappa = test_kappa_score
            self.test_bacc = test_bacc_score
            self.test_f1 = test_f1_score
            self.test_mcc = test_mcc_score
            self.test_outputs = test_outputs
            self.test_labels = test_labels

            # Save epoch
            self.best_auc_epoch = epoch

            if self.args.save_checkpoints == True:
                
                #checkpoint = {'mol_id': np.hstack(mol_indexs), 'output': np.hstack(outputs)}                
                check_dir =  'warm_start_bestepoch_visual_GDSC.pth'
                torch.save(self.model.state_dict(), check_dir)
        
        else:
            self.patience += 1

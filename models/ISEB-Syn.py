import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from torch_geometric.nn import Set2Set

from embedder import embedder
from layers import GINE
from utils import create_batch_mask

from torch_scatter import scatter_mean, scatter_add, scatter_std

import time

class ISEB-Syn_ModelTrainer(embedder):
    def __init__(self, args, train_df, valid_df, train_exp,  repeat, fold):
        embedder.__init__(self, args, train_df, valid_df, train_exp, repeat, fold)

        self.model = IGIB(args, device = self.device, tau = self.args.tau, num_step_message_passing = self.args.message_passing,EM=self.args.EM_NUM).to(self.device)
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=self.args.patience, mode='max', verbose=True)
        
    def train(self):        
        
        loss_function_BCE = nn.BCEWithLogitsLoss(reduction='none')
        
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            self.train_loss = 0
            preserve = 0

            start = time.time()
            for bc, samples in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                masks = create_batch_mask(samples)
                #outputs, _ = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device), samples[2].to(self.device)])
                #loss = loss_function_BCE(outputs, samples[3].reshape(-1, 1).to(self.device).float()).mean()

                # Information Bottleneck.
                outputs, KL_Loss_1,KL_Loss_2, cont_loss_1,cont_loss_2, preserve_rate = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device), samples[2].to(self.device)], bottle = True)
                loss = loss_function_BCE(outputs, samples[3].reshape(-1, 1).to(self.device).float()).mean()
                #print(cont_loss_1)
                #print(KL_Loss_1)
                loss += self.args.beta_1 * KL_Loss_1
                loss += self.args.beta_1 * cont_loss_1
                loss += self.args.beta_2 * KL_Loss_2
                loss += self.args.beta_2 * cont_loss_2
                #print(loss)
                loss.backward()
                self.optimizer.step()
                self.train_loss += loss
                #preserve += preserve_rate

            self.epoch_time = time.time() - start

            self.model.eval()
            self.evaluate(epoch)

            self.scheduler.step(self.train_auc_score)

            # Write Statistics


            # Early stopping
            print(self.patience)
            if self.patience > int(self.args.es / self.args.eval_freq):
                        break
        return self.test_auc, self.test_aupr, self.test_acc, self.test_kappa, self.test_bacc, self.test_f1, self.test_mcc
    
    def explain(self):
        import_score_druga_list = []
        import_score_drugb_list = []
        for bc, samples in enumerate(self.explain_loader):
            if samples[3].detach().cpu().numpy()[0] == 1:
                
                masks = create_batch_mask(samples)

                drugA = samples[0].to(self.device)
                drugB = samples[1].to(self.device)
                drug_A_len = masks[0].to(self.device)
                drug_B_len = masks[1].to(self.device)
                #cell_features = data[4]
                # node embeddings after interaction phase
                drug_A_features = self.model.gather(drugA) #公式2 节点个数,300 GIN得到的分子表征
                drug_B_features = self.model.gather(drugB) #公式3 节点个数,300

                drug_A_features = F.normalize(drug_A_features, dim = 1)
                drug_B_features = F.normalize(drug_B_features, dim = 1)

                len_map = torch.sparse.mm(drug_A_len.t(), drug_B_len) #指示哪些原子是属于同一个样本对 如果 len_map[i, j] == 1：意味着 药物A的第 $i$ 个原子 和 药物B的第 $j$ 个原子 属于同一个训练样本（Pair）
                interaction_map = torch.mm(drug_A_features, drug_B_features.t())
                ret_interaction_map = torch.clone(interaction_map)
                ret_interaction_map = interaction_map * len_map.to_dense()
                interaction_map = interaction_map * len_map.to_dense()
                drug_B_prime = torch.mm(interaction_map.t(), drug_A_features)
                drug_A_prime = torch.mm(interaction_map, drug_B_features)

                drug_A_features = torch.cat((drug_A_features, drug_A_prime), dim=1)
                drug_B_features = torch.cat((drug_B_features, drug_B_prime), dim=1)

                 #公式3 前一个是药物1 后一个是药物2
            #
            #output,_,_ = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device), samples[2].to(self.device)], test = True, causal=True)
                lambda_pos_A, importance_A = self.model.compress(drug_A_features)
                lambda_pos_B, importance_B = self.model.compress(drug_B_features)
                import_score_druga_list.append(importance_A.detach().cpu().numpy())
                import_score_drugb_list.append(importance_B.detach().cpu().numpy())

                

                del drugA, drugB, drug_A_features, drug_B_features
                del lambda_pos_A, importance_A, lambda_pos_B, importance_B
                torch.cuda.empty_cache()


        return import_score_druga_list, import_score_drugb_list
    
    def predict_tridrug(self):
        import_score_druga_list = []
        import_score_drugb_list = []
        for bc, samples in enumerate(self.explain_loader):
            if samples[3].detach().cpu().numpy()[0] == 1:
                masks = create_batch_mask(samples)

                drugA = samples[0].to(self.device)
                drugB = samples[1].to(self.device)
                self.model.drugA_len = masks[0].to(self.device)
                self.model.drugB_len = masks[1].to(self.device)
                #cell_features = data[4]
                # node embeddings after interaction phase
                _drugA_features = self.model.gather(drugA) #公式2 节点个数,300 GIN得到的分子表征
                _drugB_features = self.model.gather(drugB) #公式3 节点个数,300

                drugA_features, drugB_features = self.model.interaction(_drugA_features, _drugB_features) #公式3 前一个是药物1 后一个是药物2
            #
            #output,_,_ = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device), samples[2].to(self.device)], test = True, causal=True)
                lambda_pos_A, importance_A = self.model.compress(drugA_features)
                lambda_pos_B, importance_B = self.model.compress(drugB_features)
                import_score_druga_list.append(importance_A.detach().cpu().numpy())
                import_score_drugb_list.append(importance_B.detach().cpu().numpy())

                del drugA, drugB, _drugA_features, _drugB_features, drugA_features, drugB_features
                del lambda_pos_A, importance_A, lambda_pos_B, importance_B, self.model.drugA_len, self.model.drugB_len
                torch.cuda.empty_cache()
                gc.collect()

        return import_score_druga_list, import_score_drugb_list
    
    

class IGIB(nn.Module):

    def __init__(self,
                args,
                device,
                node_input_dim=133,
                edge_input_dim=14,
                node_hidden_dim=300,
                edge_hidden_dim=300,
                num_step_message_passing=3,
                tau = 1.0,
                interaction='dot',
                num_step_set2_set=2,
                num_layer_set2set=1,
                EM = 3
                ):
        super(IGIB, self).__init__()

        self.device = device
        self.tau = tau
        self.EM = EM
        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing
        self.interaction = interaction
        self.drop_out = args.dropout

        self.gather = GINE(self.node_input_dim, self.edge_input_dim, 
                            self.node_hidden_dim, self.num_step_message_passing,
                            )

        self.predictor = nn.Sequential(
            nn.Linear(8 * self.node_hidden_dim + 640, 4* self.node_hidden_dim),
            #nn.Linear(18782, 4* self.node_hidden_dim),
            nn.BatchNorm1d(4* self.node_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(4 * self.node_hidden_dim, 2 * self.node_hidden_dim),
            nn.BatchNorm1d(2 * self.node_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(2 * self.node_hidden_dim, 1)
            )

        self.compressor = nn.Sequential(
            nn.Linear(self.node_hidden_dim*2, self.node_hidden_dim),
            nn.BatchNorm1d(self.node_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.node_hidden_dim, 1)
            )
        
        self.drug_B_predictor = nn.Linear(4 * self.node_hidden_dim, 4 * self.node_hidden_dim)
        
        self.mse_loss = torch.nn.MSELoss()

        self.num_step_set2set = num_step_set2_set
        self.num_layer_set2set = num_layer_set2set
        self.set2set_drug_A = Set2Set(2 * node_hidden_dim, self.num_step_set2set)
        self.set2set_drug_B = Set2Set(2 * node_hidden_dim, self.num_step_set2set)

        self.init_model()
    
    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def compress(self, drug_A_features):
        
        p = self.compressor(drug_A_features)
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(self.device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()

        return gate_inputs, p
    
    def forward(self, data, bottle = False):

        drug_A = data[0]
        drug_B = data[1]
        drug_A_len = data[2]
        drug_B_len = data[3]
        cell_features = data[4]

        # node embeddings after interaction phase
        drug_A_features = self.gather(drug_A)
        drug_B_features = self.gather(drug_B)

        # Add normalization
        self.drug_A_features = F.normalize(drug_A_features, dim = 1)
        self.drug_B_features = F.normalize(drug_B_features, dim = 1)
        #print(self.drug_A_features.size())
        #print(self.drug_B_features.size())

        # Interaction phase
        # Prediction phase
        #self.drug_A_features = torch.cat((self.drug_A_features, self.drug_A_prime), dim=1)
        #self.drug_B_features = torch.cat((self.drug_B_features, self.drug_B_prime), dim=1)
        # Interaction phase
        
        len_map = torch.sparse.mm(drug_A_len.t(), drug_B_len) #指示哪些原子是属于同一个样本对 如果 len_map[i, j] == 1：意味着 药物A的第 $i$ 个原子 和 药物B的第 $j$ 个原子 属于同一个训练样本（Pair）
        interaction_map = torch.mm(self.drug_A_features, self.drug_B_features.t())
        ret_interaction_map = torch.clone(interaction_map)
        ret_interaction_map = interaction_map * len_map.to_dense()
        interaction_map = interaction_map * len_map.to_dense()
        self.drug_B_prime = torch.mm(interaction_map.t(), self.drug_A_features)
        self.drug_A_prime = torch.mm(interaction_map, self.drug_B_features)
        # Prediction phase
        self.drug_A_features = torch.cat((self.drug_A_features, self.drug_A_prime), dim=1)
        self.drug_B_features = torch.cat((self.drug_B_features, self.drug_B_prime), dim=1)
        drug_A_0 =self.drug_A_features.clone() #drug_A 特征
        drug_B_0 =self.drug_B_features.clone()


        if bottle:
            EM_num = self.EM #20超参数
            for i in range(EM_num):
                if i ==0:
                    self.drug_A_features =drug_A_0.clone()
                    self.drug_B_features=drug_B_0.clone() #初始特征
                else:
                    self.drug_A_features=drug_A_noisy_node_feature.clone()
                    self.drug_B_features=drug_B_noisy_node_feature.clone()
                #Ebu 
                
                interaction_map = torch.mm(self.drug_A_features, self.drug_B_features.t()) #公式6 前半部分
                #ret_interaction_map = torch.clone(interaction_map)
                #ret_interaction_map = interaction_map * len_map.to_dense()
                interaction_map = interaction_map * len_map.to_dense() #原子之间关系矩阵
                #drug_B_prime = torch.mm(interaction_map.t(), self.drug_A_features)
                drug_A_prime = torch.mm(interaction_map, self.drug_B_features)

                lambda_pos, p = self.compress(drug_A_prime) #公式6 后半部分
                lambda_pos = lambda_pos.reshape(-1, 1)
                lambda_neg = 1 - lambda_pos 

                # Get Stats
                preserve_rate = (torch.sigmoid(p) > 0.5).float().mean()

                static_drug_A_feature = self.drug_A_features.clone().detach()
                node_feature_mean = scatter_mean(static_drug_A_feature, drug_A.batch, dim = 0)[drug_A.batch]
                node_feature_std = scatter_std(static_drug_A_feature, drug_A.batch, dim = 0)[drug_A.batch]
                
                noisy_node_feature_mean = lambda_pos * self.drug_A_features + lambda_neg * node_feature_mean
                noisy_node_feature_std = lambda_neg * node_feature_std

                drug_A_noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std #得到药物A的表征

                #MAX

                interaction_map = torch.mm(drug_A_noisy_node_feature, self.drug_B_features.t())
                #ret_interaction_map = torch.clone(interaction_map)
                #ret_interaction_map = interaction_map * len_map.to_dense()
                interaction_map = interaction_map * len_map.to_dense()
                drug_B_prime = torch.mm(interaction_map.t(), drug_A_noisy_node_feature)
                #drug_A_prime = torch.mm(interaction_map, self.drug_B_features)

                lambda_pos, p = self.compress(drug_B_prime)
                lambda_pos = lambda_pos.reshape(-1, 1)
                lambda_neg = 1 - lambda_pos

                # Get Stats
                preserve_rate = (torch.sigmoid(p) > 0.5).float().mean()

                static_drug_B_feature = self.drug_B_features.clone().detach()
                drug_B_node_feature_mean = scatter_mean(static_drug_B_feature, drug_B.batch, dim = 0)[drug_B.batch]
                drug_B_node_feature_std = scatter_std(static_drug_B_feature, drug_B.batch, dim = 0)[drug_B.batch]
                
                drug_B_noisy_node_feature_mean = lambda_pos * self.drug_B_features + lambda_neg * drug_B_node_feature_mean
                drug_B_noisy_node_feature_std = lambda_neg * drug_B_node_feature_std

                drug_B_noisy_node_feature = drug_B_noisy_node_feature_mean + torch.rand_like(drug_B_noisy_node_feature_mean) * drug_B_noisy_node_feature_std



            noisy_drug_A_subgraphs = self.set2set_drug_A(drug_A_noisy_node_feature, drug_A.batch)
            drug_B_noisy_drug_A_subgraphs = self.set2set_drug_A(drug_B_noisy_node_feature, drug_B.batch)

            epsilon = 1e-7

            KL_tensor_1 = 0.5 * scatter_add(((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2).mean(dim = 1), drug_A.batch).reshape(-1, 1) + \
                        scatter_add((((noisy_node_feature_mean - node_feature_mean)/(node_feature_std + epsilon)) ** 2), drug_A.batch, dim = 0)
            KL_Loss_1 = torch.mean(KL_tensor_1)

            KL_tensor_2 = 0.5 * scatter_add(((drug_B_noisy_node_feature_std ** 2) / (drug_B_node_feature_std + epsilon) ** 2).mean(dim = 1), drug_B.batch).reshape(-1, 1) + \
                        scatter_add((((drug_B_noisy_node_feature_mean - drug_B_node_feature_mean)/(drug_B_node_feature_std + epsilon)) ** 2), drug_B.batch, dim = 0)
            KL_Loss_2 = torch.mean(KL_tensor_2)
            
            # Contrastive Loss
            #self.drug_B_features_s2s = self.set2set_drug_B(self.drug_B_features, drug_B.batch)
            cont_loss_1 = self.contrastive_loss(noisy_drug_A_subgraphs, drug_B_noisy_drug_A_subgraphs, self.tau)
            cont_loss_2 = self.contrastive_loss(drug_B_noisy_drug_A_subgraphs,noisy_drug_A_subgraphs, self.tau)

            # Prediction Y
            final_features = torch.cat((noisy_drug_A_subgraphs, drug_B_noisy_drug_A_subgraphs, cell_features), 1)
            #print(final_features.size())
            predictions = self.predictor(final_features)

            return predictions, KL_Loss_1,KL_Loss_2, cont_loss_1,cont_loss_2, preserve_rate
        
    
    def contrastive_loss(self, drug_A, drug_B, tau):

        batch_size, _ = drug_A.size()
        drug_A_abs = drug_A.norm(dim = 1)
        drug_B_abs = drug_B.norm(dim = 1)        

        sim_matrix = torch.einsum('ik,jk->ij', drug_A, drug_B) / torch.einsum('i,j->ij', drug_A_abs, drug_B_abs)
        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss
    

    def get_checkpoints(self):
        
        return self.drug_A_features_s2s, self.drug_B_features_s2s, self.importance

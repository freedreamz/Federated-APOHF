from automatic_prompt_engineer import ape, data
from experiments.data.instruction_induction.load_data import load_data
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator, exec_evaluator
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModel
from automatic_prompt_engineer import evaluate, template, data

import re
import matplotlib.pyplot as plt

from transformers import LlamaForCausalLM
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from typing import List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from backpack import backpack, extend
from backpack.extensions import BatchGrad

import scipy as sp
import torch.optim as optim
import numpy as np
import random

tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}

def compute_metrics(y_true, y_pred):
    """
    评估模型性能，计算 MSE、RMSE、MAE、R² 决定系数 和 皮尔逊相关系数。
    
    参数:
    y_true (torch.Tensor): 真实的奖励值。
    y_pred (torch.Tensor): 预测的奖励值。
    
    返回:
    dict: 包含 MSE、RMSE、MAE、R² 决定系数 和 皮尔逊相关系数的字典。
    """
    # 计算均方误差 (MSE)
    mse = torch.mean((y_true - y_pred) ** 2)
    
    # 计算均方根误差 (RMSE)
    rmse = torch.sqrt(mse)
    
    # 计算平均绝对误差 (MAE)
    mae = torch.mean(torch.abs(y_true - y_pred))
    
    # 计算 R² 决定系数
    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    
    # 计算皮尔逊相关系数
    mean_true = torch.mean(y_true)
    mean_pred = torch.mean(y_pred)
    numerator = torch.sum((y_true - mean_true) * (y_pred - mean_pred))
    denominator = torch.sqrt(torch.sum((y_true - mean_true) ** 2) * torch.sum((y_pred - mean_pred) ** 2))
    pearson_corr = numerator / denominator
    
    # 返回结果字典
    return {
        "MSE": mse.item(),
        "RMSE": rmse.item(),
        "MAE": mae.item(),
        "R²": r2.item(),
        "Pearson Correlation": pearson_corr.item()
    }

class MLPRegression(nn.Module):

    def __init__(self, input_dim=86):
        super(MLPRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output

class CustomImageDataset(Dataset):
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def __len__(self):
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]


class ENN(nn.Module):
    def __init__(self, input_dim, hidden_size=32, depth=2, init_params=None):
        super(ENN, self).__init__()

        self.activate = nn.ReLU()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(input_dim, hidden_size))
        for i in range(depth-1):
            self.layer_list.append(nn.Linear(hidden_size, hidden_size))
        self.layer_list.append(nn.Linear(hidden_size, 1))
        
        self.layer_list_10 = nn.ModuleList()
        for i in range(10):
            new_module = nn.ModuleList()
            new_module.append(nn.Linear(input_dim, hidden_size))
            for j in range(depth-1):
                new_module.append(nn.Linear(hidden_size, hidden_size))
            new_module.append(nn.Linear(hidden_size, 1))
            self.layer_list_10.append(new_module)
        
        if init_params is None:
            ## use initialization
            for i in range(len(self.layer_list)):
                torch.nn.init.normal_(self.layer_list[i].weight, mean=0, std=1.0)
                torch.nn.init.normal_(self.layer_list[i].bias, mean=0, std=1.0)
        else:
            ### manually set the initialization vector
            for i in range(len(self.layer_list)):
                self.layer_list[i].weight.data = init_params[i*2]
                self.layer_list[i].bias.data = init_params[i*2+1]

        # copy the init params to the 10 networks
        for i in range(10):
            for j in range(len(self.layer_list)):
                self.layer_list_10[i][j].weight.data.copy_(self.layer_list[j].weight.data)
                self.layer_list_10[i][j].bias.data.copy_(self.layer_list[j].bias.data)
                
        # make the parameters in self.layer_list to be not trainable
        for param in self.layer_list.parameters():
            param.requires_grad = False
        
    def forward(self, x, idx):
        y = x
        for i in range(len(self.layer_list_10[idx])-1):
            y = self.activate(self.layer_list_10[idx][i](y))
        y = self.layer_list_10[idx][-1](y)
        return y

class DoubleTS:
    def __init__(self, input_dim, lamdba=1, nu=1, style='ucb', init_x=None, init_y=None, diagonalize=True):

        self.diagonalize = diagonalize
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.func = ENN(input_dim).to(**tkwargs)
        self.init_state_dict = deepcopy(self.func.state_dict())

        if init_x is not None:
            self.pair_embedding = init_x.to(**tkwargs)
        else:
            self.pair_embedding = None
        if init_y is not None:
            self.reward = init_y.to(**tkwargs).to(dtype=torch.int64)
        else:
            self.reward = None
        self.len = 0
        self.lamdba = lamdba

        self.nu = nu
        self.lamdba = lamdba
        self.style = style
        self.mean = None
        self.std = None


    def select(self, context, select_idx_history, prompt_domain_id=None, batch_size=300):
        context_size = context.shape[0]        
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)

        final_arms = []
        up_k = 5
        k_ = 0
        while len(final_arms) < 2:
            mu = []
            self.func.train()
            epi_idx = torch.randint(0, 10, (1,))
            for i in range(n_batchs):
                if i == n_batchs - 1:
                    context_batch = context[(i*batch_size):]
                else:
                    context_batch = context[(i*batch_size):((i+1)*batch_size)]

                mu_ = self.func(context_batch, epi_idx)

                mu.append(mu_.cpu())
            mu = torch.vstack(mu)

            # select the first one
            if prompt_domain_id is None:
                arm1 = torch.argmax(mu.view(-1))
            else:
                arm1_ = torch.argmax(mu.view(-1)[prompt_domain_id])
                prompt_domain_id_ = torch.tensor(prompt_domain_id)
                arm1 = prompt_domain_id_[arm1_]
            
            if arm1 not in final_arms:
                final_arms.append(arm1)
            else:
                k_ += 1
            if k_ > up_k:
                if prompt_domain_id is None:
                    random_arm = torch.randint(0, context_size, (2,))
                else:
                    prompt_domain_id_ = torch.tensor(prompt_domain_id)
                    random_arm = torch.randint(0, len(prompt_domain_id), (2,))
                    random_arm = prompt_domain_id_[random_arm]
                if random_arm[0] not in final_arms:
                    final_arms.append(random_arm[0])
                else:
                    final_arms.append(random_arm[1])
                break
        return final_arms[0], final_arms[1]


    def find_best(self, context, select_idx_history, all_domain=False, batch_size=300):     
        context_size = context.shape[0]        
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
        mu = []
        self.func.eval()
        for i in range(n_batchs):
            epi_idx = torch.randint(0, 10, (1,))
            if i == n_batchs - 1:
                context_batch = context[(i*batch_size):]
            else:
                context_batch = context[(i*batch_size):((i+1)*batch_size)]

            mu_ = self.func(context_batch, epi_idx)
            mu.append(mu_.cpu())
        mu = torch.vstack(mu)

        # select the first one
        if all_domain:
            arm = torch.argmax(mu.view(-1))
        else:
            all_queried = torch.tensor(select_idx_history).view(-1)
            arm_ = torch.argmax(mu.view(-1)[all_queried])
            arm = all_queried[arm_]
        
        return arm


    def train(self, context=None, reward=None, local_training_iter=30):
        if self.init_state_dict is not None:
            self.func.load_state_dict(deepcopy(self.init_state_dict))
        if context is not None:
            self.pair_embedding = torch.cat((self.pair_embedding, context.reshape(2, 1, -1).to(**tkwargs)), dim=1)
            self.reward = torch.cat((self.reward, torch.tensor([reward]).reshape(-1).to(**tkwargs).to(dtype=torch.int64)))

        self.len = self.pair_embedding.shape[1]
        optimizer = torch.optim.Adam(self.func.parameters(), lr=1e-3)
        batch_size = 32
        if self.len < batch_size:
            lamdba_ = self.lamdba
        else:
            lamdba_ = self.lamdba * batch_size / (self.len)
        self.func.train()
        reward_ = 1 - self.reward.reshape(-1)
        for _ in range(local_training_iter):
            if self.len // batch_size == 0:
                selected_idx = torch.arange(0, self.len)
                epi_idx = torch.randint(0, 10, (1,))
                self.func.zero_grad()
                optimizer.zero_grad()
                side_1 = self.pair_embedding[0, selected_idx, :].reshape(len(selected_idx), -1)
                side_2 = self.pair_embedding[1, selected_idx, :].reshape(len(selected_idx), -1)
                pred_1 = self.func(side_1, epi_idx)
                pred_2 = self.func(side_2, epi_idx)
                ce_ = torch.mean(-(1-reward_[selected_idx].to(dtype=torch.float32)) * pred_1.reshape(-1) - reward_[selected_idx].to(dtype=torch.float32) * pred_2.reshape(-1) + torch.log(torch.exp(pred_1.reshape(-1)) + torch.exp(pred_2.reshape(-1))))
                l2_reg_term = 0
                for param1, param2 in zip(self.func.layer_list_10[epi_idx], self.func.layer_list):
                    l2_reg_term += torch.sum((param1.weight - param2.weight) ** 2) + torch.sum((param1.bias - param2.bias) ** 2)
                loss = ce_ + lamdba_ * l2_reg_term
                loss.backward()
                optimizer.step()
            else:
                for batch_id in range((self.len // batch_size)):
                    if batch_id == (self.len // batch_size) - 1 and self.len % batch_size != 0:
                        selected_idx = torch.arange(batch_id*batch_size, self.len)
                    else:
                        selected_idx = torch.arange(batch_id*batch_size, (batch_id+1)*batch_size)
                    epi_idx = torch.randint(0, 10, (1,))
                    self.func.zero_grad()
                    optimizer.zero_grad()
                    side_1 = self.pair_embedding[0, selected_idx, :].reshape(len(selected_idx), -1)
                    side_2 = self.pair_embedding[1, selected_idx, :].reshape(len(selected_idx), -1)
                    pred_1 = self.func(side_1, epi_idx)
                    pred_2 = self.func(side_2, epi_idx)
                    ce_ = torch.mean(-(1-reward_[selected_idx].to(dtype=torch.float32)) * pred_1.reshape(-1) - reward_[selected_idx].to(dtype=torch.float32) * pred_2.reshape(-1) + torch.log(torch.exp(pred_1.reshape(-1)) + torch.exp(pred_2.reshape(-1))))
                    l2_reg_term = 0
                    for param1, param2 in zip(self.func.layer_list_10[epi_idx], self.func.layer_list):
                        l2_reg_term += torch.sum((param1.weight - param2.weight) ** 2) + torch.sum((param1.bias - param2.bias) ** 2)
                    loss = ce_ + lamdba_ * l2_reg_term
                    loss.backward()
                    optimizer.step()
            
        print("Training Loss : ", loss.item())
        print("CE Loss : ", ce_.item())
        

        return self.func.state_dict()


class Network(nn.Module):
    def __init__(self, input_dim, hidden_size=32, depth=2, init_params=None):
        super(Network, self).__init__()

        self.activate = nn.ReLU()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(input_dim, hidden_size))
        for _ in range(depth - 1):
            self.layer_list.append(nn.Linear(hidden_size, hidden_size))
        self.layer_list.append(nn.Linear(hidden_size, 1))

        if init_params is None:
            ## 使用 Kaiming 初始化
            for layer in self.layer_list:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.kaiming_normal_(layer.weight)
                    torch.nn.init.constant_(layer.bias,0)  # bias 设为 0
        else:
            ## 手动设置初始化参数
            for i, layer in enumerate(self.layer_list):
                layer.weight.data = init_params[i * 2]
                layer.bias.data = init_params[i * 2 + 1]

    def forward(self, x):
        y = x
        for i in range(len(self.layer_list) - 1):
            y = self.activate(self.layer_list[i](y))
        y = self.layer_list[-1](y)
        return y
    
# class Network(nn.Module):
#     def __init__(self, input_dim, init_params=None):
#         super().__init__()

#         self.layers = nn.ModuleList([
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         ])

#         if init_params is None:
#             # 初始化权重（使用 kaiming 初始化）
#             for layer in self.layers:
#                 if isinstance(layer, nn.Linear):
#                     nn.init.kaiming_normal_(layer.weight)
#                     nn.init.constant_(layer.bias, 0)
#         else:
#             # 使用传入的 init_params 初始化
#             param_idx = 0
#             for layer in self.layers:
#                 if isinstance(layer, nn.Linear):
#                     layer.weight.data = init_params[param_idx]
#                     layer.bias.data = init_params[param_idx + 1]
#                     param_idx += 2

#     def forward(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x  
    

class NeuralDBDiag:
    def __init__(self, local_training_iter, input_dim, lam=0.1, nu=0.1, style='ucb', init_x=None, init_y=None, diag=True, N=10, new_iter= 100, lr=0.03, weight_decay=1/768):
        
        self.N = N 
        self.input_dim = input_dim
        self.diag = diag  
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.func = extend(Network(input_dim).to(**tkwargs))
        self.init_state_dict = deepcopy(self.func.state_dict())

        self.select_history = [deepcopy(init_x[_]) for _ in range(self.N)]
        # self.select_history = [[int(value) for value in x] for x in self.select_history]
        self.reward_list = [deepcopy(init_y[_]) for _ in range(self.N)]
        self.local_training_iter = local_training_iter

        self.lam = lam  # Changed from 'lamdba' to 'lam'
        self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
        
        self.new_iter = new_iter
        self.nu = nu
        self.nu_2 = nu
        self.style = style
        self.mean = None
        self.std = None
       
        # Initialize as class attributes using self
        # self.save_interval = 50  # save a log file after every "save_interval" iteration
        # self.max_iter = 2000 + self.save_interval
        self.max_iter = 500

        self.midpoint1 = 0
        self.midpoint2 = 100
        self.growing_seq = np.arange(self.midpoint2) / self.midpoint2
        self.alpha_ts = np.append(np.zeros(self.midpoint1), self.growing_seq)
        self.alpha_ts = np.append(self.alpha_ts, np.ones(self.max_iter - len(self.alpha_ts) + 5))

        
     
        # number of agents
        self.t_last = 0
        
        self.stop_training_after_iter = 1500

        self.flag_not_Less_Comm = False  # Set as False by default; if don't want to run FN-UCB (Less Comm.), set this flag to True

        
        self.D = 0

        # self.run_list = np.arange(0, 3)

      
        self.W_new_list = []
        self.B_new_list = []
        self.V_local_list = []
        self.communication_flag = np.zeros(self.N)

        self.W_sync = None
        self.B_sync = None
        self.regrets = []
        self.state_dict_list = [[] for _ in range(self.N)]

        # Initialize synchronization variables
        self.V_last = None
        self.V_t_i_bar = None
        self.theta_t_i_bar = None

        self.theta_0 = []
        self.theta_0 = [param.cpu() for param in self.func.parameters()]  # 将所有张量移动到 CPU
        
        all_p = []
        for param in self.theta_0:
            all_p += list(param.detach().numpy().ravel())  # 现在可以直接转换
        all_p = np.array(all_p)
        self.p = all_p.shape[0]
       

        self.layer_list = []

        # Initialize communication flags
        self.communicated_last_round = False
        self.all_communication_flags = []
        self.sdFinal = None
        # Initialize aggregated function
        self.loss_func = nn.MSELoss()
        self.func_agg = extend(Network(self.input_dim, init_params=self.theta_0).to(**tkwargs))
        
        # self.V_sync_NN_inv = torch.inverse(self.lam * torch.diag(torch.ones(self.p)))4
        # self.uncertainty = []
        # self.uncertainty_score = []
    
        # Initialize parameter dimensions
        
        self.context_list =  [[] for i in range(self.N)]
        self.best_arm = None
        self.best_r = float('-inf')
        self.A  = [() for jj in range(self.N)]
        self.B  = [() for jj in range(self.N)]
        
        self.bacth_size = 0
        self.text_un = [[] for _ in range(10) ]
        self.text_arm = []

        self.W_syn = torch.zeros((input_dim, input_dim)).to(**tkwargs)
        self.W_new = torch.zeros((self.N, input_dim, input_dim)).to(**tkwargs)
        self.V_t = torch.stack([self.lam * torch.eye(m=input_dim, n=input_dim) for _ in range(self.N)], dim=0).to(**tkwargs)
        self.V_last = self.lam * torch.ones((input_dim, input_dim)).to(**tkwargs)

        self.lr=lr
        self.weight_decay=weight_decay

        

        print("p:", self.p) 
        print("self.diag：", self.diag)
        print("self.nu:",self.nu)
        print("self.nu_2:",self.nu_2)
        print("self.lr:",self.lr)
        print("self.weight:",self.weight_decay)
        # print("self.theta_0：", self.theta_0)


       
        
        


    def select(self, context, List_score, select_idx_history, t, init_instructions, model_forward_api, prompt_domain_id=None, batch_size=300):
        
        p = self.p
        if np.any(self.communication_flag):
            self.communicated_last_round = True
        # for itr in self.run_list:

        for i in range(self.N):
          
          # if dataset == "shuttle" or dataset == "MagicTelescope":
          #     context, rwd = b_list[i].step()
          #     fs = rwd
          # elif dataset == "cosine" or dataset == "square":
          #     context, rwd, fs = bandit_contextual(a_ground, K_arms)

                
          self.t_last = t - 1


  
          # list_temp = []
          # context_size = context.shape[0]        
          
          # n_batch  = context_size // self.N
          # batch_size = n_batch // self.new_iter
          # self.batch_size=batch_size

          # list_temp = context[(i*n_batch+t*batch_size):(i*n_batch+(t+1)*batch_size)]
          # list_temp = context[(i*n_batch):(i*n_batch+(t+1)*batch_size)]

          # print("range:",i*n_batch,i*n_batch+(t+1)*batch_size)
          # print("range:",i*n_batch+t*batch_size,i*n_batch+(t+1)*batch_size)
          # print("n_batch,new_iter,batch_size:", n_batch, self.new_iter, batch_size)

          seed = 66 + i  # 每次一个不同的种子
          torch.manual_seed(seed)
          sample  = torch.randperm(context.size(0))[:100]
          sample  = sample.tolist()
          self.context_list[i] = context[sample]
          
          g_0_list = []
          mu = []
          self.func.train()
          # for j in range(n_batchs):
          #     if j == n_batchs - 1:
          #         context_batch = context[(j*batch_size):]
          #     else:
          #         context_batch = context[(j*batch_size):((j+1)*batch_size)]

          
          mu_ = self.func(self.context_list[i])
          sum_mu = torch.sum(mu_)
          with backpack(BatchGrad()):
              sum_mu.backward()                
          g_list_ = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1)
          g_0_list.append(g_list_.cpu())
          mu.append(mu_.cpu())
          g_0_list = torch.vstack(g_0_list)
          mu = torch.vstack(mu)
          print("g_0_list:",g_0_list.shape)
          
          
         
          
          
          if self.alpha_ts[t]>=0:
            self.func_agg.eval()  # 切换到评估模式
            with torch.no_grad():  # 不需要梯度，加速 + 节省显存
                UCB_1_first = self.func_agg(self.context_list[i])
            UCB_1_first = torch.squeeze(UCB_1_first).to(**tkwargs) 
            # print("UCB_1_first:", UCB_1_first.shape)

            # self.nu * (self.lam**0.5)
            uncertainty =  self.nu * (self.lam**0.5) * torch.sqrt(torch.diagonal(self.context_list[i] @ torch.inverse(self.V_t[i]) @ self.context_list[i].T)).reshape(-1)/pow(self.N,0.5)
            UCB_news = UCB_1_first + uncertainty

            if i == 0: 
              print("func:",UCB_1_first)
              print("List_score:",np.array([List_score[j] for j in sample]).reshape(10, 10))
              print("uncertainty:",uncertainty)
              # print("self.context_list[i]:",self.context_list[i])
              # print("torch.inverse(self.V_t[i]):",torch.inverse(self.V_t[i]))
              # print("diagonal:",torch.diagonal(self.context_list[i] @ torch.inverse(self.V_t[i]) @ self.context_list[i].T))
            metrics = compute_metrics(torch.tensor([List_score[j] for j in sample]).to(**tkwargs),UCB_1_first)
            print(metrics)

            # if i == 0:
            #   self.uncertainty.append(torch.topk(UCB_1_second, k=20,dim=0).indices.tolist())
            #   self.uncertainty_score.append(UCB_1_second[0:49].tolist())

            #   if t == 49:
            #     print("***********************************************************************")
            #     for __ in range(50):
            #        print(self.uncertainty[__])
            #     for __ in range(50):
            #        print(self.uncertainty_score[__])    
            #     print("***********************************************************************")
            

            # print("UCB_2:", UCB_2.shape)
            # print("UCB_2:", UCB_2)
          
          # arm_select = torch.argmax(UCB_2)
          # if self.alpha_ts[t] == 0:
          #   arm_select1 = torch.randint(0, UCB_2.shape[0], (1,))
          #   arm_select1_ori = arm_select1
          # else:
          
          arm_select1 = torch.topk(UCB_news, k=1,dim=0).indices
          arm_select1_ori = sample[arm_select1.item()]

          # if i == 0:
          #     seed = 666 + t  # 每次一个不同的种子
          #     torch.manual_seed(seed)
          #     arm_select1 = torch.randint(0, 9, (1,))
          #     arm_select1_ori = sample[arm_select1.item()]

          #     self.text_arm.append(arm_select1.item())
          #     for __ in range(10):
          #       self.text_un[__].append(uncertainty[__].item())
          #     if t == 49:
          #       colors = ['red', 'blue', 'green', 'orange', 'purple','cyan', 'magenta', 'brown', 'pink', 'gray']

          #       markers = ['o', 's', '^', 'D', 'v','>', '<', 'p', '*', 'h'] 

          #       x = list(range(50))

          #       plt.title(f"Uncertainty")
          #       plt.xlabel("x")
          #       plt.ylabel("y")
          #       plt.xlim(0, 50)
          #       plt.ylim(0, max(max(sublist) for sublist in self.text_un) + 0.1)

          #       for __ in range(10):
          #           scores = self.text_un[__]  
          #           plt.plot(
          #               x, scores, 
          #               label=f"arm{__}", 
          #               color=colors[__ % len(colors)], 
          #               marker=markers[__ % len(markers)],
          #               linestyle='-', 
          #               markersize=4.5
          #           )

          #       plt.legend()
          #       plt.show()
          #       print("arm:",self.text_arm)

          
          print("iter {0} --- agent {1} ".format(t, i))
          print("arm_select1", arm_select1_ori) 
         
          score_1 = List_score[arm_select1_ori]
          if arm_select1_ori not in self.select_history[i]:
            self.reward_list[i].append(score_1)
            self.select_history[i] += [int(arm_select1_ori)]
          # r = (score_1 + score_1)/2 #here ..........................................
          self.W_new[i, :, :] += 10*torch.ger(context[arm_select1_ori], context[arm_select1_ori])
          self.V_t[i, :, :] = self.lam * torch.eye(self.input_dim).to(**tkwargs) + self.W_syn + self.W_new[i, :, :]

          ####这里的判断更新的代码可能不对，我建议你直接写成true，也就是每轮都更新

          criterion = (torch.sum(torch.log(torch.diagonal(self.V_t[i, :, :], 0))) - \
                      torch.sum(torch.log(torch.diagonal(self.V_last, 0))))
          if (criterion * (t - self.t_last) >= 0):
             self.communication_flag[i] = 1


      
          
          

          # reg = np.max(fs) - r

          # self.regrets_per_agent.append(reg)
          

          self.A[i] = (arm_select1_ori)
          self.B[i] = (score_1)
          

          # self.best_arm = self.best_arm if self.best_r > score_1 else arm_select1_ori
          # self.best_r = self.best_r if self.best_r > score_1 else score_1
          #...........................................................................
            
           

        self.communicated_last_round = False
        

        if np.any(self.communication_flag):
            
            if t < self.stop_training_after_iter and self.alpha_ts[t+1]>0 :
                for i in range(self.N): 
                  func = extend(Network(input_dim=768).to(**tkwargs))
                  if self.init_state_dict is not None:
                       func.load_state_dict(deepcopy(self.init_state_dict))
                  
                  
                  optimizer = torch.optim.Adam(func.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)

                  standardized_reward = torch.tensor(self.reward_list[i], dtype=torch.float32).to(**tkwargs)

                  func.train()
                  for epoch in range(self.local_training_iter):
                      func.zero_grad()
                      optimizer.zero_grad()

                      # 前向传播
                      pred = func(context[self.select_history[i]].to(**tkwargs)).view(-1)

                      # 计算损失
                      loss = self.loss_func(pred, standardized_reward)

                      # 反向传播 + 参数更新
                      loss.backward()
                      optimizer.step()
                      # print("loss:",loss.item())

                      # 可选：验证集监控（假设你有 val_pred 和 val_target）
                      # random.seed(2022+i)  
                      # val_pred = random.sample(range(0, 500), 20) 
                      # val_target = torch.tensor(List_score[val_pred], dtype=torch.float32).to(**tkwargs)
                      # func.eval()  # 切换到评估模式
                      # with torch.no_grad():
                      #     val_pred = func(context[val_pred].to(**tkwargs)).view(-1)
                          
                      #     # 计算验证损失
                      #     val_loss = self.loss_func(val_pred, val_target)
                      
                      # scheduler.step(val_loss)

                  print("Federated Training Loss : ", loss.item())
                  self.state_dict_list[i] = deepcopy(func.state_dict())

                  print("pred:",pred)
                  print("standardized_reward:",standardized_reward)
                  metrics1 = compute_metrics(standardized_reward, pred)
                  print(metrics1)

                  # print("pred_evaluate:",val_pred)
                  # print("reward_evaluate:",val_target)
                  # metrics2 = compute_metrics(val_target,val_pred)
                  # print(metrics2)


                  
                  

        self.all_communication_flags.append(np.any(self.communication_flag))
        
        # self.regrets.append(self.regrets_per_agent)
        # if t % self.save_interval == 0:
        #     all_info = {"regrets":self.regrets, "communication_flag":self.all_communication_flags}
        #     pickle.dump(all_info, open(self.log_file_name, "wb"))


        
        ## below is done by the central server
        if np.any(self.communication_flag):
            for i in range(self.N):
                self.W_syn += self.W_new[i, :, :]
                self.W_new[i, :, :] = torch.zeros((self.input_dim,self.input_dim), dtype=torch.float64).to(**tkwargs)
                self.V_last = self.lam * torch.ones((self.input_dim, self.input_dim)).to(**tkwargs) + self.W_syn


            if self.alpha_ts[t+1] > 0:

                if t<self.stop_training_after_iter:
                    ##### NN parameter aggregation
                    self.sdFinal = self.state_dict_list[0]
                    # Average all parameters
                    for key in self.state_dict_list[0]:
                        test = torch.zeros(self.state_dict_list[0][key].shape).to(**tkwargs)
                        for i in range(self.N):
                            test += self.state_dict_list[i][key].to(**tkwargs) / self.N
                        self.sdFinal[key] = test
                    self.func_agg = extend(Network(self.input_dim, init_params=self.theta_0).to(**tkwargs))
                    self.func_agg.load_state_dict(self.sdFinal)   
        return self.A, self.B


    def find_best(self, context, select_idx_history, t, score_binary=False, batch_size=300): 
        
        context_size = context.shape[0]        
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
        mu = []
        g_0_list = []
        self.func.eval()
        for i in range(n_batchs):
            if i == n_batchs - 1:
                context_batch = context[(i*batch_size):]
            else:
                context_batch = context[(i*batch_size):((i+1)*batch_size)]

            mu_ = self.func(context_batch)
            sum_mu = torch.sum(mu_)
            with backpack(BatchGrad()):
                sum_mu.backward()                
            g_list_ = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1)
            g_0_list.append(g_list_.cpu())
            mu.append(mu_.cpu())
        mu = torch.vstack(mu)
        g_0_list = torch.vstack(g_0_list)
        
        


        # UCB_2_first = torch.inner(g_0_list, self.theta_t_i_bar)
        # if self.diag:
        #   UCB_2_second = self.nu * np.sqrt(self.lam) * torch.sqrt(torch.sum(g_0_list * g_0_list * self.V_sync_NN_inv, dim=1))
        # else:
        #   tmp = torch.matmul(g_0_list, self.V_sync_NN_inv)
        #   UCB_1_second = self.nu_2 * np.sqrt(self.lam) * torch.sqrt(torch.matmul(tmp, torch.transpose(g_0_list, 0, 1)))
        #   UCB_1_second = torch.diagonal(UCB_1_second, 0)
        #   UCB_2 = (UCB_2_first + UCB_2_second).to(**tkwargs)

        UCB_1_first = self.func_agg(context)
        # UCB_2 = self.alpha_ts[t] * UCB_1 + (1 - self.alpha_ts[t]) * UCB_2
        
        # select the first one
        if score_binary:
            arm = self.best_arm
        else:
            # print("select_idx_history")
            all_queried = torch.tensor(select_idx_history).view(-1)
            arm_ = torch.argmax(UCB_1_first[all_queried])
            arm = all_queried[arm_] 
            

        return arm
   
    # def train(self, context=None, reward=None, local_training_iter=30):
    #     L_list = []
    #     for i in range(self.N):
    #         if self.init_state_dict is not None:
    #             self.l_list[i].load_state_dict(deepcopy(self.init_state_dict))
    #             # self.l_list[i] = extend(Network(self.input_dim, 32, 2, init_params=self.theta_0).to(**tkwargs))
    #         if context is not None:
    #             self.pair_embedding_list[i] = torch.cat((self.pair_embedding_list[i], context.reshape(1,-1).to(**tkwargs)), dim=0)
    #             self.reward_list[i] = torch.cat((self.reward_list[i], torch.tensor([reward]).reshape(-1).to(**tkwargs).to(dtype=torch.float32)))

    #         self.len_list[i] = self.pair_embedding_list[i].shape[0] 
            
    #         optimizer = torch.optim.Adam(self.l_list[i].parameters(), lr=1e-3, weight_decay=self.lam / (self.len_list[i]*10))
                      
            
            
    #         standardized_reward = self.reward_list[i].reshape(-1)
            
    #         self.l_list[i].train()
    #         for _ in range(self.local_training_iter):
    #             self.l_list[i].zero_grad()
    #             optimizer.zero_grad()
    #             pred = self.l_list[i](self.pair_embedding_list[i])
                
            
    #             loss = self.loss_func(pred, standardized_reward)
    #             loss.backward()
    #             optimizer.step()

                
    #             # print(loss)
    #         print("Training Loss : ", loss.item())    
    #         L_list.append(self.l_list[i].state_dict())

            
        
            
    #     return L_list



class LinearModel(nn.Module):
    def __init__(self, input_dim, init_params=None):
        super(LinearModel, self).__init__()

        self.linear = nn.Linear(input_dim, 1, bias=False)
    
    def forward(self, x):
        y = self.linear(x)
        return y


class LinearDBDiag:
    def __init__(self, input_dim, lamdba=1, nu=1, style='ucb', init_x=None, init_y=None, diagonalize=True):

        self.diagonalize = diagonalize
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.func = LinearModel(input_dim).to(**tkwargs)
        self.init_state_dict = deepcopy(self.func.state_dict())

        if init_x is not None:
            self.pair_embedding = init_x.to(**tkwargs)
        else:
            self.pair_embedding = None
        if init_y is not None:
            self.reward = init_y.to(**tkwargs).to(dtype=torch.int64)
        else:
            self.reward = None
        self.len = 0
        self.lamdba = lamdba
        self.total_param = input_dim

        self.nu = nu
        self.lamdba = lamdba
        self.style = style
        self.loss_func = nn.MSELoss()
        self.mean = None
        self.std = None

    def select(self, context, select_idx_history, prompt_domain_id=None, batch_size=300):     
        context_size = context.shape[0]        
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
        mu = []
        self.func.eval()
        for i in range(n_batchs):
            if i == n_batchs - 1:
                context_batch = context[(i*batch_size):]
            else:
                context_batch = context[(i*batch_size):((i+1)*batch_size)]

            mu_ = self.func(context_batch)
            mu.append(mu_.cpu())
        mu = torch.vstack(mu)
        
        # select the first one
        if prompt_domain_id is None:
            arm1 = torch.argmax(mu.view(-1))
        else:
            arm1_ = torch.argmax(mu.view(-1)[prompt_domain_id])
            prompt_domain_id_ = torch.tensor(prompt_domain_id)
            arm1 = prompt_domain_id_[arm1_]
        
        # select the second one
        history = torch.tensor(select_idx_history)
        grad_1 = context[history[:,0]]
        grad_2 = context[history[:,1]]
        feature = (grad_1 - grad_2).cpu()
 
        U = torch.matmul(feature.transpose(0,1), feature)
        U = U.diagonal()
        U = U + 1e-10
        
        grad_arm_1 = context[arm1]
        feature_arm_2 = (context - grad_arm_1).cpu()

        sigma = torch.sqrt(torch.sum(self.nu * feature_arm_2 * feature_arm_2 / U, dim=1))
        sample_r = mu.view(-1) + sigma.view(-1)
        
        if prompt_domain_id is None:
            sorted_idx = torch.argsort(sample_r, descending=True)
        else:
            sorted_idx_ = torch.argsort(sample_r[prompt_domain_id], descending=True)
            prompt_domain_id_ = torch.tensor(prompt_domain_id)
            sorted_idx = prompt_domain_id_[sorted_idx_]
        if sorted_idx[0] == arm1:
            arm2 = sorted_idx[1]
        else:
            arm2 = sorted_idx[0]
        return arm1, arm2


    def find_best(self, context, select_idx_history,  all_domain=False, batch_size=300):     
        context_size = context.shape[0]        
        n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
        mu = []
        self.func.eval()
        for i in range(n_batchs):
            if i == n_batchs - 1:
                context_batch = context[(i*batch_size):]
            else:
                context_batch = context[(i*batch_size):((i+1)*batch_size)]

            mu_ = self.func(context_batch)
            mu.append(mu_.cpu())
        mu = torch.vstack(mu)

        # select the first one
        if all_domain:
            arm = torch.argmax(mu.view(-1))
        else:
            all_queried = torch.tensor(select_idx_history).view(-1)
            arm_ = torch.argmax(mu.view(-1)[all_queried])
            arm = all_queried[arm_]
        
        return arm

    def train(self, context=None, reward=None, local_training_iter=30):
        if self.init_state_dict is not None:
            self.func.load_state_dict(deepcopy(self.init_state_dict))
        if context is not None:
            self.pair_embedding = torch.cat((self.pair_embedding, context.reshape(2, 1, -1).to(**tkwargs)), dim=1)
            self.reward = torch.cat((self.reward, torch.tensor([reward]).reshape(-1).to(**tkwargs).to(dtype=torch.int64)))

        self.len = self.pair_embedding.shape[1]
        optimizer = torch.optim.Adam(self.func.parameters(), lr=1e-3)
        self.func.train()
        for _ in range(local_training_iter):
            self.func.zero_grad()
            optimizer.zero_grad()
            side_1 = self.pair_embedding[0].reshape(self.len, -1)
            side_2 = self.pair_embedding[1].reshape(self.len, -1)
            pred_1 = self.func(side_1)
            pred_2 = self.func(side_2)
            logits = (pred_1 - pred_2).reshape(-1)
            reward_ = self.reward.reshape(-1)
            loss = F.binary_cross_entropy_with_logits(logits, reward_.to(dtype=torch.float32))
            
            loss.backward()
            optimizer.step()
            # print(loss)
        print("Training Loss : ", loss.item())
        return self.func.state_dict()


class MLPRegression_Train:
    def __init__(
        self,
        input_dim=4096,
        optimizer_fn=torch.optim.Adam,
        loss_fn=nn.MSELoss,
        lr=0.001,
        batch_size=64,
        epochs=30,
        device=None):

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        self.model = MLPRegression(input_dim).to(device)
        self.optimizer = optimizer_fn(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn()
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        # backup for the initial model weight and optimizer
        self.init_model_weight = deepcopy(self.model.state_dict())
        self.optimizer_fn = optimizer_fn
    
    def get_data_loader(self, X_train, Y_train):
        dataset = CustomImageDataset(X_train, Y_train)
        train_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return train_dataloader
        
    def fit(self, X_train, Y_train, verbose=False, epochs=None):
        if epochs == None:
            epochs = self.epochs

        train_loader = self.get_data_loader(X_train, Y_train)
        for e in range(epochs):
            self.model.train()
            
            # running local epochs
            for batch_idx, batch in enumerate(train_loader):
                data, label = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(data)
                loss = self.loss_fn(pred, label)
                loss.backward()
                self.optimizer.step()
            
            if verbose:
                print('Epoch: {}, Loss: {:.4f}'.format(e, loss))

        return self.model

    def select(self, context, diagonalize, lamdba, nu, style, ):
        self.model.train()
        mu = self.model(context)
        sum_mu = torch.sum(mu)
        with backpack(BatchGrad()):
            sum_mu.backward()

        g_list = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in self.func.parameters()], dim=1)

        if diagonalize:
            ### diagonalization
            sigma = torch.sqrt(torch.sum(lamdba * nu * g_list * g_list / self.U, dim=1))
        else:
            ### no diagonalization
            tmp = torch.matmul(g_list, torch.inverse(self.U))
            sigma = torch.sqrt(nu * lamdba * torch.matmul(tmp, torch.transpose(g_list, 0, 1)))
            sigma = torch.diagonal(sigma, 0)

        if style == 'ts':
            sample_r = torch.normal(mu.view(-1), sigma.view(-1))
        elif style == 'ucb':
            sample_r = mu.view(-1) + sigma.view(-1)
        arm = torch.argmax(sample_r)

        if diagonalize:
            ### diagonalization
            self.U += g_list[arm] * g_list[arm]
        else:
            ### no diagonalization
            self.U += torch.outer(g_list[arm], g_list[arm])

        return arm, g_list[arm].norm().item()

    def restart_model(self):
        self.model.load_state_dict(deepcopy(self.init_model_weight))
        self.optimizer = self.optimizer_fn(self.model.parameters(), lr=self.lr)


class LlamaForMLPRegression(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad()
    def get_last_token_hidden_state(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sequence_lengths: Optional[int] = None,
        n_prompt_tokens: Optional[int] = 0,
        pooling: Optional[str] = "last",
    ) -> Tuple:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if sequence_lengths is None:
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                if input_ids is not None:
                    sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1 + n_prompt_tokens).to(hidden_states.device)
                else:
                    sequence_lengths = -1
        if pooling == "last":
            pooled_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        elif pooling == "mean":
            pooled_states = hidden_states.mean(dim=1)
        elif pooling == "max":
            pooled_states = hidden_states.max(dim=1).values
        return (pooled_states,)

import json
import random
import torch
import numpy as np
import sys
import os
import nest_asyncio
import matplotlib.pyplot as plt
from torch.optim import SGD
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from torch.optim import LBFGS
from sklearn.preprocessing import MinMaxScaler
from backpack import backpack, extend
from backpack.extensions import BatchGrad


nest_asyncio.apply()
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将 experiments 目录添加到系统路径
experiments_dir = os.path.join(current_dir, '..')  # 假设 experiments 目录在上一级
sys.path.append(experiments_dir)

from automatic_prompt_engineer import config, llm
cwd = os.getcwd()
sys.path.append(cwd)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch import nn
from backpack import backpack, extend
from automatic_prompt_engineer import ape, data
from experiments.data.instruction_induction.load_data import load_data
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator, exec_evaluator
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
# from LlamaForMLPRegression import DoubleTS, LinearDBDiag, NeuralDBDiag
from Federated_neural_bandit import DoubleTS, LinearDBDiag, NeuralDBDiag
from automatic_prompt_engineer import evaluate, template, data
import re

import argparse
from experiments.evaluation.instruction_induction.utility import set_all_seed
import datetime
import torch.nn.functional as F
OPENAI_API_KEY = "YOUR_API_KEY_HERE"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

SMOKE_TEST = os.environ.get("SMOKE_TEST")
tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}
    
model_name = "vicuna"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
api_model = 'chatgpt'
alpha = 1
sigma = 1

class SimpleModel(nn.Module):
    def __init__(self, input_dim=768, hidden_size=32, depth=2, init_params=None, lamdba=1, nu=1,noise=10):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        super(SimpleModel, self).__init__()

        self.activate = nn.ReLU()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Linear(input_dim, hidden_size))
        for i in range(depth-1):
            self.layer_list.append(nn.Linear(hidden_size, hidden_size))
        self.layer_list.append(nn.Linear(hidden_size, 1))
        
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
        
        self.noise = noise
        self.len = 0
        self.lamdba = lamdba
        self.nu = nu
        self.loss_func = F.binary_cross_entropy_with_logits
        self.select_idx_history = []
        self.arm1_reward_history = []
        self.arm2_reward_history = []
        self.winner = []
        self.W_new = torch.zeros((input_dim, input_dim), device="cuda")
        
        
    
    def forward(self, x):
        y = x
        for i in range(len(self.layer_list)-1):
            y = self.activate(self.layer_list[i](y))
        y = self.layer_list[-1](y)
        return y
    

def select(model,score_list,context,W_sync,beta_t,batch_size=300):

    total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

    context_size = context.shape[0]        
    n_batchs = context_size // batch_size + int((context_size % batch_size) != 0)
    g_list = []
    mu = []
    model.train()
    for i in range(n_batchs):
        if i == n_batchs - 1:
            context_batch = context[(i*batch_size):]
        else:
            context_batch = context[(i*batch_size):((i+1)*batch_size)]

        mu_ = model(context_batch)
        sum_mu = torch.sum(mu_)
        with backpack(BatchGrad()):
            sum_mu.backward()                
        g_list_ = torch.cat([p.grad_batch.flatten(start_dim=1).detach() for p in model.parameters()], dim=1)
        g_list.append(g_list_.cpu())
        mu.append(mu_.cpu())
    g_list = torch.vstack(g_list).to(**tkwargs)
    mu = torch.vstack(mu).to(**tkwargs)

    # select the first one
    
    arm1 = torch.argmax(mu.view(-1))

    # select the second one
    if model.select_idx_history == []:
      indices = list(set(range(len(context))) - {arm1.item()})
      arm2 = torch.tensor(random.choice(indices))
    else:  
   
      U = torch.linalg.inv(W_sync)
      arm_ucb = []
      
      print("'-----------------------mu;",mu)
      for arm in range(len(context)):
            diff = context[arm] - context[arm1]
            uncetainty = beta_t * torch.sqrt(torch.dot(diff, U @ diff))
            ucb = model(diff) + uncetainty
            arm_ucb.append(ucb)
            print("'-----------------------uncetainty;",uncetainty)
      sample_r = torch.stack(arm_ucb).to("cuda")
      sorted_idx = torch.argsort(sample_r, descending=True)
      if sorted_idx[0].item() == arm1.item():
                arm2 = sorted_idx[1]
      else:
                arm2 = sorted_idx[0]

    score1 = score_list[arm1]
    score2 = score_list[arm2]

    diff = (score1 - score2) * model.noise
    y_prob = 1 / (1 + np.exp(-diff))
    y = np.random.binomial(n=1, p=y_prob)

   
    model.select_idx_history += [[int(arm1.item()), int(arm2.item())]]
    model.arm1_reward_history.append(score1)
    model.arm2_reward_history.append(score2)
    model.winner.append(y)
    
    diff_arm = (context[arm1] - context[arm2])*5
    model.W_new = torch.outer(diff_arm.squeeze() , diff_arm.squeeze())

    return arm1, arm2, score1, score2



def get_client_gradient(model,init_state_dict,context,local_training_iter=30):
    
    model.load_state_dict(deepcopy(init_state_dict))
    model.len = len(model.select_idx_history)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=model.lamdba / (model.len+50))

    gradient_history = []

    model.train()
    for _ in range(local_training_iter):
        model.zero_grad()
        optimizer.zero_grad()
        history = torch.tensor(model.select_idx_history).to(**tkwargs)
        history = history.long()
        side_1 = context[history[:,0]]
        side_2 = context[history[:,1]]
        pred_1 = model(side_1)
        pred_2 = model(side_2)
        logits = (pred_1 - pred_2).reshape(-1)
        reward_ = torch.tensor(model.winner).reshape(-1).to(**tkwargs)
        loss = model.loss_func(logits, reward_.to(dtype=torch.float32))
        loss.backward()

        grads = [param.grad.detach().clone() for param in model.parameters()]
        gradient_history.append(grads)

        optimizer.step()
        # print(loss)
    print("Training Loss : ", loss.item())

    # avg_grads = []
    # for layer_grads in zip(*gradient_history):  # 每一层的所有迭代的梯度
    #     stacked = torch.stack(layer_grads, dim=0)
    #     avg = torch.mean(stacked, dim=0)
    #     avg_grads.append(avg)

    # return avg_grads  # List[Tensor]
    return gradient_history


# def average_gradients(gradient_list):
#     avg_grads = []
#     for grads_per_layer in zip(*gradient_list):  # 每一层的所有客户端梯度
#         stacked = torch.stack(grads_per_layer, dim=0)  # shape: (num_clients, param_shape...)
#         avg = torch.mean(stacked, dim=0)
#         avg_grads.append(avg)
#     return avg_grads

def average_agent_histories(all_agents_grad_history):
    num_epochs = len(all_agents_grad_history[0])  # 比如30
    num_layers = len(all_agents_grad_history[0][0])  # 每次训练的层数

    avg_grad_history = []
    for epoch in range(num_epochs):
        layer_grads_epoch = []
        for layer in range(num_layers):
            # 收集所有 agent 在当前 epoch 和当前层的梯度
            grads = [agent[epoch][layer] for agent in all_agents_grad_history]
            avg = torch.stack(grads, dim=0).mean(dim=0)
            layer_grads_epoch.append(avg)
        avg_grad_history.append(layer_grads_epoch)
    return avg_grad_history  # 结构同样是 List[List[Tensor]]

class FedAdamServer:
    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in model.parameters()]
        self.v = [torch.zeros_like(p) for p in model.parameters()]
        self.t = 0

    def step(self, avg_grads):
        for gradient in avg_grads:
            self.t += 1
            with torch.no_grad():
                for i, (param, grad) in enumerate(zip(self.model.parameters(), gradient)):
                    self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                    self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)

                    m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                    update = m_hat / (v_hat.sqrt() + self.eps)
                    param.data -= self.lr * update


def extract_sub_sentence(long_sentence):
    matches = re.findall('<prompt>(.*?)</prompt>', long_sentence)
    return matches

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_sen_embedding(model, tokenizer, sentences):
    # Tokenize sentences
    # print(sentences)
    # raise NotImplementedError
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    return sentence_embeddings


class LMForwardAPI:
    def __init__(self, model_name='vicuna', eval_data=None, init_prompt=None, init_qa_gen=None, conf=None, base_conf=None,
                 prompt_gen_data=None, n_prompt_tokens=None, few_shot_data=None, random_proj=None, intrinsic_dim=None, magnitude=None, norm_method=None):
        self.init_qa_gen = init_qa_gen
        self.init_prompt = init_prompt[0]
        init_qa = self.init_qa_gen()
        self.init_token = init_prompt[0] + init_qa
        self.count = 0
                
        ## eval preparation
        self.conf = config.update_config(conf, base_conf)
        self.eval_data = eval_data
        self.eval_template = template.EvalTemplate("Instruction: [PROMPT]\n\nInput: [INPUT]\n Output: [OUTPUT]")
        self.demos_template = template.DemosTemplate("Input: [INPUT]\nOutput: [OUTPUT]")
        
        if api_model in ['llama', 'flan-t5']:
            self.api_model = exec_evaluator(api_model, self.conf)

        if few_shot_data is None:
            self.few_shot_data = prompt_gen_data
        
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.num_call = 0
        self.best_instruction = None
        self.prompts_set = dict()
        self.prompts_list = []
        self.parents = []
        self.best_score = 0
        self.score_mean = None
        self.score_std = None
        self.score_min = None
        self.score_max = None
        self.magnitude = magnitude
        self.norm_method = norm_method
        self.init_user_prompt = None

    def update_init_token(self):
        # randomly choose a qa
        init_qa = self.init_qa_gen()
        self.init_token = self.init_prompt + init_qa

    def initialize_prompts(self, num_init, task, method):
        ini_prompts_his = {}
        print(self.conf['evaluation']['model'])
        model = llm.model_from_config(self.conf['evaluation']['model'])
        if method == 'rephrase':
            model_outputs = model.generate_text(self.init_token, 1, 0.5)
            ini_prompts_his[model_outputs[0]] = 0
            self.init_user_prompt = model_outputs[0]
        while len(ini_prompts_his) < num_init:
            if method == 'induction':
                if task in ['sum', 'first_word_letter', 'periodic_elements', 'active_to_passive']:
                    random_prompt = model.generate_text(self.init_token, 1, 1, use_seed=False)[0]
                    model_outputs = model.generate_text("Rephrase the following instruction: " + random_prompt + "\n the rephrased instruction is: ", 1, 1, use_seed=False)
                else:
                    model_outputs = model.generate_text(self.init_token, 1, 0.5)
                # if model_outputs[0] not in ini_prompts_his:
                ini_prompts_his[model_outputs[0]] = 0
                self.update_init_token()
                print(f'{task}: {len(ini_prompts_his)}')
            elif method == 'rephrase':
                if task in ['odd_one_out', 'orthography_starts_with']:
                    model_outputs = model.generate_text("Rephrase the following instruction: " + self.init_user_prompt + "\n the rephrased instruction is: ", 1, 1.5, use_seed=False)
                else:
                    model_outputs = model.generate_text("Rephrase the following instruction: " + self.init_user_prompt + "\n the rephrased instruction is: ", 1, 1, use_seed=False)
                ini_prompts_his[model_outputs[0]] = 0
                print(f'{task}: {len(ini_prompts_his)}')
        return list(ini_prompts_his.keys())
    
    def selection(self, num_next_gen):
        scores = np.array([self.prompts_set[tmp] for tmp in self.parents])
        num_parents = len(self.parents)
        probability = []
        if np.sum(scores) == 0:
            probability = np.ones(num_parents)/ num_parents
        else:
            probability = scores / np.sum(scores)
        
        all_parents = []
        for i in range(num_next_gen):
            try:
                parent_pair = np.random.choice(self.parents, size=2, replace=False, p=probability)
            except:
                parent_pair = np.random.choice(self.parents, size=2, replace=True, p=probability)
            all_parents += [parent_pair]
        return all_parents
    
    def evolution(self, all_parents):

        next_gens = []
        model = llm.model_from_config(self.conf['evaluation']['model'])
        
        template = "Please follow the instruction step-by-step to generate a better prompt.\n1. Cross over the following prompts and generate a new prompt:\nPrompt 1: [prompt_id1].\nPrompt 2: [prompt_id2].\n2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>."
        for parents_ in all_parents:
            template_ = template.replace('[prompt_id1]', parents_[0])
            template_ = template_.replace('[prompt_id2]', parents_[1])
            model_outputs = model.generate_text(template_, 1, 0)
            model_outputs_ = extract_sub_sentence(model_outputs[0])
            if len(model_outputs_) != 0:
                model_outputs = model_outputs_[0]
                print(f"EVOL: {model_outputs}")
            else:
                model_outputs = model_outputs[0]
            next_gens += [model_outputs]
            #print("next_gens:",next_gens)
            #print("model_outputs_[0]:",model_outputs_[0])
        return next_gens
    
    def update(self, next_gens):
        next_gens_scores = []
        for gen_ in next_gens:
            score_ = self.eval([gen_])
            next_gens_scores += [score_]
        self.this_iter_best = np.max(next_gens_scores) 
        num_parents = len(self.parents)
        parents_next_gen = self.parents + next_gens
        all_scores = [self.prompts_set[tmp] for tmp in parents_next_gen]
        idx_rank = np.argsort(all_scores)
        selected_idx = idx_rank[-num_parents:]
        new_parents = []
        #print("self.parents:",self.parents)
        for idx_ in selected_idx:
            new_parents += [parents_next_gen[idx_]]
        self.parents = new_parents
    
    def eval(self, instruction=None, test=False):
        if instruction[0] in self.prompts_set.keys():
            dev_perf = self.prompts_set[instruction[0]]
        else:
            if api_model in ['chatgpt']: 
                dev_perf, _ = evaluate.evaluate_prompts(instruction, self.eval_template, self.eval_data, self.demos_template, self.few_shot_data, self.conf['evaluation']['method'], self.conf['evaluation'])
                dev_perf = dev_perf.sorted()[1][0]
            else:
                raise NotImplementedError

            if not test:
                if dev_perf >= self.best_last_perf:
                    self.count += 1

                if dev_perf >= self.best_dev_perf:
                    self.best_dev_perf = dev_perf
                    self.best_instruction = instruction

                if self.norm_method == 'standard':
                    dev_perf = self.magnitude * (dev_perf - self.score_mean) / self.score_std
                elif self.norm_method == 'minmax':
                    dev_perf = self.magnitude * (dev_perf - self.score_min) / (self.score_max - self.score_min)
                self.prompts_set[instruction[0]] = dev_perf
                self.prompts_list.append((len(self.prompts_list), instruction[0], dev_perf))
                print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
                    round(float(dev_perf), 4),
                    round(float(dev_perf), 4),
                    round(float(self.best_dev_perf), 4)))
                print('********* Done *********')
        return dev_perf

    def return_best_prompt(self):
        return self.best_instruction

    def return_prompts_set(self):
        return self.prompts_set

    def return_prompts_list(self):
        return self.prompts_list



def run(tasks, n_prompt_tokens,noise, nu, lamdba, n_init, n_domain, total_iter, local_training_iter, random_proj, intrinsic_dim, n_eval, gpt, init_scale, pooling, agent_N_list, method, args):
    
    # assert task in TASKS, 'Task not found!'
    num_iterations = 50
    Agent_compare_all = np.zeros((len(agent_N_list), num_iterations))
    Max_score = []


    for task in tasks:
        induce_data, test_data = load_data('induce', task), load_data('eval', task)
    
        # print(induce_data)
        
        
        induce_data_size = len(induce_data[0])
        prompt_gen_size = min(int(induce_data_size * 0.5), 100)

        # 调用 create_split 函数，使用 formatted_data 作为参数
        prompt_gen_data, eval_data = data.create_split(induce_data, split_size=prompt_gen_size)

        # Data is in the form input: single item, output: list of items
        # For prompt_gen_data, sample a single item from the output list
        prompt_gen_data = prompt_gen_data[0], [random.sample(output, 1)[0]
                                            for output in prompt_gen_data[1]]

        demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
        eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\n\nOUTPUT: [OUTPUT]" # change the evaluation template
        init_prompt = ['\n']
        prompt_gen_template = "[full_DEMO]\n\nThe instruction was to"

        base_conf = '../experiments/configs/instruction_induction.yaml'
        conf = {
            'generation': {
                'num_subsamples': 1,
                'num_demos': 5,
                'num_prompts_per_subsample': 20,
                'model': {
                    'gpt_config': {
                        'model': gpt
                    }
                }
            },
            'evaluation': {
                'method': exec_accuracy_evaluator,
                'task': task,
                'num_samples': min(50, len(eval_data[0])),
                'model': {
                    'gpt_config': {
                        'model': gpt
                    }
                }
            }
        }
        # start a prompt and use rephrasing to generate the initial instructions
        # make the demo automatically
        def init_qa_gen():
            subsampled_data = data.subsample_data(prompt_gen_data, conf['generation']['num_demos'])
            prompt_gen_template_ = template.InitQATemplate(prompt_gen_template)
            d_template = template.DemosTemplate(demos_template)
            demos = d_template.fill(subsampled_data)
            return prompt_gen_template_.fill(demos)

        model_forward_api = LMForwardAPI(model_name=model_name, eval_data=eval_data, init_prompt=init_prompt, 
                                        init_qa_gen=init_qa_gen, conf=conf, base_conf=base_conf, prompt_gen_data=prompt_gen_data,
                                        n_prompt_tokens=n_prompt_tokens, random_proj=random_proj,intrinsic_dim=intrinsic_dim)
        print(set_all_seed(args.trial))
        # check whether a certain file exists
        print(os.getcwd())
        if args.candidate_method == "induction":
            path_ = f"./query/{task}_{args.n_domain}"
        elif args.candidate_method == "rephrase":
            path_ = f"./query/{task}_{args.n_domain}_rephrase"
            
        if os.path.exists(path_):
            with open(path_, 'r') as fp:
                domains = json.load(fp)
                init_instructions = domains['instructions']
        else:
            # create the folder if it does not exist
            if not os.path.exists("./query"):
                os.mkdir("./query")
            init_instructions = model_forward_api.initialize_prompts(args.n_domain, task, args.candidate_method)
            with open(path_, 'x') as fp:
                domains = {"instructions": init_instructions}
                json.dump(domains, fp, indent=4)
        


        model_path = "/content/drive/MyDrive/APOHF-main-test/Induction/experiments/model"

        sen_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2", cache_dir=model_path)
        sen_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2", cache_dir=model_path)

        # sen_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        # sen_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        
        sen_embeddings = get_sen_embedding(sen_model, sen_tokenizer, init_instructions)
        sen_embeddings = sen_embeddings.to(**tkwargs)
        

        # 定义文件路径
        file_path = "/content/drive/MyDrive/APOHF-main-s/Induction/experiments/list_data.json"
        target_list_key = task
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                json.dump({}, f)  # 初始化为空字典，避免后续 json.load() 报错

        # **读取 JSON 数据**
        with open(file_path, "r") as f:
            try:
                datas = json.load(f)  # 读取 JSON
            except json.JSONDecodeError:
                datas = {}  # 如果文件损坏，初始化为空字典

        # **检查是否已有 "" 列表**
        if target_list_key not in datas:
            score_list = []
            for i in range(500):
                score_test = model_forward_api.eval([init_instructions[i]])  # 你的模型推理代码
                score_list.append(float(round(score_test, 2)))
            
            # **存入 JSON**
            datas[target_list_key] = score_list
            with open(file_path, "w") as f:
                json.dump(datas, f, indent=4)  # 保存数据
                

        # std = sen_embeddings.std(dim=0) + 1e-30
        # mean = sen_embeddings.mean(dim=0)
        # sen_embeddings = (sen_embeddings - mean) / std 
        List_score = datas.get(target_list_key)
        List_score = np.array(List_score)

        # scaler = MinMaxScaler()
        # List_score_norm = scaler.fit_transform(np.array(List_score).reshape(-1, 1)).flatten()
        # List_score = [10*x for x in List_score]
        # List_score = torch.tensor(List_score, dtype=torch.float32)
        # List_score = ((List_score - List_score.mean(dim=0)) / (List_score.std(dim=0) + 1e-30)).reshape(-1)
        # List_score = List_score.tolist()
        # Randomly eval scores to normalize the scores
        test_num = 50
        all_tmp_scores = []
        for tmp in range(test_num):
            prompt_tmp = np.random.choice(range(500))
            score_tmp = List_score[prompt_tmp]
            all_tmp_scores += [score_tmp]
        model_forward_api.score_mean = np.mean(all_tmp_scores)
        model_forward_api.score_std = np.std(all_tmp_scores)
        model_forward_api.score_min = np.min(all_tmp_scores)
        model_forward_api.score_max = np.max(all_tmp_scores)

        Agent_compare = [[] for _ in range(len(agent_N_list))]

        for aa in range(len(agent_N_list)):
            num_clients = agent_N_list[aa]
            contexts = [[] for _ in range(num_clients)]
            score_list = [[] for _ in range(num_clients)]
            now_values = [[] for _ in range(num_clients)]
            
            common_ratio = 0.5  # 50% 重合
            total_samples = 100
            common_count = int(total_samples * common_ratio)  # 50
            unique_count = total_samples - common_count       # 50

            # 保证每次运行都有相同的结果（可选）
            torch.manual_seed(8864)

            # 公共部分：先统一随机选 common_count 个样本
            shared_indices = torch.randperm(sen_embeddings.size(0))[:common_count]

  
            for i in range(num_clients):
                seed = 66 + i
                torch.manual_seed(seed) 

                # 剩余的 unique 部分，每个客户端单独采样
                all_indices = torch.randperm(sen_embeddings.size(0))
                unique_indices = []

                # 直到选 enough 个不在 shared 中的样本
                for idx in all_indices:
                    if idx not in shared_indices:
                        unique_indices.append(idx)
                    if len(unique_indices) >= unique_count:
                        break

                # 合并公共样本 + 独特样本
                client_sample = torch.cat([shared_indices, torch.tensor(unique_indices)], dim=0)
                sample  = client_sample.tolist()
                contexts[i] = sen_embeddings[sample]
                score_list[i] = List_score[sample]
 
            Max_score.append(max(score_list[0]))

            global_rounds = num_iterations #50

            # 初始化 server 模型和优化器
            server_model = extend(SimpleModel(lamdba=lamdba,nu=nu,noise=noise).to(**tkwargs))
            server_optimizer = FedAdamServer(server_model, lr=0.01)
            init_state_dict = deepcopy(server_model.state_dict())

            # 创建客户端
            clients = [deepcopy(server_model) for _ in range(num_clients)]
            best_instruction_over_iter = []
            value = []
            best_r = 0
            W_sync = lamdba * torch.eye(768, device="cuda")
            W_update = []
            delta = 0.1

            for t in range(global_rounds):
                print(f"iteration:{t}")
                all_grads = []
                

                #select(model,score_list,context,batch_size=300):
                for i, client_model in enumerate(clients):

                    print(f"agent{i}:")

                    beta_t = torch.sqrt(2 * torch.log(torch.tensor(1.0 / delta, device="cuda"))+ 768 * torch.log(torch.tensor(1 + t * num_clients/(lamdba * 768))))
                    arm1, arm2, score1, score2 = select(client_model,score_list[i],contexts[i],W_sync,beta_t)
                    W_update.append(client_model.W_new) 

                    print(f"arm1:{arm1},arm2:{arm2}")
                    print(f"score1:{score1},score2:{score2}")

                    if i == 0:
                        value += [score1]
                        if score1 >=  best_r:
                                best_r = score1
                                best_arm = arm1
                        best_instruction_over_iter += [(t, init_instructions[best_arm], score1)]
                        print("best_arm:", best_arm)
                        print(f"Best value found till now: {best_r}")

                    
                    grads = get_client_gradient(client_model,init_state_dict,contexts[i],local_training_iter=local_training_iter)
                    print(f"grads: {grads.shape}")
                    all_grads.append(grads)
  
                for W_new in W_update:
                  W_sync += W_new


                # Server 聚合所有客户端所有的梯度, 形成num_epochs*num_layers形状的梯度列表
              
                avg_grads = average_agent_histories(all_grads)
                server_optimizer.step(avg_grads)

                # 同步 server 模型给所有 clients（模拟广播）
                for client_model in clients:
                    for param, server_param in zip(client_model.parameters(), server_model.parameters()):
                        param.data.copy_(server_param.data)

               



            # Save results
            # output_dir = ""
            # os.makedirs(output_dir, exist_ok=True)
            # np.save(os.path.join(output_dir, "federated_cumulative_regret.npy"), np.array(cumulative_regret))

            Agent_compare[aa] = clients[0].arm1_reward_history

            print("第一行是arm1，第二行是arm2，第三行是arm1的reward")     
            for i, client_model in zip(range(num_clients), clients):
                scores1 = [float(x) for x in client_model.arm1_reward_history]
                scores2 = [float(x) for x in client_model.arm2_reward_history]
                history = np.array(client_model.select_idx_history)
                print("agent:", i, history[:,0])
                print("score1:", i, scores1 ) 
                print("agent:", i, history[:,1])   
                print("score2:", i, scores2 ) 

            
            colors = ['red', 'blue', 'green', 'orange', 'purple','cyan', 'magenta', 'brown', 'pink', 'gray']

            markers = ['o', 's', '^', 'D', 'v','>', '<', 'p', '*', 'h'] 

            x = list(range(num_iterations+1))
          

            plt.title(f"Reward Plot_{task}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.xlim(0, num_iterations)
            plt.ylim(0, 1.0)

            for i in range(num_clients):
                scores = [0] + clients[i].arm1_reward_history # 加入初始 0
                plt.plot(
                    x, scores, 
                    label=f"agent{i+1}", 
                    color=colors[i % len(colors)], 
                    marker=markers[i % len(markers)],
                    linestyle='-', 
                    markersize=3
                )

            plt.legend()
            # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # filename = f"reward_plot_{task}__n{noise}_lambda{lambda}_iter{num_iterations}_d{delta}_{timestamp}.png"
            # plt.savefig(f"/content/drive/MyDrive/Linear-main-test/Graph1/{filename}", dpi=300, bbox_inches='tight')
            plt.show()



            print('Evaluate on test data...')
            prompts = [best_instruction_over_iter[-1][1]]
            print("Best instruction is:")
            print(prompts)

            prompts_set = model_forward_api.return_prompts_set()
            print("The final instruction set is:")
            print(prompts_set)
            prompts_list = model_forward_api.return_prompts_list()
            prompts_set = model_forward_api.return_prompts_set()


            # Evaluate on test data
            print('Evaluating on test data...')

            test_conf = {
                'generation': {
                    'num_subsamples': 3,
                    'num_demos': 5,
                    'num_prompts_per_subsample': 0,
                    'model': {
                        'gpt_config': {
                            'model': gpt
                        }
                    }
                },
                'evaluation': {
                    'method': exec_accuracy_evaluator, # option: accuracy (cannot use likelihood here due to the textual outputs from ChatGPT do not have log prob)
                    'task': task,
                    #'num_samples': min(100, len(test_data[0])),应该和上面的induce_data的问题一样，将[0]去掉
                    'num_samples': min(100, len(test_data)),
                    'model': {
                        "name": "GPT_forward",
                        'gpt_config': {
                        'model': gpt
                        }
                    }
                }
            }
            # inputs = [v['input'] for k, v in test_data.items()]
            # outputs = [v['output'] for k, v in test_data.items()]
            # formatted_data = (inputs, outputs)  # 将输入和输出列表打包成元组
            test_res = ape.evaluate_prompts(prompts=prompts,
                                            eval_template=eval_template,
                                            eval_data=test_data,
                                            # eval_data=formatted_data,
                                            few_shot_data=prompt_gen_data,
                                            demos_template=demos_template,
                                            conf=test_conf,
                                            base_conf=base_conf)
            test_res = test_res[0]
            test_score = test_res.sorted()[1][0]

        
        Agent_compare_all += np.array(Agent_compare)/len(tasks)

        colors = ['red', 'blue', 'green', 'orange', 'purple','cyan', 'magenta', 'brown', 'pink', 'gray']

        markers = ['o', 's', '^', 'D', 'v','>', '<', 'p', '*', 'h'] 

        x = list(range(num_iterations+1))

        plt.title(f"Agent_num_compare_{task}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(0, num_iterations)
        plt.ylim(0, 1.0)

        for i in range(len(agent_N_list)):
            scores = [0] + Agent_compare[i]  # 加入初始 0
            plt.plot(
                x, scores, 
                label=f"agent_num_{agent_N_list[i]}", 
                color=colors[i % len(colors)], 
                # marker=markers[i % len(markers)],
                linestyle='-', 
                markersize=3
            )

        plt.legend()
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"Agent_num_compare_{task}_fd{feature_dim}_n{noise}_lr{lambda_reg}_iter{num_iterations}_d{delta}_{timestamp}.png"
        # plt.savefig(f"/content/drive/MyDrive/Linear-main-test/Graph2/{filename}", dpi=300, bbox_inches='tight')
        plt.show()

    colors = ['red', 'blue', 'green', 'orange', 'purple','cyan', 'magenta', 'brown', 'pink', 'gray']

    markers = ['o', 's', '^', 'D', 'v','>', '<', 'p', '*', 'h'] 

    x = list(range(num_iterations+1))

    plt.title(f"averge_score_{len(tasks)}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, num_iterations)
    plt.ylim(0, 0.8)

    for i in range(len(agent_N_list)):
        scores = [0] + Agent_compare_all[i].tolist()  # 加入初始 0
        plt.plot(
            x, scores, 
            label=f"agent_num_{agent_N_list[i]}", 
            color=colors[i % len(colors)], 
            # marker=markers[i % len(markers)],
            linestyle='-', 
            markersize=3
        )
        
    average_score = round(np.mean(Max_score),3)
    print(average_score) 
    plt.legend()
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # filename = f"averge_score_{len(tasks)}_fd{feature_dim}_n{noise}_lr{lambda_reg}_iter{num_iterations}_d{delta}_as{average_score}_{timestamp}.png"
    # plt.savefig(f"/content/drive/MyDrive/Linear-main-test/Graph3/{filename}", dpi=300, bbox_inches='tight')
    plt.show()
       
    return test_score, prompts, prompts_set, value, best_instruction_over_iter, init_instructions
    # print(f'Test score on ChatGPT: {test_score}')
    # antonyms

def parse_args():
    parser = argparse.ArgumentParser(description="InstructZero pipeline")

    parser.add_argument(
    "--tasks",
    type=str,
    nargs='+',  # 接收一个或多个 task
    default=['word_in_context','synonyms','cause_and_effect','object_counting','rhymes','periodic_elements','auto_categorization', 'orthography_starts_with', 'larger_animal','antonyms'],
    help="A list of task names to use.",
    )
    parser.add_argument(
        "--n_prompt_tokens",
        type=int,
        default=5,
        help="The number of prompt tokens."
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=0.3,
        help="Set the parameter nu."    
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=10,
        help="Set the parameter noise."    
    )
    parser.add_argument(
        "--lamdba",
        type=float,
        default=1,
        help="Set the lamdba parameter."    
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=100,
        help="Set the number of initialization points."    
    )
    parser.add_argument(
        "--n_domain",
        type=int,
        default=500,
        help="Set the number of domain."    
    )
    parser.add_argument(
        "--total_iter",
        type=int,
        default=165,
        help="Set the number of total queries."    
    )
    parser.add_argument(
        "--local_training_iter",
        type=int,
        default=30,
        help="Set the number of total queries."    
    )
    parser.add_argument(
        "--random_proj",
        type=str,
        default='uniform',
        help="Set the projection method."    
    )
    parser.add_argument(
        "--intrinsic_dim",
        type=int,
        default=100,
        help="Set the number of intrinsic dim."    
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=1000,
        help="Set the number of domains to be evaluated at each ucb iteration."    
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Set the name of the experiments."    
    )
    parser.add_argument(
        "--gpt",
        type=str,
        default="gpt-3.5-turbo",
        help="Which version of gpt to use."    
    )
    parser.add_argument(
        "--init_scale",
        type=float,
        default=1,
        help="Which scale to use."    
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="last",
        help="Which pooling method to use."    
    )
    parser.add_argument(
        "--func",
        type=str,
        default="neural",
        help="Which model to use, can be linear, neural."    
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="Trial ID."
    )
    parser.add_argument(
        "--magnitude",
        type=int,
        default=10,
        help="The magnitude of the scores."
    )
    parser.add_argument(
        "--norm_method",
        type=str,
        default='standard',
        help="The way to transform the value, standard, minmax."
    )
    parser.add_argument(
        "--candidate_method",
        type=str,
        default='induction',
        help="The way to generate candidates."
    )
    parser.add_argument(
      "--agent_N_list",
      type=int,
      nargs='+',               # 接收多个参数
      default=[1,3,5,7,10],
      help="The number of agents."
    )
    parser.add_argument(
      "--method",
      type=str,
      default="OGD",
      help="Algorithm."
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(set_all_seed(0))
    test_score, prompts, prompts_set, now_values, best_instruction_over_iter, init_instructions = run(
        tasks=args.tasks,
        n_prompt_tokens=args.n_prompt_tokens,
        nu=args.nu,
        noise=args.noise,
        lamdba=args.lamdba,
        n_init=args.n_init,
        n_domain=args.n_domain,
        total_iter=args.total_iter,
        local_training_iter = args.local_training_iter,
        random_proj=args.random_proj,
        intrinsic_dim=args.intrinsic_dim,
        n_eval=args.n_eval,
        gpt=args.gpt,
        init_scale=args.init_scale,
        pooling=args.pooling,
        agent_N_list=args.agent_N_list,
        method = args.method,
        args=args
    )
    
    args_dict = vars(args)
    args_dict['test_score'] = test_score
    # args_dict['valid_score'] = best_values[-1]
    args_dict['best_prompt'] = prompts
    args_dict['prompts_set'] = prompts_set
    # args_dict['best_values'] = best_values
    args_dict['best_instruction_over_iter'] = best_instruction_over_iter
    args_dict['init_instructions'] = init_instructions
    # args_dict['instruction_select_history'] = instruction_select_history
    args_dict['now_values'] = now_values

    save_dir = "./results/" + args.name
    
    # if no folder create one
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # get a path with the current time
    path = os.path.join(save_dir,args.tasks[0] + datetime.datetime.now().strftime("-%Y-%m-%d_%H-%M-%S") + "_trial{}".format(args.trial) +".json")

    with open(path, 'x') as fp:
        json.dump(args_dict, fp, indent=4)
    
    print("Finished!!!")
    print(f'Test score on ChatGPT: {test_score}')



import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import random

# ✅ 模型结构（可以根据任务替换）
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, data, target, loss_fn):
    model.eval()
    with torch.no_grad():
        output = model(data)
        loss = loss_fn(output, target).item()
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        accuracy = correct / len(target)
    return loss, accuracy


# ✅ 获取客户端上传的梯度
def get_client_gradient(model, data, target, loss_fn):
    model.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    grads = [param.grad.detach().clone() for param in model.parameters()]
    return grads

# ✅ 计算平均梯度
def average_gradients(gradient_list):
    avg_grads = []
    for grads_per_layer in zip(*gradient_list):  # 每一层的所有客户端梯度
        stacked = torch.stack(grads_per_layer, dim=0)  # shape: (num_clients, param_shape...)
        avg = torch.mean(stacked, dim=0)
        avg_grads.append(avg)
    return avg_grads

# ✅ FedAdam server optimizer
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
        self.t += 1
        with torch.no_grad():
            for i, (param, grad) in enumerate(zip(self.model.parameters(), avg_grads)):
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad.pow(2)

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                update = m_hat / (v_hat.sqrt() + self.eps)
                param.data -= self.lr * update

# ✅ 模拟联邦训练流程
def federated_training():
    num_clients = 30
    local_epochs = 10
    global_rounds = 30

    # 初始化 server 模型和优化器
    server_model = SimpleModel()
    server_optimizer = FedAdamServer(server_model, lr=0.01)

    # 创建客户端
    clients = [deepcopy(server_model) for _ in range(num_clients)]

    # 模拟数据
    dummy_data = torch.randn(10, 10)
    dummy_target = torch.randint(0, 2, (10,))
    loss_fn = nn.CrossEntropyLoss()

    for round in range(global_rounds):
        print(f"Round {round+1}")
        all_grads = []

        for client_model in clients:
            # 模拟 client 训练 local_epochs 次，每次上传一次梯度
            for _ in range(local_epochs):
                grads = get_client_gradient(client_model, dummy_data, dummy_target, loss_fn)
                all_grads.append(grads)

                # 本地更新（可选）
                with torch.no_grad():
                    for p, g in zip(client_model.parameters(), grads):
                        p -= 0.01 * g

        # Server 聚合所有客户端所有 epoch 的梯度
        avg_grads = average_gradients(all_grads)
        server_optimizer.step(avg_grads)

        # 同步 server 模型给所有 clients（模拟广播）
        for client_model in clients:
            for param, server_param in zip(client_model.parameters(), server_model.parameters()):
                param.data.copy_(server_param.data)

        eval_loss, eval_acc = evaluate(server_model, dummy_data, dummy_target, loss_fn)
        print(f"Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_acc:.4f}")

federated_training()
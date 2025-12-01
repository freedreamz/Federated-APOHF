import numpy as np
import torch
from torch.optim import SGD
import pandas as pd
import os
from tqdm import tqdm
from torch.optim import LBFGS

# 1. Federated Environment: Contextual Linear Dueling Bandit
class FederatedLinearDuelingBanditEnvironment:
    def __init__(self, feature_dim, num_arms, num_clients, noise=0.1):
        self.feature_dim = feature_dim
        self.num_arms = num_arms
        self.num_clients = num_clients
        self.noise = noise
        self.theta_star = torch.randn(feature_dim, device="cuda")  # Ground truth parameter

    def generate_context(self):
        """Generates a set of feature vectors (arms) for all clients."""
        return [torch.randn(self.num_arms, self.feature_dim, device="cuda") for _ in range(self.num_clients)]
    
    def get_preference(self, arm1_features, arm2_features):
        # Dot products
        arm1_utility = torch.dot(arm1_features, self.theta_star)
        arm2_utility = torch.dot(arm2_features, self.theta_star)

        diff = (arm1_utility - arm2_utility) * self.noise

        y_prob = torch.sigmoid(diff)

        y_prob_val = y_prob.detach().cpu().item()

        y = np.random.binomial(n=1, p=y_prob_val)

        return y


# 2. Federated Client

class FederatedClient_ogd:
    def __init__(self, client_id, feature_dim, lambda_reg):
        self.client_id = client_id
        self.feature_dim = feature_dim
        self.lambda_reg = lambda_reg
        self.V_t = lambda_reg * torch.eye(feature_dim, device="cuda")
        self.W_new = torch.zeros((feature_dim, feature_dim), device="cuda")
        self.local_gradient = torch.zeros(feature_dim, device="cuda")
        self.history_arm_1= []  
        self.history_arm_2= []  
        self.history_winner= []  

    def select_arms(self, theta_sync, context, beta_t, W_sync):
        arm1_idx = torch.argmax(context @ theta_sync).item()
        arm1 = context[arm1_idx]

        # Select the second arm (optimistic utility estimate)
        arm_ucb = []
        self.V_t = W_sync
        V_t_inv = torch.linalg.inv(self.V_t)
        for arm in context:
            diff = arm - arm1
            ucb = torch.dot(theta_sync, diff) + beta_t * torch.sqrt(torch.dot(diff, V_t_inv @ diff))
            arm_ucb.append(ucb)
        arm2_idx = torch.argmax(torch.tensor(arm_ucb, device="cuda")).item()

        return arm1_idx, arm2_idx
    
    def compute_gradient_ogd(self, theta_hat):

        gradient = torch.zeros_like(theta_hat, device="cuda")
        arm1 = self.history_arm_1[-1].clone().detach()
        arm2 = self.history_arm_2[-1].clone().detach()
        winner = self.history_winner[-1]
        diff = arm1 - arm2
        gradient = (torch.sigmoid(torch.dot(theta_hat, diff)) - winner) * diff

        return gradient
    def compute_w_update(self, arm1, arm2):
        """Computes the local W_new update."""
        diff = arm1 - arm2
        self.W_new = torch.outer(diff, diff)
        return self.W_new

# 3. Central Server
class CentralServer_ogd:
    def __init__(self, feature_dim, num_clients, lambda_reg=1.0):
        self.theta_sync = torch.zeros(feature_dim, device="cuda")
        self.W_sync = lambda_reg * torch.eye(feature_dim, device="cuda")
        self.theta_hat = torch.zeros(feature_dim, device="cuda")
        self.theta_avg = torch.zeros(feature_dim, device="cuda")
        self.theta_1 = torch.zeros(feature_dim, device="cuda")
        self.r = 1
        self.feature_dim = feature_dim
        self.lambda_reg = lambda_reg
        self.num_clients = num_clients


    def aggregate_w_updates(self, w_updates):
        """Aggregates W updates from all clients."""
        for W_new in w_updates:
            self.W_sync += W_new
    
    def solve_theta(self, clients):
        """Finds theta_t by minimizing the loss function \mathcal{L}_t(\theta)."""
        theta = torch.zeros(self.feature_dim, device="cuda", requires_grad=True)

        def loss_fn():
            loss = 0
            for client in clients:
                for x1, x2, y in zip(client.history_arm_1, client.history_arm_2, client.history_winner):
                    diff = x1 - x2
                    score = torch.dot(theta, diff)
                    if y == 1:
                        loss -= torch.log(torch.sigmoid(score))
                    else:
                        loss -= torch.log(torch.sigmoid(- score))
            loss += 0.5 * self.lambda_reg * torch.norm(theta) ** 2  # Regularization
            return loss

        optimizer = LBFGS([theta], line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            l = loss_fn()
            l.backward()
            return l

        optimizer.step(closure)
        return theta.detach()
    
    def aggregate_gradients_ogd(self, clients, theta_hat, lambda_reg=1.0):
        """Aggregates gradients from all clients."""
        gradients = []
        for client in clients:
            gradients.append(client.compute_gradient_ogd(theta_hat))
        # gradients = [client.compute_gradient(self.theta_sync) for client in clients]
        total_gradient = torch.sum(torch.stack(gradients), dim=0)
        return total_gradient
    
    def mapping(self, theta_hat, theta_1, r): ## mapping theta_hat to the ball of radius 2r around theta_1
        if torch.norm(theta_hat - theta_1) <= 2 * r:
            return theta_hat
        return theta_1 + 2 * r * (theta_hat - theta_1) / torch.norm(theta_hat - theta_1)
    
    def update_theta_ogd(self, clients, theta_1, r, learning_rate, lambda_reg):
        """Updates theta_sync using aggregated gradients until convergence."""
        total_gradient = self.aggregate_gradients_ogd(clients, self.theta_hat, lambda_reg)
        self.theta_hat = self.mapping(self.theta_hat - learning_rate * total_gradient, theta_1, r)


# 4. Federated Experiment Setup
def run_federated_experiment(num_iterations, feature_dim, num_arms, num_clients, noise_std=1, lambda_reg=1, delta=0.1, learning_rate=1e-4, alpha=50):
    # Initialize environment, clients, and server
    env = FederatedLinearDuelingBanditEnvironment(feature_dim, num_arms, num_clients, noise_std)
    clients = [FederatedClient_ogd(client_id=i, feature_dim=feature_dim, lambda_reg=lambda_reg) for i in range(num_clients)]
    server = CentralServer_ogd(feature_dim, num_clients, lambda_reg)

    cumulative_regret = []
    theta_diff = []
    theta_diff_l2 = []
    theta_diff_optimal = []
    theta_diff_optimal_ogd = []
    arm_diff = []
    total_regret = 0
    server.r = 10
    check_sum = 0
    for t in tqdm(range(1, num_iterations + 1)):
        # Generate contexts for all clients
        contexts = env.generate_context()

        gradients = []
        w_updates = []
        beta_T = torch.sqrt(2 * torch.log(torch.tensor(1.0 / delta, device="cuda"))+ feature_dim * torch.log(torch.tensor(1 + t * num_iterations * num_clients/(lambda_reg * feature_dim))))
        for client, context in zip(clients, contexts):
            # Compute beta_t
            beta_t = torch.sqrt(2 * torch.log(torch.tensor(1.0 / delta, device="cuda"))+ feature_dim * torch.log(torch.tensor(1 + t * num_clients/(lambda_reg * feature_dim))))
            

            # Client selects arms
            arm1_idx, arm2_idx = client.select_arms(server.theta_avg, context, beta_t, server.W_sync)
            arm1, arm2 = context[arm1_idx], context[arm2_idx]
            # Simulate feedback
            winner = env.get_preference(arm1, arm2)
            client.history_arm_1.append(arm1)
            client.history_arm_2.append(arm2)
            client.history_winner.append(winner)

            if t > 2:
                check_sum += torch.dot(arm1 - arm2, V_t_inv @ (arm1 - arm2)).item()
            w_updates.append(client.compute_w_update(arm1, arm2))

            # Compute regret
            best_arm_utility = torch.max(context @ env.theta_star).item()
            chosen_arm_utility = torch.dot(context[arm1_idx], env.theta_star).item() +  torch.dot(context[arm2_idx], env.theta_star).item()
            regret = 2 * best_arm_utility - chosen_arm_utility
            total_regret += regret / num_clients
    
        arm_diff.append(check_sum)
        server.aggregate_w_updates(w_updates)
        if t == 1:
            server.theta_sync = server.solve_theta(clients)
            server.theta_1 = server.theta_sync
            server.theta_hat = server.theta_sync
            server.theta_avg = server.theta_sync
        else:
            server.update_theta_ogd(clients, server.theta_1, server.r, 1/(alpha * (t - 1)), lambda_reg)

            
            server.theta_avg = ((t - 1) * server.theta_avg + server.theta_hat) / t

            theta_diff.append(torch.sqrt(torch.dot(env.theta_star - server.theta_avg, server.W_sync @ (env.theta_star - server.theta_avg))).item())
            theta_diff_l2.append(torch.norm(env.theta_star - server.theta_avg).item())
            V_t_inv = torch.linalg.inv(server.W_sync)
            # print(torch.norm(env.theta_star - server.theta_avg), torch.sqrt(torch.dot((env.theta_star - server.theta_avg) , server.W_sync @ (env.theta_star - server.theta_avg))), check_sum)

            # print(torch.norm(env.theta_star - server.theta_avg), torch.norm(env.theta_star - server.theta_sync),torch.norm(server.theta_sync - server.theta_avg), torch.sqrt(torch.dot((env.theta_star - server.theta_avg) , server.W_sync @ (env.theta_star - server.theta_avg))), torch.sqrt(torch.dot((env.theta_star - server.theta_sync) , server.W_sync @ (env.theta_star - server.theta_sync))))

        cumulative_regret.append(total_regret)

    # Save results
    output_dir = ""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "federated_cumulative_regret_ogd.npy"), np.array(cumulative_regret))


    df = pd.DataFrame({"iteration": list(range(1, len(cumulative_regret) + 1)), "cumulative_regret": cumulative_regret})
    df.to_csv(os.path.join(output_dir, "federated_cumulative_regret_ogd.csv"), index=False)

    return cumulative_regret, theta_diff, arm_diff, theta_diff_l2

# Example usage
if __name__ == "__main__":
    num_iterations = 500
    feature_dim = 5
    num_arms = 10
    num_clients = 50 #150

    cumulative_regret, theta_diff, arm_diff, theta_diff_l2 = run_federated_experiment(num_iterations, feature_dim, num_arms, num_clients, noise_std= 1, lambda_reg= 1/(num_iterations ) , delta=0.1, learning_rate=1e-4, alpha=1000)

    # Plot results
    import matplotlib.pyplot as plt


    
    plt.plot(cumulative_regret)
    plt.xlabel("Iterations")
    plt.ylabel("Cumulative Regret")
    plt.title("Federated Linear Dueling Bandit - Cumulative Regret")
    plt.show()
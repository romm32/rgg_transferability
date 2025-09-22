from torch_geometric.nn import LayerNorm, Sequential
from torch_geometric.nn.conv import MessagePassing

import random
import pickle
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TAGConv
from torch_geometric.nn import GCNConv
import os
import matplotlib.pyplot as plt
import wandb

import scipy.io

from gnn_roms import GNN1
from utils_santi import graphs_to_tensor
from utils_santi import get_gnn_inputs
from utils_santi import power_constraint
from utils_santi import get_rates, get_rates_roms
from utils_santi import objective_function
from utils_santi import mu_update
from collections import defaultdict

from utils_santi import graphs_to_tensor_synthetic


def run(dataset=990, b5g=False, num_channels=5, num_layers=5, K=3, batch_size=64, epocs=100, eps=5e-5, mu_lr=1e-4, synthetic=1, num_features=1, reinforce=True):   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("the current device is ", device)

    mu_k = torch.ones((1,1), requires_grad = False).to(device)
    epocs = epocs
    p0 = 5# # [48, 73, 96, 122] [5, 7, 10, 12]

    sigma = 1e-2

    input_dim = 1
    hidden_dim = num_features
    output_dim = 1
    num_layers = num_layers
    dropout = False
    K = K

    # gnn_model = GNN(input_dim, hidden_dim, output_dim, num_layers, dropout, K).to(device)
    num_features_list = [1] + [num_features]*num_layers # santi
    gnn_model = GNN1(num_features_list, 1, K=K).to(device)
    # gnn_model.train()
    optimizer = optim.Adam(gnn_model.parameters(), lr= mu_lr)
    
    if dataset == 0:
        dataset_name = 'dataset_noise_0.004_graphs_200.pt' #'scale3_1250_noise_0.004.pt' # # # #'dataset_noise_0.004_graphs_101_large.pt' ##'middlegraphs.pt'#
        num_channels = 484# # 729 #961 #1225 #484 # 
    elif dataset == 1:
        dataset_name = 'scale600dataset_noise_0.004_graphs_200.pt' 
        num_channels = 576
    elif dataset == 2:
        dataset_name = 'scale700dataset_noise_0.004_graphs_200.pt'
        num_channels = 676
    elif dataset == 3:
        dataset_name = 'scale801dataset_noise_0.004_graphs_200.pt' 
        num_channels = 841
    elif dataset == 4:
        dataset_name = 'scale900dataset_noise_0.004_graphs_200.pt' 
        num_channels = 900
    elif dataset == 5:
        dataset_name = 'scale1000dataset_noise_0.004_graphs_200.pt' 
        num_channels = 1024
    elif dataset == 6:
        dataset_name = 'scale1100dataset_noise_0.004_graphs_200.pt' 
        num_channels = 1089
    elif dataset == 7:
        dataset_name = 'scale1200dataset_noise_0.004_graphs_200.pt'
        num_channels = 1156
    pmax = num_channels
    print('dataset ', dataset_name)
    data = torch.load(os.path.join('data', dataset_name))
    train_data = data['train']
    dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
    dataloader_test = DataLoader(data['test'], batch_size=batch_size, shuffle=False, drop_last=True)
    exp_name = '{}_epochs_{}_lr_primal_{}_lr_dual_{}_hops_{}_num_features_{}_num_layers_{}'.format(
                    dataset_name[:8], epocs, mu_lr, eps, K, hidden_dim, num_layers)
    wandb.init(project="RGG-thebeginning", name=exp_name, 
                tags=[f"lr_dual_{eps}",
                        f"lr_primal_{mu_lr}"], config=args) 
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    results = defaultdict(list)

    for epoc in range(epocs):
        objective_function_values = []
        power_constraint_values = []
        loss_values = []
        mu_k_values = []
        normalized_psi_values = []
        transmitting = []
        gnn_model.train()
        if epoc%100 == 0:
            print("Epoc number: {}".format(epoc))
        for batch_idx, data in enumerate(dataloader_train):
            data = data.to(device)
            # print(data)
            channel_matrix_batch = data.H + torch.diag(data.x)
            channel_matrix_batch = channel_matrix_batch.view(batch_size, num_channels, num_channels).to(device)
            
            psi = gnn_model.forward(data.x, data.edge_index, data.edge_attr)
            psi = psi.squeeze(-1)
            psi = psi.view(batch_size, -1)
            psi = psi.unsqueeze(-1)
            
            normalized_psi = (torch.tanh(psi)*(0.99 - 0.01) + 1)/2 #SANTI
            normalized_psi_values.append(normalized_psi[0,:,:].squeeze(-1).detach().cpu().numpy())
            if reinforce:
                normalized_phi = torch.bernoulli(normalized_psi)
                log_p = normalized_phi * torch.log(normalized_psi) + (1 - normalized_phi) * torch.log(1 - normalized_psi)
                log_p_sum = torch.sum(log_p, dim=1)
                phi = normalized_phi * p0
            else:
                # normalized_phi = normalized_psi#(normalized_psi >= 0.5).float()
                # phi = normalized_phi*p0
                logits = torch.cat([torch.zeros_like(psi), psi], dim=-1)
                temperature = 0.5  # tune this, lower = closer to discrete
                gumbel_soft_samples = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
                phi = gumbel_soft_samples[..., 1]
                phi = phi * p0
                # print(phi)

            num_transmitting = torch.sum(gumbel_soft_samples[..., 1], dim=1)
            sum_phi = torch.sum(phi, dim=1)
            power_constr = pmax - sum_phi
            # power_constr_mean = torch.mean(power_constr, dim = 0).to(device)
            
            x = data.x.view(batch_size, num_channels, 1)
            phi = phi.view(batch_size, num_channels, 1)
            H = data.H.view(batch_size, num_channels, num_channels)
            rates = get_rates_roms(phi, x, sigma, H)
            sum_rate = torch.sum(rates, dim=1)
            sum_rate_mean = torch.mean(sum_rate, dim = 0)
            
            mu_k = mu_k.detach() - eps*power_constr.to(device)
            mu_k = torch.max(mu_k, torch.tensor(0.0)).to(device)
            cost = -(sum_rate.view(batch_size) + mu_k*power_constr.view(batch_size))
            if reinforce:
                loss = cost*log_p_sum
                loss_mean = torch.mean(loss, dim = 0)
            else:
                # penalty = torch.mean(4 * normalized_psi * (1 - normalized_psi))
                loss = (cost).view(batch_size) #- penalty
                # print(loss.shape)
                loss_mean = torch.mean(loss, dim = 0)
        
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()

            if batch_idx%10 == 0:
                power_constraint_values.append(-power_constr.detach().cpu().numpy())
                objective_function_values.append(sum_rate_mean.detach().cpu().numpy())
                loss_values.append(loss_mean.squeeze(-1).detach().cpu().numpy())
                mu_k_values.append(mu_k.squeeze(-1).detach().cpu().numpy())
                transmitting.append(num_transmitting.mean().detach().cpu().numpy())
            
        results['loss_train'].append(np.mean(np.array(loss_values)))#minimum_tx_constraint.detach().cpu().numpy())
        results['objective_function_train'].append(np.mean(np.array(objective_function_values)))#obj_function.detach().cpu().numpy())
        results['power_constraint_train'].append(np.mean(np.array(power_constraint_values)))#obj_function.detach().cpu().numpy())
        results['mu_values_train'].append(np.mean(np.array(mu_k_values)))

        # wandb.log({
        #     })
        
        objective_function_values_test = []
        power_constraint_values_test = []
        normalized_psi_values_test = []
        transmitting_test = []
        gnn_model.eval()
        for batch_idx, data in enumerate(dataloader_test):
            data = data.to(device)
            channel_matrix_batch = data.H + torch.diag(data.x)
            channel_matrix_batch = channel_matrix_batch.view(batch_size, num_channels, num_channels).to(device)

            psi = gnn_model.forward(data.x, data.edge_index, data.edge_attr)
            psi = psi.squeeze(-1)
            psi = psi.view(batch_size, -1)
            psi = psi.unsqueeze(-1)
            
            normalized_psi = (torch.tanh(psi)*(0.99 - 0.01) + 1)/2 #SANTI
            normalized_psi_values_test.append(normalized_psi[0,:,:].squeeze(-1).detach().cpu().numpy())
            if reinforce:
                normalized_phi = torch.bernoulli(normalized_psi)
                log_p = normalized_phi * torch.log(normalized_psi) + (1 - normalized_phi) * torch.log(1 - normalized_psi)
                log_p_sum = torch.sum(log_p, dim=1)
                phi = normalized_phi * p0
            else:
                # normalized_phi = normalized_psi#(normalized_psi >= 0.5).float()
                # phi = normalized_phi*p0
                logits = torch.cat([torch.zeros_like(psi), psi], dim=-1)
                temperature = 0.5  # tune this, lower = closer to discrete
                gumbel_soft_samples = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
                phi = gumbel_soft_samples[..., 1]
                phi = phi * p0

            num_transmitting = torch.sum(gumbel_soft_samples[..., 1], dim=1)
            sum_phi = torch.sum(phi, dim=1)
            power_constr = pmax - sum_phi
            power_constr_mean = torch.mean(power_constr, dim = 0).to(device)
            
            x = data.x.view(batch_size, num_channels, 1)
            phi = phi.view(batch_size, num_channels, 1)
            H = data.H.view(batch_size, num_channels, num_channels)
            rates = get_rates_roms(phi, x, sigma, H)
            sum_rate = torch.sum(rates, dim=1)
            sum_rate_mean = torch.mean(sum_rate, dim = 0)
            
            if batch_idx%10 == 0:
                power_constraint_values_test.append(-power_constr_mean.detach().cpu().numpy())
                objective_function_values_test.append(sum_rate_mean.detach().cpu().numpy())
                transmitting_test.append(num_transmitting.mean().detach().cpu().numpy())
            
        # results['loss']['tes'].append(np.mean(np.array(loss_values)))#minimum_tx_constraint.detach().cpu().numpy())
        results['objective_function_test'].append(np.mean(np.array(objective_function_values_test)))#obj_function.detach().cpu().numpy())
        results['power_constraint_test'].append(np.mean(np.array(power_constraint_values_test)))#obj_function.detach().cpu().numpy())
        # results['mu_values'].append(np.mean(np.array(mu_k_values)))
        wandb.log({
            "Train/Lagrangian": np.mean(np.array(loss_values)),
                    "Train/Objective function": np.mean(np.array(objective_function_values)),
                    "Train/Power constraint": np.mean(np.array(power_constraint_values)),
                    "Train/mu": np.mean(np.array(mu_k_values)),
                    "Train/Transmitters": np.mean(np.array(transmitting)),
                    "Test/Objective function": np.mean(np.array(objective_function_values_test)),
                    "Test/Power constraint": np.mean(np.array(power_constraint_values_test)),
                    # "Train/mu": np.mean(np.array(mu_k_values)),
                    "Test/Transmitters": np.mean(np.array(transmitting_test))
            })
        
        torch.save(results, './results/{}.json'.format(exp_name))
        torch.save(gnn_model.state_dict(), './models/{}.pt'.format(exp_name))

    wandb.finish()
    print('experiment finished')     
        
if __name__ == '__main__':
    import argparse

    torch.manual_seed(32)
    np.random.seed(32) 

    parser = argparse.ArgumentParser(description= 'System configuration')

    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--b5g', type=int, default=0)
    parser.add_argument('--num_channels', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_features', type=int, default=8)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_dual', type=float, default=1e-5)
    parser.add_argument('--lr_primal', type=float, default=1e-2)
    parser.add_argument('--synthetic', type=int, default=0)
    parser.add_argument('--reinforce', type=int, default=0)
    
    args = parser.parse_args()
    
    print(args.reinforce)
    run(dataset=args.dataset, b5g=args.b5g, num_channels=args.num_channels, num_layers=args.num_layers, K=args.K, batch_size=args.batch_size, epocs=args.epochs, eps=args.lr_dual, mu_lr=args.lr_primal, synthetic=args.synthetic, num_features=args.num_features, reinforce=args.reinforce)
    # print('Seeds: {} and {}'.format(rn, rn1))

import random
import os
import numpy as np
import torch
import networkx as nx
from collections import defaultdict
from collections import Counter

from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RadiusGraph
from torch_geometric.transforms import RemoveIsolatedNodes
from torch_geometric.data import Data, Dataset

class PAWLDataset(Dataset):
    # Define a basic dataset
    def __init__(self, data_list):
        super().__init__(None, None, None)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx], idx
    
class SAWLData(Data):
    def __init__(self,
                 x=None,
                 edge_index=None,
                 H=None,
                 edge_attr=None):
        super().__init__()
        self.x = x
        self.edge_index = edge_index
        self.H = H
        self.edge_attr = edge_attr

def create_matrix(pos, edge_index, test=False):
    m = pos.size(0)
    src, dst = edge_index
    distances = (pos[src] - pos[dst]).norm(p=2, dim=1)
    path_loss = distances**(-2)
    fading_gain = torch.distributions.Exponential(1.0).sample(sample_shape=distances.shape)
    if test:
        print('first fg', fading_gain)
    edge_attr = path_loss*fading_gain
    
    ## So far I have the adjacency matrix. I now need to find the receiver for each
    ## transmitter such that I can construct the graph signal.
    neighbors = defaultdict(list)
    channel = defaultdict(list)
    for i in range(edge_attr.size(0)):
        u = src[i].item()
        v = dst[i].item()
        d = edge_attr[i].item()
        neighbors[u].append(v)
        channel[u].append(d)
        # neighbors[v].append(u)
        # channel[v].append(d)

    receiver = defaultdict(list)
    x = torch.zeros((m,1))
    for i in range(m):
        valid = [(j, g) for j, g in zip(neighbors[i], channel[i]) if j != i]
        if valid:
            j, gain = random.choice(valid)
            x[i] = gain
            receiver[i].append(j)
    
    if test:
        print("Receiver assignment", receiver)

    edge_attr[torch.isnan(edge_attr)] = 0
    edge_attr[torch.isinf(edge_attr)] = 0
    x[torch.isnan(x)] = 0
    x[torch.isinf(x)] = 0
    degree_counts = Counter(len(set(neighbors[i])) for i in range(m))
    degree_vector = torch.zeros(11, dtype=torch.int)
    for deg in range(11):
        degree_vector[deg] = degree_counts[deg]
    # print(path_losses)
    return edge_index, edge_attr, x, degree_vector

def build_adj_matrix(edge_index, edge_attr, n):
    adj = torch.zeros(n, n)
    src, dst = edge_index
    adj[src, dst] = edge_attr
    return adj

def create_dataset(num_samples, nodes,
                    r, batch_size,
                    noise,
                    test=False, one_graph=False, diverse=True):
    
    data, data_train, data_test = [], [], []
    H, H_train, H_test = [], [], []
    number_of_graphs = num_samples['train'] + num_samples['test']

    another_transform = RemoveIsolatedNodes()
    if not diverse:
        count = 0
        degrees = []
        # sample node positions
        world_size = np.random.choice(nodes)
        num_nodes = world_size**2
        grid_dim  = world_size#math.ceil(math.sqrt(num_nodes))  
        grid_rows = grid_cols = grid_dim
        cell_w = world_size / grid_cols
        cell_h = world_size / grid_rows
        centers = np.array([[(j + .5) * cell_w, (i + .5) * cell_h]
                            for i in range(grid_rows) for j in range(grid_cols)])
        idx = np.random.choice(len(centers), size=num_nodes, replace=False)
        positions = centers[idx] + np.random.normal(scale=noise*world_size,
                                                    size=(num_nodes, 2))
        pos = torch.from_numpy(positions).float()

        # create edges of the graph based on a threshold r
        x = torch.ones(num_nodes, 1, dtype=torch.long)
        edge_index = torch.tensor([], dtype=torch.long)
        radius = r*world_size/grid_cols
        transform = RadiusGraph(radius, loop=False)
        graph_i = transform(Data(x=x, edge_index=edge_index, pos=pos))
        data_transformed_i = another_transform(graph_i)

        # verify that the graph is connected
        G = to_networkx(data_transformed_i, to_undirected=True)        
        is_connected = nx.is_connected(G)
        if is_connected:
            while count < number_of_graphs:
                # compute the channel matrix and store graph with the edges from data_transformed_i
                if not one_graph or count == 0:
                    edge_index, edge_attr, x, degrees_i = create_matrix(pos, data_transformed_i.edge_index, test=test)
                    data_i = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

                degrees.append(degrees_i)
                data.append(data_i)
                H_i = build_adj_matrix(edge_index, edge_attr, num_nodes)
                H.append(H_i)
                count = count + 1
                if count%10 == 0:
                    print(count)
    else:
        total_count = 0
        degrees = []
        while total_count < number_of_graphs:
            # sample node positions
            count = 0
            world_size = np.random.choice(nodes)
            num_nodes = world_size**2
            grid_dim  = world_size#math.ceil(math.sqrt(num_nodes))  
            grid_rows = grid_cols = grid_dim
            cell_w = world_size / grid_cols
            cell_h = world_size / grid_rows
            centers = np.array([[(j + .5) * cell_w, (i + .5) * cell_h]
                                for i in range(grid_rows) for j in range(grid_cols)])
            idx = np.random.choice(len(centers), size=num_nodes, replace=False)
            positions = centers[idx] + np.random.normal(scale=noise*world_size,
                                                        size=(num_nodes, 2))
            pos = torch.from_numpy(positions).float()

            # create edges of the graph based on a threshold r
            x = torch.ones(num_nodes, 1, dtype=torch.long)
            edge_index = torch.tensor([], dtype=torch.long)
            radius = r*world_size/grid_cols
            transform = RadiusGraph(radius, loop=False)
            graph_i = transform(Data(x=x, edge_index=edge_index, pos=pos))
            data_transformed_i = another_transform(graph_i)

            # verify that the graph is connected
            G = to_networkx(data_transformed_i, to_undirected=True)        
            is_connected = nx.is_connected(G)
            if is_connected:
                while count < batch_size:
                    # compute the channel matrix and store graph with the edges from data_transformed_i
                    if not one_graph or count == 0:
                        edge_index, edge_attr, x, degrees_i = create_matrix(pos, data_transformed_i.edge_index, test=test)
                        data_i = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

                    degrees.append(degrees_i)
                    data.append(data_i)
                    H_i = build_adj_matrix(edge_index, edge_attr, num_nodes)
                    H.append(H_i)
                    count = count + 1
                    total_count = total_count + 1
                    if count%10 == 0:
                        print(count)
                        
    for i in range(number_of_graphs):
        if i < num_samples['train']:
            data_train.append(data[i % number_of_graphs])
            H_train.append(H[i % number_of_graphs])
        else:
            data_test.append(data[i % number_of_graphs])
            H_test.append(H[i % number_of_graphs])

    pawl_data_train = []
    pawl_data_test = []
    for i in range(len(data_train)):
        H = H_train[i]
        x = data_train[i].x
        edge_index = data_train[i].edge_index
        edge_attr = data_train[i].edge_attr
        pawl_data_train.append(SAWLData(x=x, edge_index=edge_index, edge_attr=edge_attr, H=H))            
    for i in range(len(data_test)):
        H = H_test[i]
        x = data_test[i].x
        edge_index = data_test[i].edge_index
        edge_attr = data_test[i].edge_attr
        pawl_data_test.append(SAWLData(x=x, edge_index=edge_index, edge_attr=edge_attr, H=H))            

    loader = {}
    dataset_train = PAWLDataset(pawl_data_train)
    loader['train'] = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    dataset_test = PAWLDataset(pawl_data_test)
    loader['test'] = DataLoader(dataset=dataset_test, batch_size=batch_size)
    dataset_to_save = {'train': pawl_data_train, 'test': pawl_data_test}
    os.makedirs('data', exist_ok=True)
    if not one_graph:
        if not diverse:
            torch.save(dataset_to_save, os.path.join('data', 'scaledataset_noise_'+str(noise)+'_graphs_'+ str(number_of_graphs) +'.pt'))
        else:
            torch.save(dataset_to_save, os.path.join('data', 'scaledataset_noise_'+str(noise)+'_graphs_'+ str(number_of_graphs) +'_diverse.pt'))
    else:
        torch.save(dataset_to_save, os.path.join('data', 'dataset_noise_'+str(noise)+'_one_graph_.pt'))
    
    average_degree_vector = torch.stack(degrees).float().mean(dim=0)
    return loader, average_degree_vector
from argparse import Namespace
import os.path as osp
from pathlib import Path
import numpy as np

import torch
from torch_geometric.datasets import PPI, Amazon
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import (InMemoryDataset, Data, DataLoader)
import torch_geometric.transforms as T
import torch.utils.data as data
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import random

class ProcessedDataset(InMemoryDataset):
    pass



class identity(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, split, rank, vary=False):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.h = rank
        self.w = rank
        self.vary = vary
        self.rank = rank

        self.split = split
        self.inp_dim = self.h * self.w
        self.out_dim = self.h * self.w

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        if self.vary:
            rank = random.randint(2, self.rank)
        else:
            rank = self.rank

        R = np.random.uniform(-1, 1, (rank, rank))
        R_corrupt = R

        repeat = 128 // self.w + 1

        R_tile = np.tile(R, (1, repeat))

        node_features = torch.Tensor(np.zeros((rank, 1)))
        edge_index = torch.LongTensor(np.array(np.mgrid[:rank, :rank]).reshape((2, -1)))
        edge_features_context = torch.Tensor(np.tile(R_corrupt.reshape((-1, 1)), (1, 1)))
        edge_features_label = torch.Tensor(R.reshape((-1, 1)))
        noise = torch.Tensor(edge_features_label.reshape((-1, 1)))

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features_context, y=edge_features_label, noise=noise)

        return data

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return int(1e6)


class shortest(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, split, rank, vary=False):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.h = rank
        self.w = rank
        self.vary = vary
        self.rank = rank

        self.split = split
        self.inp_dim = self.h * self.w
        self.out_dim = self.h * self.w

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        if self.vary:
            rank = random.randint(2, self.rank)
        else:
            rank = self.rank

        graph = np.random.uniform(0, 1, size=[rank, rank])
        graph = graph + graph.transpose()
        np.fill_diagonal(graph, 0) # can move to self

        graph_dist, graph_predecessors = shortest_path(csgraph=csr_matrix(graph), unweighted=False, directed=False, return_predecessors=True)

        node_features = torch.Tensor(np.zeros((rank, 1)))
        edge_index = torch.LongTensor(np.array(np.mgrid[:rank, :rank]).reshape((2, -1)))

        edge_features_context = torch.Tensor(np.tile(graph.reshape((-1, 1)), (1, 1)))
        edge_features_label = torch.Tensor(graph_dist.reshape((-1, 1)))
        noise = torch.Tensor(edge_features_label.reshape((-1, 1)))
        #print(edge_features_context.shape)
        #print(edge_features_label.shape)
        #print(noise.shape)
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features_context, y=edge_features_label, noise=noise)

        return data
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return int(1e6)
    
    
    
class connected_comp(data.Dataset):
    """Constructs a dataset with N circles, N is the number of components set by flags"""

    def __init__(self, split, rank, vary=False):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.h = rank
        self.w = rank
        self.rank = rank
        self.vary = vary

        self.split = split
        self.inp_dim = self.h * self.w
        self.out_dim = self.h * self.w

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """

        if self.vary:
            rank = random.randint(2, self.rank)
        else:
            rank = self.rank

        R = np.random.uniform(0, 1, (rank, rank))
        connections = R > 0.95
        components, component_arr = connected_components(connections, directed=False)

        label = np.zeros((rank, rank))

        for i in range(components):
            component_i = np.arange(rank)[component_arr == i]

            for j in range(component_i.shape[0]):
                for k in range(component_i.shape[0]):
                    label[component_i[j], component_i[k]] = 1

        node_features = torch.Tensor(np.zeros((rank, 1)))
        edge_index = torch.LongTensor(np.array(np.mgrid[:rank, :rank]).reshape((2, -1)))

        edge_features_context = torch.Tensor(np.tile(connections.reshape((-1, 1)), (1, 1)))
        edge_features_label = torch.Tensor(label.reshape((-1, 1)))
        noise = torch.Tensor(label.reshape((-1, 1)))

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features_context, y=edge_features_label, noise=noise)

        return data

    def __len__(self):
        """Return the total number of images in the dataset."""
        # Dataset is always randomly generated
        return int(1e6)
import csv
import numpy as np
import torch
from typing import Tuple

from FunctionEncoder.Dataset.BaseDataset import BaseDataset


class WarthogDataset(BaseDataset):

    def __init__(self,
                 csv_file: str,
                 device: str = "auto",
                 dtype: torch.dtype = torch.float32,
                 n_functions:int=None,
                 n_examples:int=None,
                 n_queries:int=None,
                

                 # deprecated arguments
                 n_functions_per_sample:int = None,
                 n_examples_per_sample:int = None,
                 n_points_per_sample:int = None,

                 ):
        # default arguments. These default arguments will be placed in the constructor when the arguments are deprecated.
        # but for now they live here.
        if n_functions is None and n_functions_per_sample is None:
            n_functions = 1
        if n_examples is None and n_examples_per_sample is None:
            n_examples = 100 #1000
        if n_queries is None and n_points_per_sample is None:
            n_queries = 1000 #10000
        
        super().__init__(input_size=(14,), # time, (x,y,z) position, (x,y,z,w) orientation, linear (x,y,z), angular (x,y,z)
                         output_size=(7,), # (x,y,z) position, (x,y,z,w) orientation 

                         data_type="deterministic",
                         device=device,
                         dtype=dtype,
                         n_functions=n_functions,
                         n_examples=n_examples,
                         n_queries=n_queries,
                         )

        # load data from CSV files
        data = self.process_csv(csv_file)
        self.train_data = data[:20000,:]
        self.test_data = data[20000:,:]
        # print(self.train_data.shape)

    def sample(self) -> Tuple[  torch.tensor, # example inputs (1, 1000, 14)
                                torch.tensor, # example outputs (1, 1000, 7)
                                torch.tensor, # query inputs (1, 10000, 14)
                                torch.tensor, # query outputs (1, 10000, 7)
                                dict]: # dictionary of function parameters
        with torch.no_grad():
            # Get random indices from the data tensor.
            ex_indices = torch.randperm(self.train_data.size(0))[:self.n_examples]
            qu_indices = torch.randperm(self.train_data.size(0))[:self.n_queries]

            # Sample random rows from the data tensor. 
            ex_subset = self.train_data[ex_indices]
            qu_subset = self.train_data[qu_indices]

            # Parse out the input and output data.
            example_xs = ex_subset[:,:14].unsqueeze(dim=0)
            example_ys = ex_subset[:,14:].unsqueeze(dim=0)
            query_xs = qu_subset[:,:14].unsqueeze(dim=0)
            query_ys = qu_subset[:,14:].unsqueeze(dim=0)
            
            # Sample saved data instead of generating new data. 
            # print("example_xs: ", example_xs.shape)
            # print("example_ys: ", example_ys.shape)
            # print("query_xs: ", query_xs.shape)
            # print("query_ys: ", query_ys.shape)

            return example_xs, example_ys, query_xs, query_ys, {"mus":None}
        
    def sample_test(self) -> Tuple[  torch.tensor, 
                                torch.tensor, 
                                torch.tensor, 
                                torch.tensor, 
                                dict]: 
        with torch.no_grad():
            # Get random indices from the data tensor.
            ex_indices = torch.randperm(self.test_data.size(0))[:self.n_examples]
            qu_indices = torch.randperm(self.test_data.size(0))[:self.n_queries]

            # Sample random rows from the data tensor. 
            ex_subset = self.test_data[ex_indices]
            qu_subset = self.test_data[qu_indices]

            # Parse out the input and output data.
            example_xs = ex_subset[:,:14].unsqueeze(dim=0)
            example_ys = ex_subset[:,14:].unsqueeze(dim=0)
            query_xs = qu_subset[:,:14].unsqueeze(dim=0)
            query_ys = qu_subset[:,14:].unsqueeze(dim=0)
            
            # Sample saved data instead of generating new data. 
            # print("example_xs: ", example_xs.shape)
            # print("example_ys: ", example_ys.shape)
            # print("query_xs: ", query_xs.shape)
            # print("query_ys: ", query_ys.shape)

            return example_xs, example_ys, query_xs, query_ys, {"mus":None}


    def process_csv(self, csv_file):
        # Load CSV into a numpy array, then convert to torch tensor. 
        array = np.loadtxt(csv_file, delimiter=',')
        tensor = torch.tensor(array, device=self.device).to(torch.float32)
        # Convert the time stamps column to changes in time. 
        tensor[:-1, 0] = tensor[1:, 0] - tensor[:-1, 0]
        # Get the "next" states from the data. 
        next_states = tensor[1:, 1:8]
        # Remove the bottom row from the data. 
        new_tensor = tensor[:-1,:]
        # Append the "next" states to the tensor. 
        return torch.cat((new_tensor, next_states), dim=1)
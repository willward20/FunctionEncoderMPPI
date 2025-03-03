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
        
        super().__init__(input_size=(4,), # del_time, states (yaw), controls (lin x, ang z vel)
                         output_size=(3,), # next states (x, y, yaw)

                         data_type="deterministic",
                         device=device,
                         dtype=dtype,
                         n_functions=n_functions,
                         n_examples=n_examples,
                         n_queries=n_queries,
                         )

        # load data from CSV files
        data = self.process_csv(csv_file)
        self.train_data = data[:10000,:]
        self.test_data = data[10000:,:]
        # print(self.train_data.shape)

    def sample(self) -> Tuple[  torch.tensor, # example inputs (1, n_examples, input_size)
                                torch.tensor, # example outputs (1, n_examples, output_size)
                                torch.tensor, # query inputs (1, n_queries, input_size)
                                torch.tensor, # query outputs (1, n_queries, output_size)
                                dict]: # dictionary of function parameters
        with torch.no_grad():
            # Get random indices from the data tensor.
            ex_indices = torch.randperm(self.train_data.size(0))[:self.n_examples]
            qu_indices = torch.randperm(self.train_data.size(0))[:self.n_queries]

            # Sample random rows from the data tensor. 
            ex_subset = self.train_data[ex_indices]
            qu_subset = self.train_data[qu_indices]

            # Parse out the input and output data.
            example_xs = ex_subset[:,:4].unsqueeze(dim=0)
            example_ys = ex_subset[:,4:].unsqueeze(dim=0)
            query_xs = qu_subset[:,:4].unsqueeze(dim=0)
            query_ys = qu_subset[:,4:].unsqueeze(dim=0)
            
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
            example_xs = ex_subset[:,:4].unsqueeze(dim=0)
            example_ys = ex_subset[:,4:].unsqueeze(dim=0)
            query_xs = qu_subset[:,:4].unsqueeze(dim=0)
            query_ys = qu_subset[:,4:].unsqueeze(dim=0)
            
            # Sample saved data instead of generating new data. 
            # print("example_xs: ", example_xs.shape)
            # print("example_ys: ", example_ys.shape)
            # print("query_xs: ", query_xs.shape)
            # print("query_ys: ", query_ys.shape)

            return example_xs, example_ys, query_xs, query_ys, {"mus":None}


    def process_csv(self, csv_file):
        # Load CSV into a numpy array. 
        array = np.loadtxt(csv_file, delimiter=',')
        # Unwrap the yaw meaurements.
        array[:, 3] = np.unwrap(array[:,3])
        # Convert numpy array to torch tensor. 
        tensor = torch.tensor(array, device=self.device).to(torch.float32)
        # Convert the time stamps column to changes in time. 
        tensor[:-1, 0] = tensor[1:, 0] - tensor[:-1, 0]
        # Get the change in states from the data. 
        del_states = tensor[1:, 1:4] - tensor[:-1, 1:4]
        # Remove the bottom row from the data. 
        new_tensor = tensor[:-1,:]
        # Remove the xPos and yPos from the data.
        final_tensor = new_tensor[:,[0,3,4,5]]
        # Append the change in states to the tensor
        return torch.cat((final_tensor, del_states), dim=1)
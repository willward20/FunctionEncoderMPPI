import csv
import numpy as np
import torch
from typing import Tuple
import matplotlib.pyplot as plt

from FunctionEncoder.Dataset.BaseDataset import BaseDataset


class WarthogDatasetFull2D(BaseDataset):

    def __init__(self,
                 odom_csv: str,
                 cmdvel_csv: str,
                 device: str = "auto",
                 dtype: torch.dtype = torch.float32,
                 n_functions:int=1,
                 n_examples:int=100, # 1000
                 n_queries:int=1000, # 10000
                 ):
        
        super().__init__(input_size=(9,), # del_time, states (x, y, yaw, x_vel, y_vel, ang_vel), controls (lin x, ang z vel)
                         output_size=(6,), # change in states (x, y, yaw, x_vel, y_vel, ang_vel)

                         data_type="deterministic",
                         device=device,
                         dtype=dtype,
                         n_functions=n_functions,
                         n_examples=n_examples,
                         n_queries=n_queries,
                         )

        # load data from CSV files
        data = self.process_csvs(odom_csv, cmdvel_csv)
        print("data: ", data.shape)

        # Shuffle indices
        num_samples = data.shape[0]
        indices = torch.randperm(num_samples)  # Generate a random permutation of indices

        # Split into 90% train and 10% test
        split = round(num_samples * 0.9)
        train_indices = indices[:split]
        test_indices = indices[split:]

        self.train_data = data[train_indices, :]
        self.test_data = data[test_indices, :]

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
            example_xs = ex_subset[:,:9].unsqueeze(dim=0)
            example_ys = ex_subset[:,9:].unsqueeze(dim=0)
            query_xs = qu_subset[:,:9].unsqueeze(dim=0)
            query_ys = qu_subset[:,9:].unsqueeze(dim=0)

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
            example_xs = ex_subset[:,:9].unsqueeze(dim=0)
            example_ys = ex_subset[:,9:].unsqueeze(dim=0)
            query_xs = qu_subset[:,:9].unsqueeze(dim=0)
            query_ys = qu_subset[:,9:].unsqueeze(dim=0)

            return example_xs, example_ys, query_xs, query_ys, {"mus":None}


    def process_csvs(self, odom_csv, cmdvel_csv):
        # Load odom CSV into a numpy array. 
        odom_array = np.loadtxt(odom_csv, delimiter=',')
        # Unwrap the yaw meaurements.
        odom_array[:, 3] = np.unwrap(odom_array[:,3])
        # Convert numpy array to torch tensor. 
        odom_tensor = torch.tensor(odom_array, device=self.device).to(torch.float32)
        # Convert the time stamps column to changes in time. 
        odom_tensor[:-1, 0] = odom_tensor[1:, 0] - odom_tensor[:-1, 0]

        # Get the change in position/heading at each time expressed in the inertial frame. 
        del_pose_I = odom_tensor[1:, 1:4] - odom_tensor[:-1, 1:4]
        # Rotate the position/heading data from the inertial
        # frame and into the initial body frame.  
        del_pose_Bi = self.inertial_to_body(
            odom_tensor[:-1, 3], # don't need the final data point
            del_pose_I
        )

        # Get the velcoities at the next state in the final body frame.
        # For now, I'm not transforming because that's complicated and
        # may not be neccessary.  
        vel_next_Bf = odom_tensor[1:,4:]

        # Rotate the velocities of the next body frame into the inertial frame. 
        vel_next_I = self.vel_body_to_inertial(
            odom_tensor[1:, 3],  # yaw angles of the next body frame relative to I 
            vel_next_Bf, # velocities of the next body frame
        )
        # Rotate the velocities from the inertial frame to the initial body frame. 
        vel_next_Bi = self.inertial_to_body(
            odom_tensor[:-1, 3],  # yaw angles of the initial frame relative to I
            vel_next_I, # velocities of the next body frame relative to I, expressed in I
        )
        # These two rotations gives me the velocity of the next body frame
        # relative to the inertial frame but express in the initial body frame. 
        # Now, take the difference between the velocities in Bi. 
        del_vel_Bi = vel_next_Bi - odom_tensor[:-1,4:]


        # Zero out the initial pose to put it into the body frame.
        odom_tensor[:,1:4] = torch.zeros(odom_tensor.shape[0], 3)
        # Remove the bottom row from the data. 
        odom_tensor = odom_tensor[:-1,:]

        # Plot the scatter plot
        # del_states_np = del_states.cpu().numpy()
        # plt.scatter(del_states_np[:, 0], del_states_np[:, 1], c='blue', alpha=0.6, edgecolors='k')
        # plt.xlabel("X-axis")
        # plt.ylabel("Y-axis")
        # plt.title("Scatter Plot of 2D Torch Tensor")
        # plt.grid(True)
        # plt.show()

        # Load cmd_vel CSV into a numpy array. 
        cmdvel_array = np.loadtxt(cmdvel_csv, delimiter=',')
        # Find the closest cmd_vel timestamps to the odom timestamps.
        indices = np.abs(cmdvel_array[:, 0, None] - odom_array[:, 0]).argmin(axis=0)
        # Filter the cmd_vel data to only keep closest values. 
        cmdvel_array = cmdvel_array[indices,:]
        # Convert numpy array to torch tensor. 
        cmdvel_tensor = torch.tensor(cmdvel_array, device=self.device).to(torch.float32)
        # Remove the time stamps from the cmd_vel data.  
        cmdvel_tensor = cmdvel_tensor[:, [1,2]]
        # Remove the bottom row from the data. 
        cmdvel_tensor = cmdvel_tensor[:-1,:]

        # Append states, inputs, and changes in states to one tensor. 
        return torch.cat((odom_tensor, cmdvel_tensor, del_pose_Bi, del_vel_Bi), dim=1)
    
    def inertial_to_body(
            self, 
            yaws,    # (T x 1) rotation of the body frame w.r.t inertial frame
            xIMat,    # (T x 3) matrix of vectors in the inertial frame
    ):
        """ Transforms inertial frame vectors into the body frame. """

        cos_yaw = torch.cos(yaws)
        sin_yaw = torch.sin(yaws)
        zeros = torch.zeros(yaws.shape[0], device=self.device)
        ones = torch.ones(yaws.shape[0], device=self.device)

        # Construct the batch of rotation matrices
        R = torch.stack([
            torch.stack([cos_yaw, sin_yaw, zeros], dim=1),
            torch.stack([-sin_yaw, cos_yaw, zeros], dim=1),
            torch.stack([zeros, zeros, ones], dim=1)
        ], dim=1)  # Shape: (N, 3, 3)

        # Perform batch matrix-vector multiplication
        xBMat = torch.bmm(R, (xIMat).unsqueeze(-1)).squeeze(-1)  # Shape: (T, 3)
        return xBMat
    
    def vel_body_to_inertial(
            self, 
            yaws,    # (T x 1) rotation of the body frame w.r.t. inertial frame. 
            xIMat,    # (T x 3) matrix of vectors in the body frame
    ):
        """ Transforms the velocity of the body frame vectors into the inertial frame. """

        cos_yaw = torch.cos(yaws)
        sin_yaw = torch.sin(yaws)
        zeros = torch.zeros(yaws.shape[0], device=self.device)
        ones = torch.ones(yaws.shape[0], device=self.device)

        # Construct the batch of rotation matrices
        R = torch.stack([
            torch.stack([cos_yaw, -sin_yaw, zeros], dim=1),
            torch.stack([sin_yaw, cos_yaw, zeros], dim=1),
            torch.stack([zeros, zeros, ones], dim=1)
        ], dim=1)  # Shape: (N, 3, 3)

        # Perform batch matrix-vector multiplication
        xBMat = torch.bmm(R, (xIMat).unsqueeze(-1)).squeeze(-1)  # Shape: (T, 3)
        return xBMat
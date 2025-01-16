import torch
from typing import Tuple

from FunctionEncoder.Dataset.BaseDataset import BaseDataset


class PathtrackingDataset(BaseDataset):

    def __init__(self,
                 mu_range = (0.0, 0.1), # mu coef. range
                 pos_range = (-40.0, 40.0), # x state range
                 yaw_range = (-2*torch.pi, 2*torch.pi), # yaw range
                 v_range = (0.0, 50.0), # velocity range
                 accel_range = (-2.0, 2.0),
                 steer_range = (-0.523, 0.523),
                 t_dif = 0.05,
                 wheel_base = 2.5,
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
            n_functions = 10
        if n_examples is None and n_examples_per_sample is None:
            n_examples = 100 #1000
        if n_queries is None and n_points_per_sample is None:
            n_queries = 1000 #10000
        
        super().__init__(input_size=(7,), # time, x, y, yaw, v, steer, accel
                         output_size=(4,), # x, y, yaw, v 

                         data_type="deterministic",
                         device=device,
                         dtype=dtype,
                         n_functions=n_functions,
                         n_examples=n_examples,
                         n_queries=n_queries,
                         )
        self.mu_range = torch.tensor(mu_range, device=self.device, dtype=self.dtype)
        self.pos_range = torch.tensor(pos_range, device=self.device, dtype=self.dtype)
        self.yaw_range = torch.tensor(yaw_range, device=self.device, dtype=self.dtype)
        self.v_range = torch.tensor(v_range, device=self.device, dtype=self.dtype)
        self.accel_range = torch.tensor(accel_range, device=self.device, dtype=self.dtype)
        self.steer_range = torch.tensor(steer_range, device=self.device, dtype=self.dtype)
        self.t_dif = t_dif
        self.wheel_base = wheel_base

    def sample(self) -> Tuple[  torch.tensor, # example inputs (10, 1000, 3)
                                torch.tensor, # example outputs (10, 1000, 2)
                                torch.tensor, # query inputs (10, 10000, 3)
                                torch.tensor, # query outputs (10, 10000, 2)
                                dict]: # dictionary of function parameters
        with torch.no_grad():
            # Generate random states (including initial controls, but not including
            # time
            query_xs = self.get_input_points(self.n_queries)
            example_xs = self.get_input_points(self.n_examples)

            # Create the pathtracking model function and generate the mus
            pathtracking_models, mus = self.get_pathtracking_models()

            # Propagate the states forward using the dynamics.
            query_ys = pathtracking_models(query_xs) 
            example_ys = pathtracking_models(example_xs) 

            return example_xs, example_ys, query_xs, query_ys, {"mus":mus}

    def get_input_points(self, n_samples):
        """
        Return a tensor of inputs values (not including time) for each
        mu function and for each example/query.
        """
        # create vector of time values 
        t_difs = self.t_dif * torch.ones((self.n_functions, n_samples), dtype=self.dtype, device=self.device)
       
        # generate random states and control inputs
        xs = torch.rand((self.n_functions, n_samples, 6), dtype=self.dtype, device=self.device)
        
        # constrain each set of inputs within their respective ranges
        x = xs[:,:,0] * (self.pos_range[1] - self.pos_range[0]) + self.pos_range[0]
        y = xs[:,:,1] * (self.pos_range[1] - self.pos_range[0]) + self.pos_range[0]
        yaw = xs[:,:,2] * (self.yaw_range[1] - self.yaw_range[0]) + self.yaw_range[0]
        v = xs[:,:,3] * (self.v_range[1] - self.v_range[0]) + self.v_range[0]
        steer = xs[:,:,4] * (self.steer_range[1] - self.steer_range[0]) + self.steer_range[0]
        accel = xs[:,:,5] * (self.accel_range[1] - self.accel_range[0]) + self.accel_range[0]
        
        # merge the time, inputs, and contorls back into one tensor
        return torch.stack([t_difs, x, y, yaw, v, steer, accel], dim=2)

    def get_pathtracking_models(self):
        # Generate the random parameters for each function. 
        mus = torch.rand(self.n_functions, 1, device=self.device) * (self.mu_range[1] - self.mu_range[0]) + self.mu_range[0]
        
        # Define a new function for the pathtracking discrete-time dynamics. 
        def pathtracking_points(state, t_dif):
            x = state[:,:,1]
            y = state[:,:,2]
            yaw = state[:,:,3]
            v = state[:,:,4]
            steer = state[:,:,5]
            accel = state[:,:,6]
            # print("v: ", v.shape)
            # print("mus: ", mus.shape)
            # exit()

            # Apply friction/drag to the velocity
            v = v - mus*torch.log(v+1)

            # update state variables
            new_x = x + v * torch.cos(yaw) * t_dif
            new_y = y + v * torch.sin(yaw) * t_dif
            new_yaw = yaw + v / self.wheel_base * torch.tan(steer) * t_dif
            new_v = v + accel * t_dif
            # print("new_x: ", new_x.shape)
            # print("new_y: ", new_y.shape)
            # print("new_yaw: ", new_yaw.shape)
            # print("new_v: ", new_v.shape)
            # exit()
            return torch.stack([new_x, new_y, new_yaw, new_v], dim=2)
            
        return lambda x_0: self.integrate_points(pathtracking_points, x_0), mus
    
    def integrate_points(self, model, x_0):
        """
        Integrate every point one step forward using the model. 
        x0: (n_func, n_samp, 4)
        """

        # propagate forward to the next states
        x_next = model(x_0, self.t_dif)
        output = x_next - x_0[:,:,1:5] # change in states

        return output
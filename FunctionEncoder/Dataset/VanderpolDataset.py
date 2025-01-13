from typing import Tuple, Union

import torch

from FunctionEncoder.Dataset.BaseDataset import BaseDataset


class VanderpolDataset(BaseDataset):

    def __init__(self,
                 mu_range = (0.1, 3.0), # mu coef. range
                 t_dif_range = (0.0, 1.0), # time range
                 x_range = (-2.0, 2.0), # x state range
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

        super().__init__(input_size=(3,), # time?
                         output_size=(2,), # x & y positions? 

                         data_type="deterministic",
                         device=device,
                         dtype=dtype,
                         n_functions=n_functions,
                         n_examples=n_examples,
                         n_queries=n_queries,

                         # deprecated arguments
                         total_n_functions=None,
                         total_n_samples_per_function=None,
                         n_functions_per_sample=n_functions_per_sample,
                         n_examples_per_sample=n_examples_per_sample,
                         n_points_per_sample=n_points_per_sample,

                         )
        self.mu_range = torch.tensor(mu_range, device=self.device, dtype=self.dtype)
        self.t_dif_range = torch.tensor(t_dif_range, device=self.device, dtype=self.dtype)
        self.x_range = torch.tensor(x_range, device=self.device, dtype=self.dtype)

    def sample(self) -> Tuple[  torch.tensor, # example inputs (10, 1000, 3)
                                torch.tensor, # example outputs (10, 1000, 2)
                                torch.tensor, # query inputs (10, 10000, 3)
                                torch.tensor, # query outputs (10, 10000, 2)
                                dict]: # dictionary of function parameters
        with torch.no_grad():
            # Choose method 1 to generate random input points.
            # Choose method 2 to generate a full trajectory. 
            method = 2

            if method ==1:
                # Generate random input values for each query/example. 
                # Note that this creates random input states for each 
                # sample, but when generating a trajectory, we only
                # use the first one. 
                query_xs = self.get_input_points(self.n_queries)
                example_xs = self.get_input_points(self.n_examples)
                
                # create the vanderpol_models functions and generate the mus
                vanderpol_models, mus = self.get_vanderpol_models(method)
                query_ys = vanderpol_models(query_xs) # call the integrate function returned by get_vanderpol_models()
                example_ys = vanderpol_models(example_xs)

            elif method == 2:
                # Generate random initial states. We only care about position here. 
                query_x0 = torch.rand((self.n_functions, 2), dtype=self.dtype, device=self.device)
                example_x0 = torch.rand((self.n_functions, 2), dtype=self.dtype, device=self.device)
                # Bound the initial conditions within the chosen range.
                query_x0 = query_x0 * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
                example_x0 = example_x0 * (self.x_range[1] - self.x_range[0]) + self.x_range[0]

                # Create the vanderpol_models functions and generate the mus
                vanderpol_models, mus = self.get_vanderpol_models(method)
                # Propagate the initial state forward using the Van der Pol dynamics.
                query_xs, query_ys = vanderpol_models(query_x0, self.n_queries) 
                example_xs, example_ys = vanderpol_models(example_x0, self.n_examples) 

            return example_xs, example_ys, query_xs, query_ys, {"mus":mus}

    def get_input_points(self, n_samples):
        """
        Return a tensor of inputs values (x, y coordinates) for each
        mu function and for each example/query.
        """
        # generate n_samples_per_function samples for each function
        xs = torch.rand((self.n_functions, n_samples, *self.input_size), dtype=self.dtype, device=self.device)
        # constrain each set of inputs within their respective ranges
        xs0 = xs[:,:,0] * (self.t_dif_range[1] - self.t_dif_range[0]) + self.t_dif_range[0]
        xs1 = xs[:,:,1] * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        xs2 = xs[:,:,2] * (self.x_range[1] - self.x_range[0]) + self.x_range[0]
        # merge the inputs back into one tensor
        return torch.stack([xs0, xs1, xs2], dim=2)

    def get_vanderpol_models(self, method):
        # Generate the random parameters for each function. 
        mus = torch.rand(self.n_functions, 1, device=self.device) * (self.mu_range[1] - self.mu_range[0]) + self.mu_range[0]

        # Define a new function for the Van der Pol instantaneous dynamics. 
        def vanderpol(x):
            if len(x.shape) == 2:
                # Dynamics for when you're simulating a full trajectory. 
                # x is a (n_functions, 2) tensor. 
                x0 = x[:,0].unsqueeze(1)
                x1 = x[:,1].unsqueeze(1)
                dx = x1
                dy = mus * (1 - x0**2) * x1 - x0
                return torch.concat([dx, dy], dim=1)
            else:
                # Dynamics for when you're sampling a bunch of points.
                # x is a (n_functions, n_samples, 2) tensor.
                dx = x[:, :, 1]
                dy = mus * (1 - x[:, :, 0]**2) * x[:, :, 1] - x[:, :, 0]
                return torch.stack([dx, dy], dim=0) # (2, n_functions, n_samples)
            
        if method == 1:
            return lambda x_0: self.integrate_points(vanderpol, x_0), mus
        elif method == 2:
            return lambda x_0, n_samples: self.integrate_traj(vanderpol, x_0, n_samples), mus
    
    # we use rk4
    def integrate_points(self, model, x_0):
        """
        Integrate the model from t_0 to t_f with initial condition x_0.
        Input Size (x_0) = (n_func, n_samples, 3)
        Output Size = (n_func, n_samples, 2)
        """
        t_dif = x_0[:,:,0]
        xy = x_0[:,:,1:]
        #print("t_dif: ", t_dif.shape)

        k1 = model(xy)
        #print("k1: ", k1.shape)
        k2 = model(xy + 0.5 * (t_dif * k1).permute(1,2,0))
        k3 = model(xy + 0.5 * (t_dif * k2).permute(1,2,0))
        k4 = model(xy + (t_dif * k3).permute(1,2,0))

        # returns the distance between states
        check = t_dif / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return check.permute(1,2,0)
    
    # we use rk4
    def integrate_traj(self, model, x_0, n_samples):
        """
        Integrate the model from t_0 to t_f with initial condition x_0.
        """
        # Generate random time differences for integration and bound them.
        t_dif = torch.rand((self.n_functions, n_samples, 1), dtype=self.dtype, device=self.device)
        t_dif = t_dif * (self.t_dif_range[1] - self.t_dif_range[0]) + self.t_dif_range[0]

        # initialize space for input/output states
        x_in = torch.zeros(self.n_functions, n_samples, 2, device=self.device)
        x_out = torch.zeros(self.n_functions, n_samples, 2, device=self.device)
        x_in[:, 0, :] = x_0

        # gather data
        for i in range(n_samples):
            x_now = x_in[:, i, :]

            k1 = model(x_now)
            k2 = model(x_now + 0.5 * (t_dif[:,i] * k1))
            k3 = model(x_now + 0.5 * (t_dif[:,i] * k2))
            k4 = model(x_now + (t_dif[:,i] * k3))

            x_dif = t_dif[:,i] / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            # Record input/output data. The output data is the
            # CHANGE in state, not the next state itself. 
            x_out[:, i, :] = x_dif
            if i != n_samples-1:
                x_in[:, i+1, :] = x_now + x_dif

        # stack the time diff data with the input data.
        x_in = torch.cat([t_dif, x_in], dim=2)

        return x_in, x_out
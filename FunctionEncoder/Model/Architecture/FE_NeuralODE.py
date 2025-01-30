import math

import torch

from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture
from FunctionEncoder.Model.Architecture.MLP import get_activation


class ParallelLinear(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, n_parallel, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        n = n_inputs
        m = n_outputs
        p = n_parallel
        self.W = torch.nn.Parameter(torch.zeros((p, m, n), **factory_kwargs))
        self.b = torch.nn.Parameter(torch.zeros((p, m), **factory_kwargs))
        self.n = n
        self.m = m
        self.p = p
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # this function was designed for a single layer, so we need to do it k times to not break their code.
        # this is slow but we only pay this cost once.
        for i in range(self.p):
            torch.nn.init.kaiming_uniform_(self.W[i, :, :], a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.W[i, :, :])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.b[i, :], -bound, bound)

    def forward(self, x):
        assert x.shape[-2] == self.n, f"Input size of model '{self.n}' does not match input size of data '{x.shape[-2]}'"
        assert x.shape[-1] == self.p, f"Batch size of model '{self.p}' does not match batch size of data '{x.shape[-1]}'"
        y = torch.einsum("pmn,...np->...mp", self.W, x) + self.b.T.reshape(1, 1, self.m, self.p)
        return y

    def num_params(self):
        return self.W.numel() + self.b.numel()

    def __repr__(self):
        return f"ParallelLinear({self.n}, {self.m}, {self.p})"

    def __str__(self):
        return self.__repr__()
    def __call__(self, x):
        return self.forward(x)



class FE_NeuralODE(BaseArchitecture):
    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, hidden_size=77, n_layers=4, learn_basis_functions=True, *args, **kwargs):
        input_size = input_size[0]
        output_size = output_size[0]
        # +1 accounts for bias
        n_params =  (input_size+1) * hidden_size + \
                    (hidden_size+1) * hidden_size * (n_layers - 2) + \
                    (hidden_size+1) * output_size
        if learn_basis_functions:
            n_params *= n_basis
        return n_params


    def __init__(self,
                 input_size:tuple[int],
                 output_size:tuple[int],
                 n_basis:int=100,
                 hidden_size:int=77,
                 n_layers:int=4,
                 activation:str="relu",
                 learn_basis_functions=True):
        super(FE_NeuralODE, self).__init__()
        assert type(input_size) == tuple, "input_size must be a tuple"
        assert type(output_size) == tuple, "output_size must be a tuple"
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.learn_basis_functions = learn_basis_functions

        if not self.learn_basis_functions:
            n_basis = 1
            self.n_basis = 1

        # get inputs
        input_size = input_size[0] - 1 # only 1D input supported for now
        output_size = output_size[0]

        # build net
        layers = []
        layers.append(ParallelLinear(input_size, hidden_size, n_basis))
        layers.append(get_activation(activation))
        for _ in range(n_layers - 2):
            layers.append(ParallelLinear(hidden_size, hidden_size, n_basis))
            layers.append(get_activation(activation))
        layers.append(ParallelLinear(hidden_size, output_size, n_basis))
        self.model = torch.nn.Sequential(*layers)

        # verify number of parameters
        n_params = sum([p.numel() for p in self.parameters()])
        estimated_n_params = self.predict_number_params(self.input_size, self.output_size, n_basis, hidden_size, n_layers, learn_basis_functions=learn_basis_functions)
        #assert n_params == estimated_n_params, f"Model has {n_params} parameters, but expected {estimated_n_params} parameters."


    def forward(self, x):
        assert x.shape[-1] == self.input_size[0], f"Expected input size {self.input_size[0]}, got {x.shape[-1]}"
        reshape = None
        if len(x.shape) == 1:
            reshape = 1
            x = x.reshape(1, 1, -1)
        if len(x.shape) == 2:
            reshape = 2
            x = x.unsqueeze(0)

        # need to append the batch dimension as the last dim
        x = x.unsqueeze(-1).repeat(1, 1, 1, self.n_basis)

        # this is the main part of this function. The rest is just error handling
        #print("x: ", x.shape)
        outs = self.predict(x)
        #print("outs: ", outs.shape)
        if self.learn_basis_functions:
            Gs = outs.view(*x.shape[:2], *self.output_size, self.n_basis)
        else:
            Gs = outs.view(*x.shape[:2], *self.output_size)

        # send back to the given shape
        if reshape == 1:
            Gs = Gs.squeeze(0).squeeze(0)
        if reshape == 2:
            Gs = Gs.squeeze(0)
        return Gs


    def rk4(self, states):
        t_dif = states[:,:,0,:]
        xy = states[:,:,1:,:]
        #print("xy: ", xy.shape)
        #print("t_dif: ", t_dif.shape)

        k1 = self.model(xy)
        #print("k1: ", k1.shape)
        #print((t_dif * k1.permute(2,0,1,3)).shape)
        k2 = self.model(xy + 0.5 * (t_dif * k1.permute(2,0,1,3)).permute(1,2,0,3))
        k3 = self.model(xy + 0.5 * (t_dif * k2.permute(2,0,1,3)).permute(1,2,0,3))
        k4 = self.model(xy + (t_dif * k3.permute(2,0,1,3)).permute(1,2,0,3))
        return (t_dif / 6 * (k1.permute(2,0,1,3) + 2 * k2.permute(2,0,1,3) + 2 * k3.permute(2,0,1,3) + k4.permute(2,0,1,3))).permute(1,2,0,3)
    
    def rk4_actions(self, state_actions):
        """ TODO: generalize the code for any number of states/actions"""
        t_dif = state_actions[:,:,0,:]
        states = state_actions[:,:,1:8,:]
        inputs = state_actions[:,:,8:,:]
        # print("states: ", states.shape)
        # print("t_dif: ", t_dif.shape)
        # print("inputs: ", inputs.shape)
        # print("state_actions: ", state_actions.shape)

        k1 = self.predict_xdot(states, inputs)
        #print("k1: ", k1.shape)
        #print((t_dif * k1.permute(2,0,1,3)).shape)
        k2 = self.predict_xdot(states + 0.5 * (t_dif * k1.permute(2,0,1,3)).permute(1,2,0,3), inputs)
        k3 = self.predict_xdot(states + 0.5 * (t_dif * k2.permute(2,0,1,3)).permute(1,2,0,3), inputs)
        k4 = self.predict_xdot(states + (t_dif * k3.permute(2,0,1,3)).permute(1,2,0,3), inputs)
        return (t_dif / 6 * (k1.permute(2,0,1,3) + 2 * k2.permute(2,0,1,3) + 2 * k3.permute(2,0,1,3) + k4.permute(2,0,1,3))).permute(1,2,0,3)

    def predict_xdot(self, states, actions):
        inputs = torch.cat([states, actions], dim=2) 
        return self.model(inputs)

    # predicts the next states given states and actions
    def predict(self, states):
        # assert states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(states.shape[-1], self.state_size)
        # assert actions.shape[-1] == self.action_size, "Input size is {}, expected {}".format(actions.shape[-1], self.action_size)
        # assert example_states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(example_states.shape[-1], self.state_size)
        # assert example_actions.shape[-1] == self.action_size, "Input size is {}, expected {}".format(example_actions.shape[-1], self.action_size)
        # assert example_next_states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(example_next_states.shape[-1], self.state_size)
        return self.rk4_actions(states)
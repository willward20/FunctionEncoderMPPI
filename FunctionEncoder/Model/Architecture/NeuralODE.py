import torch

from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture


# Returns the desired activation function by name
def get_activation( activation):
    if activation == "relu":
        return torch.nn.ReLU()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation: {activation}")

class NeuralODE(BaseArchitecture):


    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, hidden_size=256, n_layers=4, learn_basis_functions=True, *args, **kwargs):
        input_size = input_size[0]
        output_size = output_size[0] * n_basis if learn_basis_functions else output_size[0]
        n_params =  input_size * hidden_size + hidden_size + \
                    (n_layers - 2) * hidden_size * hidden_size + (n_layers - 2) * hidden_size + \
                    hidden_size * output_size + output_size
        return n_params

    def __init__(self,
                 input_size:tuple[int],
                 output_size:tuple[int],
                 n_basis:int=100,
                 hidden_size:int=256,
                 n_layers:int=4,
                 activation:str="relu",
                 learn_basis_functions=True):
        super(NeuralODE, self).__init__()
        assert type(input_size) == tuple, "input_size must be a tuple"
        assert type(output_size) == tuple, "output_size must be a tuple"
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.learn_basis_functions = learn_basis_functions


        # get inputs
        input_size = input_size[0] - 1 # only 1D input supported for now
        output_size = output_size[0] * n_basis if learn_basis_functions else output_size[0]

        # build net
        layers = []
        layers.append(torch.nn.Linear(input_size, hidden_size))
        layers.append(get_activation(activation))
        for _ in range(n_layers - 2):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(get_activation(activation))
        layers.append(torch.nn.Linear(hidden_size, output_size))
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

        # this is the main part of this function. The rest is just error handling
        outs = self.predict(x)
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
        t_dif = states[:,:,0]
        xy = states[:,:,1:]
        #print("xy: ", xy.shape)
        #print("t_dif: ", t_dif.shape)

        k1 = self.model(xy)
        #print("k1: ", k1.permute(2,0,1).shape)
        #print((t_dif * k1.permute(2,0,1)).shape)
        k2 = self.model(xy + 0.5 * (t_dif * k1.permute(2,0,1)).permute(1,2,0))
        k3 = self.model(xy + 0.5 * (t_dif * k2.permute(2,0,1)).permute(1,2,0))
        k4 = self.model(xy + (t_dif * k3.permute(2,0,1)).permute(1,2,0))
        return (t_dif / 6 * (k1.permute(2,0,1) + 2 * k2.permute(2,0,1) + 2 * k3.permute(2,0,1) + k4.permute(2,0,1))).permute(1,2,0)
    
    # predicts the next states given states and actions
    def predict(self, states):
        # assert states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(states.shape[-1], self.state_size)
        # assert actions.shape[-1] == self.action_size, "Input size is {}, expected {}".format(actions.shape[-1], self.action_size)
        # assert example_states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(example_states.shape[-1], self.state_size)
        # assert example_actions.shape[-1] == self.action_size, "Input size is {}, expected {}".format(example_actions.shape[-1], self.action_size)
        # assert example_next_states.shape[-1] == self.state_size, "Input size is {}, expected {}".format(example_next_states.shape[-1], self.state_size)
        return self.rk4(states)

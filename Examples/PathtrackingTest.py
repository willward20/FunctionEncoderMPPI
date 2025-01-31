import argparse
import torch

import matplotlib.pyplot as plt

from FunctionEncoder import PathtrackingDataset, FunctionEncoder

""" This script loads a pre-trained model, tests it on 
    new queries, and plots the residuals. """

# Parse the input arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=11)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=1_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true") # default = False 
parser.add_argument("--parallel", action="store_true") # default = False
args = parser.parse_args()

# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cuda" if torch.cuda.is_available() else "cpu"
train_method = args.train_method
seed = args.seed
residuals = args.residuals
arch = "FE_NeuralODE" 

# create a dataset
dataset = PathtrackingDataset()

# create the model
model = FunctionEncoder(input_size=dataset.input_size,
                        output_size=dataset.output_size,
                        data_type=dataset.data_type,
                        n_basis=n_basis,
                        model_type=arch,
                        method=train_method,
                        use_residuals_method=residuals).to(device)
print('Number of parameters:', sum(p.numel() for p in model.parameters()))

# load a pre-trained parameters
path = "/home/wward/projects/FunctionEncoderMPPI/logs/pathtracking_example/least_squares/shared_model/2025-01-19_09-32-50"
model.load_state_dict(torch.load(f"{path}/model.pth"))

# test model
with torch.no_grad():
    n_plots = 9
    n_examples = 100
    example_xs, example_ys, query_xs, query_ys, info = dataset.sample()
    example_xs, example_ys = example_xs[:, :n_examples, :], example_ys[:, :n_examples, :]
    y_hats_ls = model.predict_from_examples(example_xs, example_ys, query_xs, method=train_method)
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(n_plots):
        ax = axs[i // 3, i % 3]
        ax.plot(
            torch.abs(query_ys[i,:100,0].cpu() - y_hats_ls[i,:100,0].cpu()), label='x error'
        )
        ax.plot(
            torch.abs(query_ys[i,:100,1].cpu() - y_hats_ls[i,:100,1].cpu()), label='y error'
        )
        ax.plot(
            torch.abs(query_ys[i,:100,2].cpu() - y_hats_ls[i,:100,2].cpu()), label='yaw error'
        )
        ax.plot(
            torch.abs(query_ys[i,:100,3].cpu() - y_hats_ls[i,:100,3].cpu()), label='v error'
        )

        if i == n_plots - 1:
            ax.legend()
        title = f"${info['mus'][i].item():.2f}$"
        ax.set_title(title)

    plt.show()
    # plt.tight_layout()
    # plt.savefig(f"{path}/state_errors.png")
    # plt.clf()


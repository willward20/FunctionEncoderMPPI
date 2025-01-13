from datetime import datetime

import matplotlib.pyplot as plt
import torch

from FunctionEncoder import VanderpolDataset, FunctionEncoder, ListCallback, TensorboardCallback, \
    DistanceCallback

import argparse
from FunctionEncoder.Model.Function import Function


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=11)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=1_000) #1_000
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--residuals", action="store_true") # NOTE: this defaults to false, but you might try changing it to true. Tyler says it can help. 
parser.add_argument("--parallel", action="store_true")
args = parser.parse_args()


# hyper params
epochs = args.epochs
n_basis = args.n_basis
device = "cuda" if torch.cuda.is_available() else "cpu"
train_method = args.train_method
seed = args.seed
load_path = args.load_path
residuals = args.residuals
if load_path is None:
    logdir = f"logs/vanderpol_example/{train_method}/{'shared_model' if not args.parallel else 'parallel_models'}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path
arch = "FE_NeuralODE" #"MLP" if not args.parallel else "ParallelMLP"
print(arch)

# seed torch
#torch.manual_seed(seed)

# create a dataset
mu_range = (0.1, 3.0) # mu range
t_dif_range = (0.0, 0.1) # time range
x_range = (-2.0, 2.0) # x and y state range
dataset = VanderpolDataset(mu_range=mu_range, t_dif_range=t_dif_range, x_range=x_range)

if load_path is None:
    # create the model
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            model_type=arch,
                            method=train_method,
                            use_residuals_method=residuals).to(device)
    # model = Function(input_size=dataset.input_size,
    #                     output_size=dataset.output_size,
    #                     data_type=dataset.data_type,
    #                     model_type=arch).to(device)
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))

    # create callbacks
    cb1 = TensorboardCallback(logdir) # this one logs training data
    cb2 = DistanceCallback(dataset, tensorboard=cb1.tensorboard) # this one tests and logs the results
    callback = ListCallback([cb1, cb2])

    # train the model
    model.train_model(dataset, epochs=epochs, callback=callback)

    # save the model
    torch.save(model.state_dict(), f"{logdir}/model.pth")
else:
    # load the model
    model = FunctionEncoder(input_size=dataset.input_size,
                            output_size=dataset.output_size,
                            data_type=dataset.data_type,
                            n_basis=n_basis,
                            model_type=arch,
                            method=train_method,
                            use_residuals_method=residuals).to(device)
    model.load_state_dict(torch.load(f"{logdir}/model.pth"))

# plot
with torch.no_grad():
    n_plots = 9
    n_examples = 100
    example_xs, example_ys, query_xs, query_ys, info = dataset.sample()
    example_xs, example_ys = example_xs[:, :n_examples, :], example_ys[:, :n_examples, :]
    if train_method == "inner_product":
        y_hats_ip = model.predict_from_examples(example_xs, example_ys, query_xs, method="inner_product")
    y_hats_ls = model.predict_from_examples(example_xs, example_ys, query_xs, method="least_squares")
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(n_plots):
        ax = axs[i // 3, i % 3]
        # Plotting for Method 1 (points/vectors)
        # ax.quiver(
        #     query_xs[i,900:999,1].cpu(), query_xs[i,900:999,2].cpu(), # x and y coordinates of vector
        #     query_ys[i,900:999,0].cpu(), query_ys[i,900:999,1].cpu(), # x and y components of the vector
        #     angles='xy', scale_units='xy', scale=1, label="True", width=0.005
        # )
        # ax.quiver(
        #     query_xs[i,900:999,1].cpu(), query_xs[i,900:999,2].cpu(), # x and y coordinates of vector
        #     y_hats_ls[i,900:999,0].cpu(), y_hats_ls[i,900:999,1].cpu(), # x and y components of the vector
        #     angles='xy', scale_units='xy', scale=1, label="LS", width=0.005, color='b'
        # )
        # Plotting for Method 2 (trajectory)
        ax.plot((query_xs[i,:,1] + query_ys[i,:,0]).cpu(), (query_xs[i,:,2] + query_ys[i,:,1]).cpu(), label="True")
        ax.plot((query_xs[i,:,1] + y_hats_ls[i,:,0]).cpu(), (query_xs[i,:,2] + y_hats_ls[i,:,1]).cpu(), label="LS")

        if i == n_plots - 1:
            ax.legend()
        title = f"${info['mus'][i].item():.2f}$"
        ax.set_title(title)
        #y_min, y_max = query_ys[i].min().item(), query_ys[i].max().item()
        #ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f"{logdir}/plot.png")
    plt.clf()

    # # plot the basis functions
    # fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    # query_xs = torch.linspace(input_range[0], input_range[1], 1_000).reshape(1000, 1).to(device)
    # basis = model.forward_basis_functions(query_xs)
    # for i in range(n_basis):
    #     ax.plot(query_xs.flatten().cpu(), basis[:, 0, i].cpu(), color="black")
    # if residuals:
    #     avg_function = model.average_function.forward(query_xs)
    #     ax.plot(query_xs.flatten().cpu(), avg_function.flatten().cpu(), color="blue")

    # plt.tight_layout()
    # plt.savefig(f"{logdir}/basis.png")

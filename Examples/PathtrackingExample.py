import argparse
import torch

from datetime import datetime
import matplotlib.pyplot as plt

from FunctionEncoder import PathtrackingDataset, FunctionEncoder, ListCallback, TensorboardCallback, \
    DistanceCallback

# Parse the input arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=11)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=1_000)
parser.add_argument("--load_path", type=str, default=None)
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
load_path = args.load_path
residuals = args.residuals
if load_path is None:
    logdir = f"logs/pathtracking_example/{train_method}/{'shared_model' if not args.parallel else 'parallel_models'}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path
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

# create callbacks
cb1 = TensorboardCallback(logdir) # this one logs training data
cb2 = DistanceCallback(dataset, tensorboard=cb1.tensorboard) # this one tests and logs the results
callback = ListCallback([cb1, cb2])

# train the model
model.train_model(dataset, epochs=epochs, callback=callback)

# save the model
torch.save(model.state_dict(), f"{logdir}/model.pth")

# plot
with torch.no_grad():
    n_plots = 9
    n_examples = 100
    example_xs, example_ys, query_xs, query_ys, info = dataset.sample()
    example_xs, example_ys = example_xs[:, :n_examples, :], example_ys[:, :n_examples, :]
    y_hats_ls = model.predict_from_examples(example_xs, example_ys, query_xs, method="least_squares")
    
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(n_plots):
        ax = axs[i // 3, i % 3]
        # Plotting for Method 1 (points/vectors)
        ax.quiver(
            query_xs[i,:100,1].cpu(), query_xs[i,:100,2].cpu(), # x and y coordinates of vector
            query_ys[i,:100,0].cpu(), query_ys[i,:100,1].cpu(), # x and y components of the vector
            angles='xy', scale_units='xy', scale=1, label="True", width=0.005
        )
        ax.quiver(
            query_xs[i,:100,1].cpu(), query_xs[i,:100,2].cpu(), # x and y coordinates of vector
            y_hats_ls[i,:100,0].cpu(), y_hats_ls[i,:100,1].cpu(), # x and y components of the vector
            angles='xy', scale_units='xy', scale=1, label="LS", width=0.005, color='b'
        )
        # Plotting for Method 2 (trajectory)
        # ax.plot((query_xs[i,:,1] + query_ys[i,:,0]).cpu(), (query_xs[i,:,2] + query_ys[i,:,1]).cpu(), label="True")
        # ax.plot((query_xs[i,:,1] + y_hats_ls[i,:,0]).cpu(), (query_xs[i,:,2] + y_hats_ls[i,:,1]).cpu(), label="LS")

        if i == n_plots - 1:
            ax.legend()
        title = f"${info['mus'][i].item():.2f}$"
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(f"{logdir}/plot.png")
    plt.clf()


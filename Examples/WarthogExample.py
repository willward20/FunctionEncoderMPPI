import argparse
import torch

from datetime import datetime
import matplotlib.pyplot as plt
import os
import sys

# Add path to the FunctionEncoder package.
home = os.path.expanduser('~')
sys.path.append(f'{home}/FunctionEncoderMPPI')
from FunctionEncoder import WarthogDataset, FunctionEncoder, ListCallback, TensorboardCallback, \
    DistanceCallback

# Parse the input arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--n_basis", type=int, default=11)
parser.add_argument("--train_method", type=str, default="least_squares")
parser.add_argument("--epochs", type=int, default=1_00000)
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
    logdir = f"logs/warthog_example/{train_method}/{'shared_model' if not args.parallel else 'parallel_models'}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
else:
    logdir = load_path
arch = "MLP" 

# create a dataset
csv_path = "/home/arl/catkin_ws/src/mppi_rollouts/data/2025-02-28-10-58-11/warty-odom_processed-10hz-20.csv"
dataset = WarthogDataset(csv_file=csv_path)

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
exit()

# plot
with torch.no_grad():
    n_examples = 100
    example_xs, example_ys, query_xs, query_ys, _ = dataset.sample_test()
    example_xs, example_ys = example_xs[:, :n_examples, :], example_ys[:, :n_examples, :]
    y_hats_ls = model.predict_from_examples(example_xs, example_ys, query_xs, method=train_method)
    
    # Plot residuals. 
    plt.plot(
        torch.abs(query_ys[:,:100,0].cpu() - y_hats_ls[:,:100,0].cpu()).flatten(), label='x error'
    )
    plt.plot(
        torch.abs(query_ys[:,:100,1].cpu() - y_hats_ls[:,:100,1].cpu()).flatten(), label='y error'
    )
    plt.plot(
        torch.abs(query_ys[:,:100,2].cpu() - y_hats_ls[:,:100,2].cpu()).flatten(), label='yaw error'
    )
    plt.legend()
    # plt.show()
    plt.tight_layout()
    plt.savefig(f"{logdir}/plot.png")
    plt.clf()


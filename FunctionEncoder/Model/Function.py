from typing import Union, Tuple
import torch
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder.Callbacks.BaseCallback import BaseCallback
from FunctionEncoder.Dataset.BaseDataset import BaseDataset
from FunctionEncoder.Model.Architecture.BaseArchitecture import BaseArchitecture
from FunctionEncoder.Model.Architecture.CNN import CNN
from FunctionEncoder.Model.Architecture.Euclidean import Euclidean
from FunctionEncoder.Model.Architecture.MLP import MLP
from FunctionEncoder.Model.Architecture.ParallelMLP import ParallelMLP
from FunctionEncoder.Model.Architecture.NeuralODE import NeuralODE


class Function(torch.nn.Module):
    
    def __init__(self,
                 input_size:tuple[int], 
                 output_size:tuple[int], 
                 data_type:str, 
                 model_type:Union[str, type]="MLP",
                 model_kwargs:dict=dict(),
                 regularization_parameter:float=1.0, # if you normalize your data, this is usually good
                 gradient_accumulation:int=1, # default: no gradient accumulation
                 optimizer=torch.optim.Adam,
                 optimizer_kwargs:dict={"lr":1e-3},
                 ):
        """ Initializes a function encoder.

        Args:
        input_size: tuple[int]: The size of the input space, e.g. (1,) for 1D input
        output_size: tuple[int]: The size of the output space, e.g. (1,) for 1D output
        data_type: str: "deterministic" or "stochastic". Determines which defintion of inner product is used.
        n_basis: int: Number of basis functions to use.
        model_type: str: The type of model to use. See the types and kwargs in FunctionEncoder/Model/Architecture. Typically a MLP.
        model_kwargs: Union[dict, type(None)]: The kwargs to pass to the model. See the types and kwargs in FunctionEncoder/Model/Architecture.
        method: str: "inner_product" or "least_squares". Determines how to compute the coefficients of the basis functions.
        use_residuals_method: bool: Whether to use the residuals method. If True, uses an average function to predict the average of the data, and then learns the error with a function encoder.
        regularization_parameter: float: The regularization parameter for the least squares method, that encourages the basis functions to be unit length. 1 is usually good, but if your ys are very large, this may need to be increased.
        gradient_accumulation: int: The number of batches to accumulate gradients over. Typically its best to have n_functions>=10 or so, and have gradient_accumulation=1. However, sometimes due to memory reasons, or because the functions do not have the same amount of data, its necesary for n_functions=1 and gradient_accumulation>=10.
        """
        if model_type == "MLP":
            assert len(input_size) == 1, "MLP only supports 1D input"
        if model_type == "ParallelMLP":
            assert len(input_size) == 1, "ParallelMLP only supports 1D input"
        if model_type == "CNN":
            assert len(input_size) == 3, "CNN only supports 3D input"
        if isinstance(model_type, type):
            assert issubclass(model_type, BaseArchitecture), "model_type should be a subclass of BaseArchitecture. This just gives a way of predicting the number of parameters before init."
        assert len(input_size) in [1, 3], "Input must either be 1-Dimensional (euclidean vector) or 3-Dimensional (image)"
        assert input_size[0] >= 1, "Input size must be at least 1"
        assert len(output_size) == 1, "Only 1D output supported for now"
        assert output_size[0] >= 1, "Output size must be at least 1"
        assert data_type in ["deterministic", "stochastic", "categorical"], f"Unknown data type: {data_type}"
        super(Function, self).__init__()
        
        # hyperparameters
        self.input_size = input_size # just for getting NODE to work
        self.output_size = output_size
        self.data_type = data_type
        
        self.method = "inner_product"

        # models and optimizers
        self.average_function = self._build(model_type, model_kwargs, average_function=True) 
        self.opt = optimizer(self.average_function.parameters(), **optimizer_kwargs) # usually ADAM with lr 1e-3

        # regulation only used for LS method
        self.regularization_parameter = regularization_parameter
        # accumulates gradients over multiple batches, typically used when n_functions=1 for memory reasons. 
        self.gradient_accumulation = gradient_accumulation

        # for printing
        self.model_type = model_type
        self.model_kwargs = model_kwargs

        # verify number of parameters
        n_params = sum([p.numel() for p in self.parameters()])
        estimated_n_params = Function.predict_number_params(input_size=input_size, output_size=output_size, n_basis=1, model_type=model_type, model_kwargs=model_kwargs, use_residuals_method=True)
        #assert n_params == estimated_n_params, f"Model has {n_params} parameters, but expected {estimated_n_params} parameters."



    def _build(self, 
               model_type:Union[str, type],
               model_kwargs:dict, 
               average_function:bool=False) -> torch.nn.Module:
        """Builds a function encoder as a single model. Can also build the average function. 
        
        Args:
        model_type: Union[str, type]: The type of model to use. See the types and kwargs in FunctionEncoder/Model/Architecture. Typically "MLP", can also be a custom class.
        model_kwargs: dict: The kwargs to pass to the model. See the kwargs in FunctionEncoder/Model/Architecture/.
        average_function: bool: Whether to build the average function. If True, builds a single function model.

        Returns:
        torch.nn.Module: The basis functions or average function model.
        """

        # if provided as a string, parse the string into a class
        if type(model_type) == str:
            if model_type == "MLP":
                return MLP(input_size=self.input_size,
                           output_size=self.output_size,
                           n_basis=1,
                           learn_basis_functions=True,
                           **model_kwargs)
            if model_type == "NeuralODE":
                return NeuralODE(input_size=self.input_size,
                           output_size=self.output_size,
                           n_basis=1,
                           learn_basis_functions=True,
                           **model_kwargs)
            if model_type == "ParallelMLP":
                return ParallelMLP(input_size=self.input_size,
                                   output_size=self.output_size,
                                   n_basis=1,
                                   learn_basis_functions=True,
                                   **model_kwargs)
            elif model_type == "Euclidean":
                return Euclidean(input_size=self.input_size,
                                 output_size=self.output_size,
                                 n_basis=1,
                                 **model_kwargs)
            elif model_type == "CNN":
                return CNN(input_size=self.input_size,
                           output_size=self.output_size,
                           n_basis=1,
                           learn_basis_functions=True,
                           **model_kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}. Should be one of 'MLP', 'ParallelMLP', 'Euclidean', or 'CNN'")
        else:  # otherwise, assume it is a class and directly instantiate it
            return model_type(input_size=self.input_size,
                              output_size=self.output_size,
                              n_basis=self.n_basis,
                              learn_basis_functions=True,
                              **model_kwargs)

    
    def _deterministic_inner_product(self, 
                                     fs:torch.tensor, 
                                     gs:torch.tensor,) -> torch.tensor:
        """Approximates the L2 inner product between fs and gs using a Monte Carlo approximation.
        Latex: \langle f, g \rangle = \frac{1}{V}\int_X f(x)g(x) dx \approx \frac{1}{n} \sum_{i=1}^n f(x_i)g(x_i)
        Note we are scaling the L2 inner product by 1/volume, which removes volume from the monte carlo approximation.
        Since scaling an inner product is still a valid inner product, this is still an inner product.
        
        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """
        
        assert len(fs.shape) in [3,4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3,4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2], f"Expected fs and gs to have the same output size, got {fs.shape[2]} and {gs.shape[2]}"

        # reshaping
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True

        # compute inner products via MC integration
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)

        # undo reshaping
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _stochastic_inner_product(self, 
                                  fs:torch.tensor, 
                                  gs:torch.tensor,) -> torch.tensor:
        """ Approximates the logit version of the inner product between continuous distributions. 
        Latex: \langle f, g \rangle = \int_X (f(x) - \Bar{f}(x) )(g(x) - \Bar{g}(x)) dx \approx \frac{1}{n} \sum_{i=1}^n (f(x_i) - \Bar{f}(x_i))(g(x_i) - \Bar{g}(x_i))
        
        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """
        
        assert len(fs.shape) in [3,4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3,4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2] == 1, f"Expected fs and gs to have the same output size, which is 1 for the stochastic case since it learns the pdf(x), got {fs.shape[2]} and {gs.shape[2]}"

        # reshaping
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True
        assert len(fs.shape) == 4 and len(gs.shape) == 4, "Expected fs and gs to have shape (f,d,m,k)"

        # compute means and subtract them
        mean_f = torch.mean(fs, dim=1, keepdim=True)
        mean_g = torch.mean(gs, dim=1, keepdim=True)
        fs = fs - mean_f
        gs = gs - mean_g

        # compute inner products
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)
        # Technically we should multiply by volume, but we are assuming that the volume is 1 since it is often not known

        # undo reshaping
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _categorical_inner_product(self,
                                   fs:torch.tensor,
                                   gs:torch.tensor,) -> torch.tensor:
        """ Approximates the inner product between discrete conditional probability distributions.

        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, n_categories, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, n_categories, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """
        
        assert len(fs.shape) in [3, 4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3, 4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2], f"Expected fs and gs to have the same output size, which is the number of categories in this case, got {fs.shape[2]} and {gs.shape[2]}"

        # reshaping
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True
        assert len(fs.shape) == 4 and len(gs.shape) == 4, "Expected fs and gs to have shape (f,d,m,k)"

        # compute means and subtract them
        mean_f = torch.mean(fs, dim=2, keepdim=True)
        mean_g = torch.mean(gs, dim=2, keepdim=True)
        fs = fs - mean_f
        gs = gs - mean_g

        # compute inner products
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)

        # undo reshaping
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _inner_product(self, 
                       fs:torch.tensor, 
                       gs:torch.tensor) -> torch.tensor:
        """ Computes the inner product between fs and gs. This passes the data to either the deterministic or stochastic inner product methods.

        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """
        
        assert len(fs.shape) in [3,4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3,4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2], f"Expected fs and gs to have the same output size, got {fs.shape[2]} and {gs.shape[2]}"

        if self.data_type == "deterministic":
            return self._deterministic_inner_product(fs, gs)
        elif self.data_type == "stochastic":
            return self._stochastic_inner_product(fs, gs)
        elif self.data_type == "categorical":
            return self._categorical_inner_product(fs, gs)
        else:
            raise ValueError(f"Unknown data type: '{self.data_type}'. Should be 'deterministic', 'stochastic', or 'categorical'")

    def _norm(self, fs:torch.tensor, squared=False) -> torch.tensor:
        """ Computes the norm of fs according to the chosen inner product.

        Args:
        fs: torch.tensor: The function outputs. Shape can vary, but typically (n_functions, n_datapoints, input_size)

        Returns:
        torch.tensor: The Hilbert norm of fs.
        """
        norm_squared = self._inner_product(fs, fs)
        if not squared:
            return norm_squared.sqrt()
        else:
            return norm_squared

    def _distance(self, fs:torch.tensor, gs:torch.tensor, squared=False) -> torch.tensor:
        """ Computes the distance between fs and gs according to the chosen inner product.

        Args:
        fs: torch.tensor: The first set of function outputs. Shape can vary, but typically (n_functions, n_datapoints, input_size)
        gs: torch.tensor: The second set of function outputs. Shape can vary, but typically (n_functions, n_datapoints, input_size)
        returns:
        torch.tensor: The distance between fs and gs.
        """
        return self._norm(fs - gs, squared=squared)


    def predict(self, 
                query_xs:torch.tensor,
                representations:torch.tensor, 
                precomputed_average_ys:Union[torch.tensor, None]=None) -> torch.tensor:
        """ Predicts the output of the function encoder given the input data and the coefficients of the basis functions. Uses the average function if it exists.

        Args:
        xs: torch.tensor: The input data. Shape (n_functions, n_datapoints, input_size)
        representations: torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis)
        precomputed_average_ys: Union[torch.tensor, None]: The average function output. If None, computes it. Shape (n_functions, n_datapoints, output_size)
        
        Returns:
        torch.tensor: The predicted output. Shape (n_functions, n_datapoints, output_size)
        """

        assert len(query_xs.shape) == 2 + len(self.input_size), f"Expected xs to have shape (f,d,*n), got {query_xs.shape}"

        # optionally add the average function
        # it is allowed to be precomputed, which is helpful for training
        # otherwise, compute it

        average_ys = self.average_function.forward(query_xs)[:,:,:,0]
        return average_ys

    def predict_from_examples(self, 
                              example_xs:torch.tensor, 
                              example_ys:torch.tensor, 
                              query_xs:torch.tensor,
                              method:str="least_squares",
                              **kwargs):
        """ Predicts the output of the function encoder given the input data and the example data. Uses the average function if it exists.
        
        Args:
        example_xs: torch.tensor: The example input data used to compute a representation. Shape (n_example_datapoints, input_size)
        example_ys: torch.tensor: The example output data used to compute a representation. Shape (n_example_datapoints, output_size)
        xs: torch.tensor: The input data. Shape (n_functions, n_datapoints, input_size)
        method: str: "inner_product" or "least_squares". Determines how to compute the coefficients of the basis functions.
        kwargs: dict: Additional kwargs to pass to the least squares method.

        Returns:
        torch.tensor: The predicted output. Shape (n_functions, n_datapoints, output_size)
        """

        assert len(example_xs.shape) == 2 + len(self.input_size), f"Expected example_xs to have shape (f,d,*n), got {example_xs.shape}"
        assert len(example_ys.shape) == 2 + len(self.output_size), f"Expected example_ys to have shape (f,d,*m), got {example_ys.shape}"
        assert len(query_xs.shape) == 2 + len(self.input_size), f"Expected xs to have shape (f,d,*n), got {query_xs.shape}"
        assert example_xs.shape[-len(self.input_size):] == self.input_size, f"Expected example_xs to have shape (..., {self.input_size}), got {example_xs.shape[-1]}"
        assert example_ys.shape[-len(self.output_size):] == self.output_size, f"Expected example_ys to have shape (..., {self.output_size}), got {example_ys.shape[-1]}"
        assert query_xs.shape[-len(self.input_size):] == self.input_size, f"Expected xs to have shape (..., {self.input_size}), got {query_xs.shape[-1]}"
        assert example_xs.shape[0] == example_ys.shape[0], f"Expected example_xs and example_ys to have the same number of functions, got {example_xs.shape[0]} and {example_ys.shape[0]}"
        assert example_xs.shape[1] == example_xs.shape[1], f"Expected example_xs and example_ys to have the same number of datapoints, got {example_xs.shape[1]} and {example_ys.shape[1]}"
        assert example_xs.shape[0] == query_xs.shape[0], f"Expected example_xs and xs to have the same number of functions, got {example_xs.shape[0]} and {query_xs.shape[0]}"

        y_hats = self.predict(query_xs, None)
        return y_hats


    def estimate_L2_error(self, example_xs, example_ys):
        """ Estimates the L2 error of the function encoder on the example data. 
        This gives an idea if the example data lies in the span of the basis, or not.
        
        Args:
        example_xs: torch.tensor: The example input data used to compute a representation. Shape (n_functions, n_example_datapoints, input_size)
        example_ys: torch.tensor: The example output data used to compute a representation. Shape (n_functions, n_example_datapoints, output_size)
        
        Returns:
        torch.tensor: The estimated L2 error. Shape (n_functions,)
        """
        representation, gram = self.compute_representation(example_xs, example_ys, method="least_squares")
        f_hat_norm_squared = representation @ gram @ representation.T
        f_norm_squared = self._inner_product(example_ys, example_ys)
        l2_distance = torch.sqrt(f_norm_squared - f_hat_norm_squared)
        return l2_distance



    def train_model(self,
                    dataset: BaseDataset,
                    epochs: int,
                    progress_bar=True,
                    callback:BaseCallback=None,
                    **kwargs):
        """ Trains the function encoder on the dataset for some number of epochs.
        
        Args:
        dataset: BaseDataset: The dataset to train on.
        epochs: int: The number of epochs to train for.
        progress_bar: bool: Whether to show a progress bar.
        callback: BaseCallback: A callback to use during training. Can be used to test loss, etc. 
        
        Returns:
        list[float]: The losses at each epoch."""

        # verify dataset is correct
        dataset.check_dataset()
        
        # set device
        device = next(self.parameters()).device

        # Let callbacks few starting data
        if callback is not None:
            callback.on_training_start(locals())

        # method to use for representation during training
        assert self.method in ["inner_product", "least_squares"], f"Unknown method: {self.method}"

        losses = []
        bar = trange(epochs) if progress_bar else range(epochs)
        for epoch in bar:
            example_xs, example_ys, query_xs, query_ys, _ = dataset.sample()

            # train average function, if it exists
            # predict averages
            expected_yhats = self.predict_from_examples(example_xs, example_ys, query_xs)

            # compute average function loss
            average_function_loss = self._distance(expected_yhats, query_ys, squared=True).mean()


            # add loss components
            loss = average_function_loss
            
            # backprop with gradient clipping
            loss.backward()
            if (epoch+1) % self.gradient_accumulation == 0:
                norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.opt.step()
                self.opt.zero_grad()

            # callbacks
            if callback is not None:
                callback.on_step(locals())

        # let callbacks know its done
        if callback is not None:
            callback.on_training_end(locals())

    def _param_string(self):
        """ Returns a dictionary of hyperparameters for logging."""
        params = {}
        params["input_size"] = self.input_size
        params["output_size"] = self.output_size
        params["model_type"] = self.model_type
        params["regularization_parameter"] = self.regularization_parameter
        for k, v in self.model_kwargs.items():
            params[k] = v
        params = {k: str(v) for k, v in params.items()}
        return params

    @staticmethod
    def predict_number_params(input_size:tuple[int],
                             output_size:tuple[int],
                             n_basis:int=100,
                             model_type:Union[str, type]="MLP",
                             model_kwargs:dict=dict(),
                             use_residuals_method: bool = False,
                             *args, **kwargs):
        """ Predicts the number of parameters in the function encoder.
        Useful for ensuring all experiments use the same number of params"""
        n_params = 0
        if model_type == "MLP":
                n_params += MLP.predict_number_params(input_size, output_size, n_basis,  learn_basis_functions=False, **model_kwargs)
        elif model_type == "NeuralODE":
                n_params += NeuralODE.predict_number_params(input_size, output_size, n_basis,  learn_basis_functions=False, **model_kwargs)
        elif model_type == "ParallelMLP":
                n_params += ParallelMLP.predict_number_params(input_size, output_size, n_basis, learn_basis_functions=False, **model_kwargs)
        elif model_type == "Euclidean":
                n_params += Euclidean.predict_number_params(output_size, n_basis)
        elif model_type == "CNN":
                n_params += CNN.predict_number_params(input_size, output_size, n_basis, learn_basis_functions=False, **model_kwargs)
        elif isinstance(model_type, type):
                n_params += model_type.predict_number_params(input_size, output_size, n_basis, learn_basis_functions=False, **model_kwargs)
        else:
            raise ValueError(f"Unknown model type: '{model_type}'. Should be one of 'MLP', 'ParallelMLP', 'Euclidean', or 'CNN'")

        return n_params


    def forward_average_function(self, xs:torch.tensor) -> torch.tensor:
        """ Forward pass of the average function. """
        return self.average_function.forward(xs) if self.average_function is not None else None
from excursion import *
from excursion.optimizer.builders import *
import torch
import numpy as np

def test_init_nD():
    ndims = range(1,5,1) #troble starts at n=3
    tol = 1e-6
    torch.manual_seed(0) #for reproducibility

    device_opt = ['cuda', 'cpu', 'skcpu']
    dtype_opt = [torch.float64, np.float64]
    dtype_str = 'float64'
    n_initial_points = 3
    jump_start_opt = [True, False]
    model_type_opt = ['ExactGP', 'GridGP', 'SKLearnGP']
    model_fit_opt = ['Adam', 'LBFGS']
    likelihood_options = [0.0, 0.2]
    acq_opt = ['pes', 'mes']

    dtype = dtype_opt[1]
    device_str = device_opt[2]
    device = torch.device(device_str) if device_str != 'skcpu' else device_str
    jump_start = jump_start_opt[0]
    model_type = model_type_opt[2]
    fit_optimizer = model_fit_opt[0]
    acq_type = acq_opt[0]

    base_model_kwargs = {}
    base_model_kwargs['device'] = device
    base_model_kwargs['dtype'] = dtype
    base_model_kwargs['likelihood_type'] = 'GaussianLikelihood'
    base_model_kwargs['epsilon'] = likelihood_options[0]

    if jump_start:
        plus_iterations = 0
    else:
        plus_iterations = n_initial_points

    result_length = 0
    n_iterations = 15

    for n in ndims:
        print(f'############## {n}')
        print(f"bounding box { [[-1,1]]*n }")
        print(f"grid step size {[100]*n}")

        problem_details = ExcursionProblem(
        thresholds=[0.0], 
        bounding_box=[[-1, 1]]*n,
        ndim=n,
        grid_step_size=[100]*n,
        functions= [lambda x: x],
        )

        base_model = build_model(
            model_type, rangedef=problem_details.rangedef, **base_model_kwargs
        )
    
    
        #ensure init objects are not None
        assert type(problem_details) != type(None)
        assert type(base_model) != type(None)

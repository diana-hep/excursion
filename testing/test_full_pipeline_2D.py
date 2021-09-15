from excursion import *
from excursion.optimizer.builders import *
import torch
import numpy as np

import excursion.testcases.fast_2D as testcase_2D

def test_full_pipeline_2D():
    ndim = 2
    tol = 1e-3
    torch.manual_seed(42) #for reproducibility

    problem_details = ExcursionProblem(
        thresholds=[0.0],
        bounding_box=[[-1, 1]]*ndim,
        ndim=ndim,
        grid_step_size=[30]*ndim,
        functions= testcase_2D.test_functions
    )

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

    
    base_model = build_model(
        model_type, rangedef=problem_details.rangedef, **base_model_kwargs
    )
    acq_func = build_acquisition_func(acq_type, device_opt=device, dtype=dtype)
    
    optimizer = Optimizer(
        problem_details=problem_details,
        base_model=base_model,
        acq_func=acq_func,
        jump_start=jump_start,
        device=device_str,
        n_initial_points=n_initial_points,
        initial_point_generator="random",
        fit_optimizer=fit_optimizer,
        base_model_kwargs={},
        dtype=dtype_str,
        log=False,
    )

    """
    for x in range(n_iterations + plus_iterations):
        x = optimizer.ask()
        y = problem_details.functions[0](x)
        result = optimizer.tell(x, y)
    
    #ensure we find the level set at a fixed tol and new points are not None
    assert type(torch.abs(base_model.train_targets) <= tol) != type(None)
    """
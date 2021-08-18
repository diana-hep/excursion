import test_suite.test_functions.fast_2D as testcase_2D
import test_suite.test_functions.fast_1D as testcase_1D
import test_suite.test_functions.fast_3D as testcase_3D
from excursion import *
from excursion.optimizer.builders import *


def test_full_simple_dimensions():
    ndim = 3
    problem_three_details = ExcursionProblem(thresholds=[0.0], bounding_box=[[0.0, 1.5]] * ndim, ndim=ndim,
                                             grid_step_size=[30] * ndim, functions=testcase_3D.true_functions)
    ndim = 2
    problem_two_details = ExcursionProblem(thresholds=[0.0], bounding_box=[[0.0, 1.5]] * ndim, ndim=ndim,
                                           grid_step_size=[41] * ndim, functions=testcase_2D.true_functions)
    ndim = 1
    problem_one_details = ExcursionProblem(thresholds=[0.0], bounding_box=[[0.0, 1.5]], ndim=ndim, grid_step_size=[100],
                                           functions=testcase_1D.true_functions)

    device_opt = ['cuda', 'cpu']
    dtype = torch.float64
    n_initial_points = 3
    jump_start_opt = [True, False]
    model_type_opt = ['ExactGP', 'GridGP']
    model_fit_opt = ['Adam', 'LBFGS']
    likelihood_options = [0.0, 0.2]
    acq_opt = ['pes', 'mes']

    device_str = device_opt[0]
    device = torch.device(device_str)
    jump_start = jump_start_opt[0]
    model_type = model_type_opt[0]
    fit_optimizer = model_fit_opt[1]
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

    for problem_details in [problem_one_details, problem_two_details, problem_three_details]:
        # three toy examples
        base_model = build_model(model_type, grid=problem_details.rangedef, **base_model_kwargs)
        acq_func = build_acquisition_func(acq_type, device_opt=device, dtype=dtype)
        optimizer = Optimizer(problem_details=problem_details, base_model=base_model, acq_func=acq_func,
                              jump_start=jump_start, device=device_str, n_initial_points=n_initial_points,
                              initial_point_generator='random', fit_optimizer=fit_optimizer, base_model_kwargs={},
                              dtype='torch.float64', log=False)

        for x in range(n_iterations+plus_iterations):
            x = optimizer.ask()
            y = problem_details.functions[0](x)
            result = optimizer.tell(x, y)
        # ensure that the number of y values recorded for every problem type is equal to the number of times optimizer
        # should have updated the model during training
        result_length += len(base_model.train_targets)
    assert (result_length == len(range((n_iterations+n_initial_points)*3)))

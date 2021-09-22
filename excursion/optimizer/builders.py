from excursion.sampler import *
from excursion.models import ExcursionModel, SKLearnGP
from excursion.acquisition import *
from excursion.excursion import ExcursionProblem, ExcursionResult

# # move this into the excursion result, unless we add scikit learn implementation # #


def build_result(details: ExcursionProblem, acquisition, **kwargs):
    if kwargs['device'] == 'skcpu':
        X_pointsgrid = details.X_pointsgrid
        true_y = details.functions[0](X_pointsgrid)
    else:
        raise NotImplementedError("Only supports device 'SKCPU'")

    acquisition = acquisition # What if they passed in their own acq, then there is no string here.
    return ExcursionResult(ndim=details.ndim, thresholds=details.thresholds, true_y=true_y,
                           invalid_region=details.invalid_region, X_pointsgrid=details.X_pointsgrid,
                           X_meshgrid=details.X_meshgrid, rangedef=details.rangedef)


def build_sampler(generator: str or SampleGenerator, **kwargs):
    """Build a default random sample generator.
     For the special generator called "random" the return value is None.

     Parameters
     ----------
     generator : "random", "latin_sample", "latin_hypercube" or SampleGenerator instance"
         Should inherit from `skopt.sampler.SampleGenerator`.
     kwargs : dict
         Extra parameters provided to the generator at init time.
     """
    if generator is None:
        generator = "random"
    elif isinstance(generator, str):
        generator = generator.lower()
        allowed_generator = ["random"]
        if generator not in allowed_generator:
            raise ValueError("Valid strings for the generator parameter "
                             " are: 'latin', 'latin_hypercube', or 'random' not "
                             "%s." % generator)
    elif not isinstance(generator, SampleGenerator):
        raise ValueError("generator has to be a SampleGenerator or str."
                         "Got %s" % (str(type(generator))))

    if isinstance(generator, str):
        if generator == "random":
            generator = RandomChoice()
    generator.set_params(**kwargs)

    return generator


def build_acquisition_func(acq_function: str or AcquisitionFunction, **kwargs):
    """Build an acquisition function.
     For the special acq_function called "random" the return value is None.
     Parameters
     ----------
     function : "MES", "PES", or AcquisitionFunction instance"
         Should inherit from `skopt.sampler.SampleGenerator`.
     kwargs : dict
         Extra parameters provided to the acq_function at init time.
     """
    if acq_function is None:
        acq_function = "PES"
    elif isinstance(acq_function, str):
        acq_function = acq_function.lower()
        allowed_acq_funcs = ["pes"]
        if acq_function not in allowed_acq_funcs:
            raise ValueError("Valid strings for the acq_function parameter "
                             " are: %s, not %s." % (",".join(allowed_acq_funcs), acq_function))
    elif not isinstance(acq_function, AcquisitionFunction):
        raise TypeError("acq_function has to be an AcquisitionFunction. Got %s" % (str(type(acq_function))))

    if isinstance(acq_function, str):
        if acq_function == "pes":
            acq_function = SKPES()
    acq_function.set_params(**kwargs)

    return acq_function


def build_model(model: str or ExcursionModel, rangedef, **kwargs):
    """
    Build an acquisition function.
    For the special acq_function called "random" the return value is None.

     Parameters
     ----------
     model : "GPyTorchGP", "GridGP", or ExcursionModel instance"
         Should inherit from `excursion.models.ExcursionModel`.
     kwargs : dict
         Extra parameters provided to the acq_function at init time.
     """
    if model is None:
        model = "sklearngp"
    elif isinstance(model, str):
        model = model.lower()
        allowed_models = ["sklearngp"]
        if model not in allowed_models:
            raise ValueError("Valid strings for the model parameter are: 'SKLearnGP' not %s." % model)
    elif not isinstance(model, ExcursionModel):
        raise TypeError("model has to be an ExcursionModel or str. Got %s" % (str(type(model))))

    if isinstance(model, str):
        if model == "sklearngp":
            model = SKLearnGP(ndim=len(rangedef))

    model.set_params(**kwargs)

    return model


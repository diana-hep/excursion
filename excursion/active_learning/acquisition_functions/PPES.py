def PPES(gp, testcase, thresholds, x_candidate):
    """
    Calculates information gain of choosing x_candidadate as next point to evaluate.
    Performs this calculation with the Predictive Entropy Search approximation weighted by the posterior.
    Roughly,
    PES(x_candidate) = int Y(x)dx { H[Y(x_candidate)] - E_{S(x=j)} H[Y(x_candidate)|S(x=j)] }
    Notation: PES(x_candidate) = int dx H0 - E_Sj H1

    """

    # compute predictive posterior of Y(x) | train data
    raise NotImplmentedError(
        "Should be same strcture as PES but the cumulative info gain is weighted"
    )


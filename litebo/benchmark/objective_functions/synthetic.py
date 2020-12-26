import numpy as np
from typing import Optional

from litebo.config_space import ConfigurationSpace, Configuration


class BaseTestProblem(object):
    """
    Base class for synthetic test problems.
    """

    def __init__(self, config_space: ConfigurationSpace,
                 num_objs=1, num_constraints=0,
                 noise_std: Optional[float] = None,
                 optimal_value: Optional[float] = None):
        """
        Parameters
        ----------
        config_space : Config space of the test problem.

        noise_std : Standard deviation of the observation noise. (optional)

        optimal_value : Optimal value of the test problem. (optional)
        """
        self.config_space = config_space
        self.num_objs = num_objs
        self.num_constraints = num_constraints
        self.noise_std = noise_std
        self.optimal_value = optimal_value

    def evaluate(self, config: Configuration):
        result = self._evaluate(config)
        result

    def _evaluate(self, config: Configuration):
        """
        Evaluate the test function.

        Returns
        -------
        result : dict
            Result of the evaluation.
            result['objs'] is the objective value or an iterable of objective values
            result['constraints'] is an iterable of constraint values
        """
        raise NotImplementedError()


class Ackley(BaseTestProblem):
    r"""Ackley test function.

    d-dimensional function.

    :math:`f(x) = -A exp(-B sqrt(1/d sum_{i=1}^d x_i^2)) - exp(1/d sum_{i=1}^d cos(c x_i)) + A + exp(1)`

    f has one minimizer for its global minimum at :math:`x_{*} = (0, 0, ..., 0)` with
    :math:`f(x_{*}) = 0`.
    """

    def __init__(dim=10, lb=-10, ub=15,
                 noise_std: Optional[float] = None):
        self.dim = dim
        self.
        ackley_params = {
            'float': {f'x{i}': (-10, 15, 2.5) for i in range(1, 11)}
        }
        ackley_cs = ConfigurationSpace()
        ackley_cs.add_hyperparameters([UniformFloatHyperparameter(e, *ackley_params['float'][e]) for e in ackley_params['float']])



    def evaluate(config):
        """
        X is a d-dimensional vector.
        We take d = 10.
        """
        res = dict()
        X = np.array(list(config.get_dictionary().values()))
        a = 20
        b = 0.2
        c = 2*np.pi
        d = self.d
        s1 = -a*np.exp(-b*np.sqrt(1/d*(X**2).sum()))
        s2 = -np.exp(1/d*np.cos(c*X).sum())
        s3 = a + np.exp(1)
        res['objs'] = s1 + s2 + s3
        res['constraints'] = []
        return res

ackley_params = {
    'float': {f'x{i}': (-10, 15, 2.5) for i in range(1, 11)}
}
ackley_cs = ConfigurationSpace()
ackley_cs.add_hyperparameters([UniformFloatHyperparameter(e, *ackley_params['float'][e]) for e in ackley_params['float']])


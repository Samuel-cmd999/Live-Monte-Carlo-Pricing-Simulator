import numpy as np


class GeometricBrownianMotion:
    """
    Simulates stock price paths using Geometric Brownian Motion.
    """

    def __init__(self, S0, r, sigma, T, steps, simulations):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.steps = steps
        self.simulations = simulations
        self.dt = T / steps

    def generate_paths(self, antithetic=True):
        """
        Generate Monte Carlo stock paths.
        Uses antithetic variates for variance reduction if enabled.
        """

        if antithetic:
            Z = np.random.standard_normal((self.steps, self.simulations // 2))
            Z = np.concatenate((Z, -Z), axis=1)
        else:
            Z = np.random.standard_normal((self.steps, self.simulations))

        paths = np.zeros((self.steps, self.simulations))
        paths[0] = self.S0

        for t in range(1, self.steps):
            paths[t] = paths[t-1] * np.exp(
                (self.r - 0.5 * self.sigma**2) * self.dt +
                self.sigma * np.sqrt(self.dt) * Z[t]
            )

        return paths

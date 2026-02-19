import numpy as np
from pricer import EuropeanOptionPricer


class MonteCarloGreeks:
    """
    Computes option Greeks using finite-difference
    Monte Carlo pricing.
    """

    def __init__(self, S0, K, T, r, sigma,
                 steps=252, simulations=10000,
                 epsilon=1e-2):

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.steps = steps
        self.simulations = simulations
        self.epsilon = epsilon

    def _price(self, S0=None, r=None, sigma=None, T=None, option_type="call"):
        pricer = EuropeanOptionPricer(
            S0 or self.S0,
            self.K,
            T or self.T,
            r or self.r,
            sigma or self.sigma,
            self.steps,
            self.simulations
        )
        price, _, _ = pricer.monte_carlo_price(option_type)
        return price

    # --------------------
    # Greeks
    # --------------------

    def delta(self, option_type="call"):
        return (
            self._price(S0=self.S0 + self.epsilon, option_type=option_type)
            - self._price(S0=self.S0 - self.epsilon, option_type=option_type)
        ) / (2 * self.epsilon)

    def gamma(self, option_type="call"):
        return (
            self._price(S0=self.S0 + self.epsilon, option_type=option_type)
            - 2 * self._price(S0=self.S0, option_type=option_type)
            + self._price(S0=self.S0 - self.epsilon, option_type=option_type)
        ) / (self.epsilon ** 2)

    def vega(self, option_type="call"):
        return (
            self._price(sigma=self.sigma + self.epsilon,
                        option_type=option_type)
            - self._price(sigma=self.sigma - self.epsilon,
                          option_type=option_type)
        ) / (2 * self.epsilon)

    def theta(self, option_type="call"):
        return (
            self._price(T=self.T - self.epsilon, option_type=option_type)
            - self._price(T=self.T, option_type=option_type)
        ) / self.epsilon

    def rho(self, option_type="call"):
        return (
            self._price(r=self.r + self.epsilon, option_type=option_type)
            - self._price(r=self.r - self.epsilon, option_type=option_type)
        ) / (2 * self.epsilon)

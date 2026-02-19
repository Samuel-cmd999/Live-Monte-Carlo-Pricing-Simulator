import numpy as np
from scipy.stats import norm
from models import GeometricBrownianMotion


class EuropeanOptionPricer:

    def __init__(self, S0, K, T, r, sigma, steps=252, simulations=10000):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.steps = steps
        self.simulations = simulations

        self.model = GeometricBrownianMotion(
            S0, r, sigma, T, steps, simulations
        )

    def black_scholes(self, option_type="call"):
        d1 = (np.log(self.S0 / self.K) +
              (self.r + 0.5 * self.sigma**2) * self.T) / \
             (self.sigma * np.sqrt(self.T))

        d2 = d1 - self.sigma * np.sqrt(self.T)

        if option_type == "call":
            price = self.S0 * norm.cdf(d1) - \
                self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - \
                self.S0 * norm.cdf(-d1)

        return price

    def monte_carlo_price(self, option_type="call", antithetic=True):
        paths = self.model.generate_paths(antithetic=antithetic)
        ST = paths[-1]

        if option_type == "call":
            payoffs = np.maximum(ST - self.K, 0)
        else:
            payoffs = np.maximum(self.K - ST, 0)

        discounted = np.exp(-self.r * self.T) * payoffs

        price = np.mean(discounted)
        std_error = np.std(discounted) / np.sqrt(self.simulations)

        confidence_interval = (
            price - 1.96 * std_error,
            price + 1.96 * std_error
        )

        return price, confidence_interval, paths

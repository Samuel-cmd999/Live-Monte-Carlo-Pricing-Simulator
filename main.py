import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pricer import EuropeanOptionPricer
from greeks import MonteCarloGreeks


def animate_terminal(paths, pricer, option_type="call"):
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#0d0d0d")

    # Layout: price chart + histogram
    gs = fig.add_gridspec(2, 2)

    ax_price = fig.add_subplot(gs[:, 0])
    ax_hist = fig.add_subplot(gs[0, 1])

    ax_price.set_facecolor("#111111")
    ax_hist.set_facecolor("#111111")

    ax_price.set_title("Live Monte Carlo Stock Simulation",
                       fontsize=14, fontweight="bold")

    ax_price.set_xlabel("Time Steps")
    ax_price.set_ylabel("Stock Price")
    ax_price.grid(True, linestyle="--", alpha=0.3)

    num_paths_to_show = 8
    lines = []

    for i in range(num_paths_to_show):
        line, = ax_price.plot([], [], lw=1.2)
        lines.append(line)

    # Moving average line
    ma_line, = ax_price.plot([], [], lw=2, linestyle="--")

    ax_price.set_xlim(0, paths.shape[0])
    ax_price.set_ylim(np.min(paths) * 0.95, np.max(paths) * 1.05)

    # Text overlays
    ticker_text = ax_price.text(
        0.02, 0.95, "", transform=ax_price.transAxes,
        fontsize=12, verticalalignment="top"
    )

    greeks_text = ax_price.text(
        0.02, 0.80, "", transform=ax_price.transAxes,
        fontsize=10, verticalalignment="top"
    )

    # Compute Greeks once
    greek_calc = MonteCarloGreeks(
        pricer.S0, pricer.K, pricer.T,
        pricer.r, pricer.sigma,
        pricer.steps, pricer.simulations
    )

    delta = greek_calc.delta(option_type)
    gamma = greek_calc.gamma(option_type)
    vega = greek_calc.vega(option_type)
    theta = greek_calc.theta(option_type)
    rho = greek_calc.rho(option_type)

    ST = paths[-1]

    if option_type == "call":
        payoffs = np.maximum(ST - pricer.K, 0)
    else:
        payoffs = np.maximum(pricer.K - ST, 0)

    def init():
        for line in lines:
            line.set_data([], [])
        ma_line.set_data([], [])
        return lines + [ma_line]

    def update(frame):
        for i, line in enumerate(lines):
            line.set_data(range(frame), paths[:frame, i])

        # Moving average
        current_prices = paths[:frame, 0]
        if len(current_prices) > 5:
            ma = np.convolve(current_prices,
                             np.ones(5)/5, mode='valid')
            ma_line.set_data(range(len(ma)), ma)

        # Live ticker
        if frame > 0:
            current_price = paths[frame - 1, 0]
            ticker_text.set_text(
                f"Live Price: {current_price:.2f}"
            )

        greeks_text.set_text(
            f"Delta: {delta:.4f}\n"
            f"Gamma: {gamma:.4f}\n"
            f"Vega:  {vega:.4f}\n"
            f"Theta: {theta:.4f}\n"
            f"Rho:   {rho:.4f}"
        )

        return lines + [ma_line]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=paths.shape[0],
        init_func=init,
        interval=15,
        blit=True,
        repeat=False
    )

    # Histogram
    ax_hist.hist(payoffs, bins=50)
    ax_hist.set_title("Payoff Distribution")
    ax_hist.set_xlabel("Payoff")
    ax_hist.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def main():
    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2

    steps = 252
    simulations = 10000
    option_type = "call"

    pricer = EuropeanOptionPricer(
        S0, K, T, r, sigma,
        steps=steps,
        simulations=simulations
    )

    bs_price = pricer.black_scholes(option_type)
    mc_price, ci, paths = pricer.monte_carlo_price(option_type)

    print("\n========== OPTION PRICING ==========")
    print(f"Black-Scholes Price : {bs_price:.4f}")
    print(f"Monte Carlo Price   : {mc_price:.4f}")
    print(f"95% Confidence Int. : ({ci[0]:.4f}, {ci[1]:.4f})")
    print("====================================\n")

    animate_terminal(paths, pricer, option_type)


if __name__ == "__main__":
    main()

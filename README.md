# 📈 Monte Carlo Options Pricing & Risk Analytics Terminal
A modular quantitative finance engine for pricing European options using both analytical and simulation-based methods, enhanced with real-time visualization and risk sensitivity analysis.
---
## 🚀 Overview
This project implements a complete options pricing framework featuring:
- Geometric Brownian Motion (GBM) stock simulation
- Black–Scholes analytical pricing
- Monte Carlo pricing with confidence intervals
- Finite-difference Greeks computation
- Animated dark-theme trading-style visualization
- Payoff distribution analysis
The system is structured to reflect real-world quantitative research architecture.
---
## 🧠 Mathematical Foundations
### 1️⃣ Geometric Brownian Motion
Stock prices are modeled as:
dSₜ = rSₜ dt + σSₜ dWₜ
Discretized as:
Sₜ₊₁ = Sₜ · exp[(r − ½σ²)Δt + σ√Δt Z]
Where:
- r = risk-free rate  
- σ = volatility  
- Z ~ N(0,1)
---
### 2️⃣ Black–Scholes Pricing
European call option:
C = S₀N(d₁) − Ke^(−rT)N(d₂)
Used as a benchmark for Monte Carlo validation.
---
### 3️⃣ Monte Carlo Pricing
1. Simulate thousands of price paths  
2. Compute payoff at maturity  
3. Discount expected payoff  
4. Estimate standard error & confidence interval  
---
### 4️⃣ Greeks (Risk Sensitivities)
Finite-difference approximations are used to compute:
- Delta
- Gamma
- Vega
- Theta
- Rho

This reflects real-world risk management techniques.

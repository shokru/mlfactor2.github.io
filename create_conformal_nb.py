#!/usr/bin/env python3
"""Script to create chap_conformal.ipynb"""

import json
import uuid

def cell_id():
    return str(uuid.uuid4())[:8]

def md_cell(source):
    if isinstance(source, str):
        source = [source]
    return {
        "cell_type": "markdown",
        "id": cell_id(),
        "metadata": {},
        "source": source
    }

def code_cell(source):
    if isinstance(source, str):
        source = [source]
    return {
        "cell_type": "code",
        "id": cell_id(),
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None
    }

cells = []

# CELL 0 — frontmatter + title
cells.append(md_cell(
"""---
numbering:
  enumerator: "15.%s"
  headings: true
  equation:
    template: "%s"
---
(chap_conformal)=
# Uncertainty quantification and conformal prediction"""
))

# CELL 1 — Introduction
cells.append(md_cell(
"""Standard machine learning models produce point estimates. They offer no indication of reliability. An investor who knows a forecast is imprecise can reduce position sizes and avoid costly errors. Uncertainty quantification (UQ) addresses this gap. UQ attaches a measure of confidence to every prediction. It therefore transforms a raw signal into actionable information.

This chapter covers three traditions in UQ as applied to asset pricing. The first is conformal prediction, which delivers finite sample, distribution free coverage guarantees. The second is ambiguity aversion, which models investor behavior under Knightian uncertainty about the return generating process. The third is machine forecast disagreement, which treats dispersion across ML models as a proxy for hard to price risk. Together, these traditions offer a rich toolkit for practitioners who take model uncertainty seriously."""
))

# CELL 2 — Sources of prediction uncertainty
cells.append(md_cell(
"""(sec_uncertainty_types)=
## Sources of prediction uncertainty

Prediction uncertainty has two conceptually distinct sources. **Aleatoric** uncertainty is irreducible. It arises from idiosyncratic shocks that no model can eliminate regardless of sample size. In factor investing, idiosyncratic return variance is a canonical example. No amount of additional data removes it.

**Epistemic** uncertainty is reducible in principle. It stems from finite data and from limitations of the chosen model class. A model trained on few observations will disagree with a model trained on many. That disagreement shrinks as data accumulate. The two components call for different responses: aleatoric uncertainty should inform position sizing, while epistemic uncertainty motivates model averaging or Bayesian approaches.

Decomposing these components formally is not straightforward. @ahdritz2024calibration introduce higher order calibration to achieve this decomposition with formal guarantees. Their framework requires no distributional assumptions on the data generating process. It provides a provably valid separation between aleatoric and epistemic contributions to total prediction error."""
))

# CELL 3 — Conformal prediction intro
cells.append(md_cell(
"""(sec_conformal)=
## Conformal prediction

Conformal prediction is a framework for constructing prediction intervals that carry a finite sample, distribution free coverage guarantee. Under exchangeability of the data, the coverage holds exactly regardless of the underlying distribution and regardless of the model used to generate predictions. The foundational theory is laid out in @vovk2005algorithmic. A concise and accessible introduction is provided by @angelopoulos2023gentle.

The key appeal in financial applications is the absence of parametric assumptions. Return distributions have heavy tails and change over time. Classical interval methods that assume normality are unreliable. Conformal methods sidestep this entirely. @liao2025uncertainty apply related ideas to asset pricing and show that neural network forecasts share an asymptotic distribution with nonparametric estimators, enabling closed form standard errors and bootstrap confidence intervals. They find that accounting for forecast uncertainty improves out of sample portfolio performance."""
))

# CELL 4 — Split conformal algorithm
cells.append(md_cell(
r"""### The split conformal algorithm

The split conformal algorithm proceeds in three steps. First, the data are partitioned into a training set, a calibration set, and a test set. Second, a model $\hat{f}$ is fitted on the training set and used to predict on the calibration set. Third, nonconformity scores are computed, a quantile is derived from these scores, and intervals are formed for the test observations.

The coverage guarantee states that for a chosen miscoverage level $\alpha \in (0,1)$,

```{math}
:label: eq_conformal_coverage
P(y_{n+1} \in \hat{C}(x_{n+1})) \geq 1 - \alpha.
```

The nonconformity score for calibration observation $i$ is the absolute residual of the model:

```{math}
:label: eq_nonconformity
s_i = |y_i - \hat{f}(x_i)|.
```

The prediction interval for a new observation $x_{n+1}$ is then

```{math}
:label: eq_conformal_interval
\hat{C}(x_{n+1}) = [\hat{f}(x_{n+1}) - \hat{q},\; \hat{f}(x_{n+1}) + \hat{q}],
```

where $\hat{q}$ is the $\lceil(n+1)(1-\alpha)\rceil/n$ quantile of the calibration scores. This inflated quantile accounts for the finite size of the calibration set and is what delivers the guarantee in {eq}`eq_conformal_coverage`. The interval is symmetric around the point prediction. It grows wider as $\alpha$ decreases (tighter coverage) and narrows as the calibration set grows."""
))

# CELL 5 — transition to code
cells.append(md_cell(
"""The code below illustrates split conformal on simulated stock return data. We generate a panel with N stocks over T periods using K firm characteristics. A ridge regression is fitted on the training set. We then compute conformal intervals and check empirical coverage."""
))

# CELL 6 — data generation
cells.append(code_cell(
"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

np.random.seed(2024)

N, T, K = 300, 80, 8
beta_true = np.array([0.5, -0.35, 0.25, 0.15, 0.1, 0.05, 0., 0.])
sigma_e   = 1.0

X = np.random.randn(N * T, K)
y = X @ beta_true + sigma_e * np.random.randn(N * T)
print(f"Panel: {N} stocks x {T} periods, {K} characteristics")
print(f"Signal-to-noise ratio: {np.var(X @ beta_true) / sigma_e**2:.2f}")"""
))

# CELL 7 — conformal intervals
cells.append(code_cell(
"""n    = len(y)
n_tr = int(0.60 * n)
n_ca = int(0.20 * n)

Xtr, ytr = X[:n_tr],              y[:n_tr]
Xca, yca = X[n_tr:n_tr+n_ca],     y[n_tr:n_tr+n_ca]
Xte, yte = X[n_tr+n_ca:],         y[n_tr+n_ca:]

sc = StandardScaler().fit(Xtr)
Xtr_s = sc.transform(Xtr)
Xca_s = sc.transform(Xca)
Xte_s = sc.transform(Xte)

mdl    = Ridge(alpha=1.0).fit(Xtr_s, ytr)
scores = np.abs(yca - mdl.predict(Xca_s))

alpha  = 0.10
n_ca   = len(scores)
q_hat  = np.quantile(scores, np.ceil((n_ca + 1) * (1 - alpha)) / n_ca)

ypred = mdl.predict(Xte_s)
lo    = ypred - q_hat
hi    = ypred + q_hat
cov   = np.mean((yte >= lo) & (yte <= hi))

print(f"Target coverage    : {1 - alpha:.0%}")
print(f"Empirical coverage : {cov:.1%}")
print(f"Interval half-width: {q_hat:.3f}")"""
))

# CELL 8 — local uncertainty and UA sorting
cells.append(code_cell(
"""nn = NearestNeighbors(n_neighbors=30).fit(Xca_s)
_, idx_nn    = nn.kneighbors(Xte_s)
sigma_loc    = np.array([scores[i].std() for i in idx_nn])

Ncs          = N
pred_cs, y_cs, sig_cs = ypred[:Ncs], yte[:Ncs], sigma_loc[:Ncs]

q10, q90 = np.percentile(pred_cs, [10, 90])
ls_std   = y_cs[pred_cs >= q90].mean() - y_cs[pred_cs <= q10].mean()

hi_cs  = pred_cs + sig_cs
lo_cs  = pred_cs - sig_cs
q10u   = np.percentile(lo_cs, 10)
q90u   = np.percentile(hi_cs, 90)
ls_ua  = y_cs[hi_cs >= q90u].mean() - y_cs[lo_cs <= q10u].mean()

print(f"Standard long short return            : {ls_std:.4f}")
print(f"Uncertainty adjusted long short return: {ls_ua:.4f}")"""
))

# CELL 9 — visualization
cells.append(code_cell(
r"""import os
os.makedirs("images", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: prediction intervals on 60 stocks sorted by prediction
n_show = 60
sort_idx = np.argsort(pred_cs)[:n_show]
x_pos = np.arange(n_show)

axes[0].fill_between(
    x_pos,
    lo[sort_idx],
    hi[sort_idx],
    alpha=0.3,
    color="steelblue",
    label="Conformal interval"
)
axes[0].scatter(x_pos, pred_cs[sort_idx], s=15, color="steelblue", label="Point prediction", zorder=3)
axes[0].scatter(x_pos, y_cs[sort_idx], s=15, color="tomato", marker="x", label="Realized return", zorder=4)
axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_xlabel("Stock (sorted by prediction)")
axes[0].set_ylabel("Return")
axes[0].set_title("Conformal prediction intervals")
axes[0].legend(fontsize=8)

# Right panel: bar chart comparing standard vs uncertainty-adjusted LS
bar_labels = ["Standard\nlong short", "Uncertainty\nadjusted"]
bar_values = [ls_std, ls_ua]
colors = ["steelblue", "darkorange"]
axes[1].bar(bar_labels, bar_values, color=colors, width=0.5, edgecolor="black", linewidth=0.8)
axes[1].set_ylabel("Average return (simulated)")
axes[1].set_title("Portfolio return comparison")
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
for i, v in enumerate(bar_values):
    axes[1].text(i, v + 0.002 * np.sign(v), f"{v:.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=10)

plt.tight_layout()
plt.savefig("images/conformal_coverage.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure saved to images/conformal_coverage.png")"""
))

# CELL 10 — Uncertainty-adjusted portfolio sorting
cells.append(md_cell(
"""(sec_ua_sorting)=
## Uncertainty adjusted portfolio sorting

The standard approach in empirical asset pricing is to sort stocks on point predictions and form decile portfolios. The top decile goes long and the bottom decile goes short. This ignores the reliability of each prediction. A stock at the top of the predicted return distribution may have a very wide confidence interval. Its prediction could easily be wrong.

@liu2026sorting propose uncertainty adjusted sorting. Instead of ranking stocks on their point prediction $\hat{f}(x)$, investors go long on stocks with the highest upper prediction bounds and short on stocks with the lowest lower prediction bounds. This selects confident predictions. A stock enters the long leg only if its lower bound is also elevated. A stock enters the short leg only if its upper bound is also low.

The key empirical finding from @liu2026sorting is that performance gains relative to standard sorting come mainly from reduced portfolio volatility rather than from higher average returns. The mechanism is asset level uncertainty rather than aggregate market uncertainty. When individual stock uncertainty is high, the sorting procedure naturally avoids those stocks, reducing idiosyncratic risk in the portfolio.

@allena2025confident reaches similar conclusions from a different angle. He derives ex ante confidence intervals from linear and ML models including Lasso, Elastic Net, Ridge and neural networks. Confident high low strategies, which trade only on stocks with precise forecasts, systematically outperform their standard counterparts. The result holds across all model classes. This convergence of evidence across methods suggests that incorporating forecast precision into portfolio construction is a robust improvement over point prediction sorting."""
))

# CELL 11 — Machine forecast disagreement
cells.append(md_cell(
"""(sec_mfd)=
## Machine forecast disagreement

@bali2025disagreement introduce the concept of machine forecast disagreement (MFD). The idea draws on the classical analyst disagreement literature but replaces human analysts with ML model specifications. Different model architectures, different hyperparameters, and different feature sets produce different return forecasts for the same stock. The dispersion of these forecasts measures how much the "ML crowd" disagrees about a stock's prospects.

The analogy with investor disagreement is natural. In models with short sale constraints, disagreement among investors leads to overpricing because the most optimistic investors set prices while pessimists cannot easily arbitrage. The same logic applies here. Stocks with high MFD are harder to price, harder to arbitrage, and consequently overpriced.

Empirically, @bali2025disagreement find that stocks in the highest MFD decile earn returns roughly 14% per year lower than stocks in the lowest MFD decile on a value weighted basis. This spread is robust to standard risk controls. MFD also outperforms traditional analyst forecast dispersion measures in predicting cross sectional returns. The reason is that ML models cover the full cross section of stocks including those with no analyst coverage, and they do so at high frequency with consistent methodology.

The connection to limits to arbitrage is important. High MFD stocks tend to be small, illiquid, and costly to short. The same frictions that prevent arbitrageurs from correcting overpricing are correlated with the conditions that generate high ML disagreement. MFD therefore acts as a joint measure of uncertainty and arbitrage difficulty."""
))

# CELL 12 — Ambiguity aversion
cells.append(md_cell(
"""(sec_ambiguity)=
## Ambiguity aversion and portfolio choice

Knight distinguished risk, where probabilities are known, from ambiguity, where they are not. ML models introduce a specific form of ambiguity: different plausible models of the return generating process produce different optimal portfolios. @ghysels2022ambiguity quantify this ML ambiguity via bootstrap resampling. Each bootstrap resample yields a different trained model and hence a different forecast. The spread across resamples separately captures upside and downside scenarios, a concept the authors term directional uncertainty. Investors who internalize this ambiguity by adopting pessimistic positions where disagreement is high achieve better Sharpe ratios than those who rely on a single model estimate.

@hara2022ambiguity provide a theoretical foundation for thinking about implied ambiguity. They derive a one to one mapping between the degree of implied ambiguity inferred from portfolio inefficiency, the observed Sharpe ratio, and pricing errors in the cross section. The result holds under a smooth ambiguity model where the representative investor has preferences over both risk and ambiguity. Calibrating the model to US data, they find significant ambiguity aversion. This suggests that the equity premium itself reflects compensation not only for risk but for Knightian uncertainty about model parameters.

@bianchi2026uncertainty take a Bayesian neural network (BNN) approach to the same problem. Rather than mapping characteristics to expected returns, BNNs map firm characteristics directly to portfolio weights by maximizing expected utility under the posterior distribution over network parameters. The posterior induces a distribution over optimal weights. Averaging over this distribution yields more stable allocations than point estimates from standard deep learning. No trade regions arise naturally because the posterior uncertainty about a stock's weight may be so large that no position is justified. Posterior sparsity further reduces turnover. The resulting strategy outperforms linear models and standard neural networks, particularly after transaction costs, suggesting that parameter uncertainty is economically significant."""
))

# CELL 13 — Time-varying uncertainty
cells.append(md_cell(
"""(sec_timevarying_unc)=
## Time varying uncertainty and signal selection

The optimal weighting of predictive signals changes over time. During periods of high economic uncertainty, short term signals become noisier while long term structural relationships remain more stable. @chen2026uncertainty formalize this intuition with U-LASSO, an adaptive penalization scheme that mixes long term and short term predictive signals based on the current level of economic uncertainty.

The mechanism operates through two distinct channels. The noisy signal channel, driven by real and financial uncertainty, pushes investors toward stable long term signals. These signals are less affected by transient shocks and provide more reliable forecasts when the environment is volatile. The volatile state channel, driven by macro uncertainty and the VIX, pushes investors toward short term signals because the economy is changing rapidly enough that long term averages become stale.

@chen2026uncertainty show that economic uncertainty explains nearly 30% of the variation in the optimal signal horizon weight. This is a large share of explained variation for a single variable and underlines the economic importance of the uncertainty regime. U-LASSO outperforms both traditional penalized regressions and standard ML benchmarks in out of sample portfolio performance. The result holds across different uncertainty measures and subperiods, suggesting robustness beyond any particular calibration choice."""
))

# CELL 14 — LLM uncertainty
cells.append(md_cell(
"""(sec_llm_unc)=
## Uncertainty in large language model predictions

Large language models (LLMs) are increasingly used to classify financial text and generate investment signals. However, standard LLM outputs come with no natural measure of confidence. The model simply returns a predicted label or score. @chen2026llm address this gap by constructing an entropy measure from the inner conditional probabilities over possible outputs rather than from the generated text itself.

The approach exploits the token level probability distributions that LLMs compute internally. High entropy means the model is uncertain about which output to generate. Low entropy means the model assigns high probability to one particular output. Crucially, this inner probability differs from the confidence that some models report in their generated text. Decoding biases cause models to express false certainty in their generated language even when their internal distributions are diffuse. The entropy measure bypasses this problem.

@chen2026llm show that high confidence predictions (low entropy) are systematically more accurate in financial text classification tasks. A portfolio built on high confidence LLM signals achieves a Sharpe ratio approximately 20% above the benchmark. Low confidence predictions contribute no excess return. This finding implies that filtering LLM signals by their internal uncertainty is a meaningful source of alpha, and that practitioners should not rely on LLM self assessment to determine when to trust the model."""
))

# CELL 15 — Exercises
cells.append(md_cell(
"""## Exercises

1. Using the simulated data above, vary the miscoverage level alpha from 0.05 to 0.30. Plot the empirical coverage and interval width as a function of alpha. Does the finite sample guarantee hold throughout?

2. Replace the ridge regression with a gradient boosting model. Recompute the conformal quantile and assess whether empirical coverage is maintained.

3. Implement an asymmetric conformal interval where the long leg uses the upper bound $\\hat{f}(x) + \\hat{q}$ and the short leg uses the lower bound $\\hat{f}(x) - \\hat{q}$. Compare the resulting portfolio with the uncertainty adjusted sort described in the chapter.

4. Simulate machine forecast disagreement by fitting five ridge regressions on different bootstrap samples of the training data. Compute the dispersion of predictions and test whether high disagreement stocks earn lower simulated returns.

5. Discuss conceptually how the U-LASSO of @chen2026uncertainty could be implemented in a panel regression setting. What observable variable would you use as the uncertainty measure, and how would it enter the penalization scheme?"""
))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

output_path = "/Users/coqueret/Documents/IT/BOOK_ML/v2/chap_conformal.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook written: {output_path}")
print(f"Total cells: {len(cells)}")
for i, c in enumerate(cells):
    ctype = c["cell_type"]
    snippet = (c["source"] if isinstance(c["source"], str) else "".join(c["source"]))[:60].replace("\n", " ")
    print(f"  Cell {i:2d} [{ctype[:4]}]: {snippet}")

# Factor Model Design: Architecture & Methodology

## Overview

This document outlines the design principles for constructing a multi-factor equity model that combines fundamental factors (valuation, growth, quality), price-based factors (momentum, volatility), and alternative data (short interest). The model prioritizes **interpretability** and **real-world usability** while managing multicollinearity through targeted orthogonalization.

---

## Core Principles

### 1. Interpretability Over Mathematical Elegance
- Each factor must have a clear economic story explainable in 10 words
- Avoid black-box transformations (full PCA, symmetric orthogonalization) that destroy factor meaning
- Factor returns should be communicable to portfolio managers and clients

### 2. Hierarchical Construction
- Build factors in layers from specific → general
- Orthogonalize only when necessary to remove known structural relationships
- Preserve the economic interpretation of primary factors

### 3. Variance Decomposition
- Assign each factor a score proportional to its contribution in reducing residual variance
- Use this to validate factor relevance and identify redundant factors
- Track contributions over time to detect regime changes

---

## Factor Architecture

### Layer 1: Primary Themes (Interpretable Composites)

Build 7 core factors with clear economic meaning:

#### 1. **Market**
- Equal-weight portfolio return or market index (e.g., SPY)
- Captures systematic risk

#### 2. **Size**
- `log(market_cap)` - NOT raw market cap
- **Why log?**
  - Captures proportional differences (2x matters more for small caps than mega caps)
  - Normal-ish distribution vs extreme right skew of raw caps
  - Industry standard (Fama-French, Barra, AQR all use logs)
  - Interpretable regression coefficients
- No orthogonalization - Size is a fundamental structural factor

#### 3. **Value**
- Composite: IC-weighted average of `sP/S`, `sP/E`, `P/S`, `P/Ee`, `P/GP`
- Classic valuation multiples
- Lower values = cheaper = higher expected returns

#### 4. **Growth**
- Composite: IC-weighted average of `GS`, `GE`, `HSG`, `SGD`, `GGP`
- Forward growth (`GS`, `GE`) + historical quality (`HSG`) + acceleration (`SGD`)
- **Apply targeted orthogonalization:** Residualize vs Size (small caps grow faster structurally)
  ```python
  Growth_resid = Growth - β_size * Size
  ```

#### 5. **Quality**
- Composite: IC-weighted average of `OM`, `ROI`, `ROE`, `r2_S`, `r2_E`
- Profitability and earnings stability
- **Apply targeted orthogonalization:** Residualize vs Size (large caps have stable margins)
  ```python
  Quality_resid = Quality - β_size * Size
  ```

#### 6. **Momentum**
- Existing price momentum composite
- No orthogonalization (momentum is a distinct behavioral factor)

#### 7. **Short Interest**
- Existing short interest composite
- Captures crowding and sentiment

---

### Within-Theme Aggregation Methods

For each composite (Value, Growth, Quality), choose one:

#### Option A: Equal-Weight Average
```python
Growth = (GS + GE + HSG + SGD + GGP) / 5
```
- **Pros:** Simple, transparent, no data mining
- **Cons:** Treats all metrics equally regardless of predictive power

#### Option B: IC-Weighted Average
```python
# Compute historical IC (Information Coefficient) for each metric
IC_GS = rolling_corr(GS, forward_returns, window=252)
IC_GE = rolling_corr(GE, forward_returns, window=252)
...

# Weight by absolute IC
weights = abs([IC_GS, IC_GE, IC_HSG, IC_SGD, IC_GGP])
weights = weights / sum(weights)  # normalize

Growth = sum(w_i * metric_i for w_i, metric_i in zip(weights, metrics))
```
- **Pros:** Data-driven, emphasizes metrics that actually predict returns
- **Cons:** Requires backtest, risk of overfitting, weights drift over time

#### Option C: First Principal Component
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
Growth = pca.fit_transform(growth_metrics_matrix)
```
- **Pros:** Captures maximum variance
- **Cons:** Hard to interpret, unstable loadings, not recommended for primary factors

**Recommendation:** Start with **IC-weighting** using 252-day rolling correlations. Fall back to equal-weight if ICs are unstable or negative.

---

### Layer 2: Targeted Orthogonalization

**Philosophy:** Only remove known structural dependencies, not all correlations.

#### When to Orthogonalize
- **Size effects:** Small caps are structurally more volatile, faster-growing, lower-quality
- **Sector effects:** If you have sector-specific factors (optional)

#### When NOT to Orthogonalize
- Value vs Growth correlation (economically meaningful - value stocks grow slower)
- Quality vs Growth correlation (high-quality companies often sustain growth)
- Momentum vs Volatility correlation (trending stocks are often volatile)

Accept these correlations - they reflect real economic relationships. Multicollinearity inflates standard errors but doesn't bias coefficients.

#### Implementation: Size-Neutralization
```python
from sklearn.linear_model import LinearRegression

# Orthogonalize Growth vs Size
model = LinearRegression().fit(Size.values.reshape(-1, 1), Growth.values)
Growth_resid = Growth - model.predict(Size.values.reshape(-1, 1))

# Orthogonalize Quality vs Size
model = LinearRegression().fit(Size.values.reshape(-1, 1), Quality.values)
Quality_resid = Quality - model.predict(Size.values.reshape(-1, 1))

# Orthogonalize Volatility vs Size (optional)
model = LinearRegression().fit(Size.values.reshape(-1, 1), Volatility.values)
Vol_resid = Volatility - model.predict(Size.values.reshape(-1, 1))
```

**Result:** 
- **Growth_resid** = growth signal independent of size
- **Quality_resid** = quality signal independent of size
- Original Size factor unchanged

---

### Layer 3: Interaction Terms (Optional)

Add interaction factors to capture non-linear effects:

#### Example Interactions
```python
# Crowded Momentum = high momentum + high short interest (fragile)
Crowded_Momentum = Momentum * ShortInterest

# Quality Growth = high quality + high growth (sustainable growers)
Quality_Growth = Quality_resid * Growth_resid

# Small-Cap Value = low size + high value (classic anomaly)
SmallCap_Value = (1 / Size) * Value  # inverse size for small = high
```

**Use sparingly** - each interaction:
- Dilutes interpretability
- Adds parameters (overfitting risk)
- Requires economic justification

**Validation:** Only include if interaction term has:
- Clear economic story
- Significant t-stat in regression (|t| > 2)
- Positive out-of-sample Sharpe

---

## Feature Preprocessing Pipeline

### Step 1: Winsorization (Before Anything Else)
```python
def winsorize(x, lower=0.01, upper=0.99):
    """Clip extreme outliers to reduce influence of data errors."""
    return np.clip(x, np.quantile(x, lower), np.quantile(x, upper))

# Apply to all raw metrics
metrics_winsorized = {k: winsorize(v) for k, v in raw_metrics.items()}
```

**Why winsorize?**
- Financial data has fat tails and occasional data errors
- Standard scaling assumes normality - winsorizing helps
- 1st/99th percentile clip is standard (2.5%/97.5% for more aggressive)

---

### Step 2: Composite Construction
```python
# Example: Growth composite via IC-weighting
growth_metrics = ['GS', 'GE', 'HSG', 'SGD', 'GGP']
ics = {m: compute_rolling_ic(metrics_win[m], forward_returns) for m in growth_metrics}

# Take most recent IC
ic_current = {m: ics[m].iloc[-1] for m in growth_metrics}

# Weight by absolute IC
weights = {m: abs(ic_current[m]) for m in growth_metrics}
weight_sum = sum(weights.values())
weights = {m: w / weight_sum for m, w in weights.items()}

# Construct composite
Growth = sum(weights[m] * metrics_win[m] for m in growth_metrics)
```

---

### Step 3: Targeted Orthogonalization
```python
# Size-neutralize Growth and Quality
Growth_resid = residualize(Growth, Size)
Quality_resid = residualize(Quality, Size)

def residualize(y, X):
    """Return residuals from regressing y on X."""
    X = X.values.reshape(-1, 1) if X.ndim == 1 else X
    y = y.values.reshape(-1, 1) if y.ndim == 1 else y
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ beta
```

---

### Step 4: Standardization
```python
def standardize(x):
    """Z-score: mean=0, std=1."""
    return (x - x.mean()) / x.std()

# Apply to all factors
factors_scaled = {
    'Market': Market,  # Don't scale market - keep as return
    'Size': standardize(log_Size),
    'Value': standardize(Value),
    'Growth': standardize(Growth_resid),
    'Quality': standardize(Quality_resid),
    'Momentum': standardize(Momentum),
    'ShortInterest': standardize(ShortInterest)
}
```

**Alternative: Rank Transformation** (for non-stationarity robustness)
```python
def rank_transform(x):
    """Convert to fractional ranks in [-1, 1]."""
    ranks = x.rank(pct=True)  # percentile ranks [0, 1]
    return (ranks - 0.5) * 2   # rescale to [-1, 1]

factors_ranked = {k: rank_transform(v) for k, v in factors.items()}
```

Rank transformation is more robust to regime changes but loses magnitude information. **Recommendation:** Start with z-scores, switch to ranks if factor returns are unstable.

---

## Variance Decomposition: Measuring Factor Contributions

### Goal
Assign each factor a score representing its contribution to explaining return variance. Use this to:
- Validate factor relevance
- Identify redundant factors (drop if contribution < 2%)
- Track regime shifts (if Growth contribution drops from 8% → 2%, investigate)

---

### Method 1: Partial R² (Recommended)

**Concept:** Measure variance explained **uniquely** by each factor (after controlling for all others).

```python
from sklearn.linear_model import LinearRegression

def partial_r2_contributions(returns, factors):
    """
    Compute each factor's unique contribution to R².
    
    Args:
        returns: pd.Series of cross-sectional returns
        factors: dict of {factor_name: pd.Series}
    
    Returns:
        partial_r2: dict of {factor_name: unique_r2}
        r2_full: total R² of full model
    """
    factor_names = list(factors.keys())
    X_full = np.column_stack([factors[f] for f in factor_names])
    
    # Full model R²
    model_full = LinearRegression().fit(X_full, returns)
    r2_full = model_full.score(X_full, returns)
    
    partial_r2 = {}
    
    for target in factor_names:
        # Refit without target factor
        other_factors = [f for f in factor_names if f != target]
        X_reduced = np.column_stack([factors[f] for f in other_factors])
        model_reduced = LinearRegression().fit(X_reduced, returns)
        r2_reduced = model_reduced.score(X_reduced, returns)
        
        # Unique contribution = drop in R² when target is removed
        partial_r2[target] = r2_full - r2_reduced
    
    return partial_r2, r2_full
```

**Interpretation:**
- `partial_r2['Growth'] = 0.05` → Growth uniquely explains 5% of variance
- `sum(partial_r2.values()) ≤ r2_full` because overlapping variance is excluded

**Normalization** (to sum to 100% of explained variance):
```python
def normalize_contributions(partial_r2, r2_full):
    """
    Allocate shared variance proportionally to each factor's unique contribution.
    """
    total_partial = sum(partial_r2.values())
    shared_variance = r2_full - total_partial
    
    normalized = {}
    for factor, pr2 in partial_r2.items():
        # Unique + proportional share of overlap
        normalized[factor] = pr2 + (pr2 / total_partial) * shared_variance if total_partial > 0 else 0
    
    return normalized

# Usage
partial_r2, r2_full = partial_r2_contributions(returns, factors)
normalized_contrib = normalize_contributions(partial_r2, r2_full)

print(f"Total R²: {r2_full:.2%}")
for factor, contrib in normalized_contrib.items():
    print(f"{factor}: {contrib:.2%}")
```

**Pros:**
- Fast: O(n) regressions for n factors
- Intuitive: "What do I lose if I drop this factor?"
- Handles multicollinearity gracefully

**Cons:**
- Sum of partials < full R² (shared variance needs allocation)
- Sensitive to factor ordering if factors are highly correlated

---

### Method 2: Shapley R² (Theoretically Perfect, Computationally Expensive)

**Concept:** Average marginal R² contribution across all possible factor orderings.

```python
from itertools import combinations

def shapley_r2_contribution(returns, factors):
    """
    Compute Shapley values for R² - each factor's average marginal contribution.
    
    Warning: O(2^n) complexity - only feasible for n ≤ 10 factors.
    """
    n_factors = len(factors)
    factor_names = list(factors.keys())
    shapley_values = {f: 0 for f in factor_names}
    
    for target_factor in factor_names:
        marginal_r2s = []
        other_factors = [f for f in factor_names if f != target_factor]
        
        # For each subset size k
        for k in range(n_factors):
            # All k-sized subsets NOT containing target
            for subset in combinations(other_factors, k):
                # R² without target
                if subset:
                    X_without = np.column_stack([factors[f] for f in subset])
                    r2_without = LinearRegression().fit(X_without, returns).score(X_without, returns)
                else:
                    r2_without = 0
                
                # R² with target added
                subset_with_target = list(subset) + [target_factor]
                X_with = np.column_stack([factors[f] for f in subset_with_target])
                r2_with = LinearRegression().fit(X_with, returns).score(X_with, returns)
                
                # Marginal contribution in this ordering
                marginal_r2s.append(r2_with - r2_without)
        
        # Average across all orderings
        shapley_values[target_factor] = np.mean(marginal_r2s)
    
    return shapley_values
```

**Key Property:** `sum(shapley_values) = R²_full_model` (exactly)

**Pros:**
- Theoretically rigorous (game theory)
- Fair allocation of shared variance
- Order-independent

**Cons:**
- Computationally expensive: 2^7 = 128 regressions for 7 factors (manageable)
- For 10+ factors, need approximations (e.g., sampling orderings)

---

### Method 3: Hierarchical Variance Decomposition (Layered Models)

**Concept:** Add factors sequentially, measure incremental adjusted R².

```python
def hierarchical_variance_decomposition(returns, factor_layers):
    """
    Add factors in priority order, track incremental R² at each step.
    
    Args:
        returns: pd.Series
        factor_layers: list of (name, values) in priority order
                       e.g., [('Market', market), ('Size', size), ('Value', value), ...]
    
    Returns:
        dict of {factor_name: incremental_adjusted_r2}
    """
    cumulative_factors = []
    decomposition = {}
    prev_r2_adj = 0
    
    for factor_name, factor_values in factor_layers:
        cumulative_factors.append(factor_values)
        X = np.column_stack(cumulative_factors)
        
        # Fit model
        model = LinearRegression().fit(X, returns)
        n, k = X.shape
        r2 = model.score(X, returns)
        
        # Adjusted R² (penalizes model complexity)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        
        # Incremental contribution
        decomposition[factor_name] = r2_adj - prev_r2_adj
        prev_r2_adj = r2_adj
    
    return decomposition

# Usage (order matters!)
factor_order = [
    ('Market', factors['Market']),
    ('Size', factors['Size']),
    ('Value', factors['Value']),
    ('Growth', factors['Growth']),
    ('Quality', factors['Quality']),
    ('Momentum', factors['Momentum']),
    ('ShortInterest', factors['ShortInterest'])
]

incremental_r2 = hierarchical_variance_decomposition(returns, factor_order)
```

**Pros:**
- Respects economic hierarchy (Market → Size → themes)
- Adjusted R² penalizes overfitting
- Fast

**Cons:**
- **Order-dependent:** Value's contribution changes if added before/after Growth
- Doesn't capture symmetric interactions

**When to use:** If you have a strong prior on factor importance (e.g., Market > Size > thematic factors).

---

### Recommendation: Hybrid Approach

```python
# Step 1: Hierarchical decomposition for primary factors
primary_order = [('Market', market), ('Size', size), ('Value', value)]
primary_contrib = hierarchical_variance_decomposition(returns, primary_order)

# Step 2: Partial R² for theme factors (no clear ordering)
theme_factors = {
    'Growth': growth_resid,
    'Quality': quality_resid,
    'Momentum': momentum,
    'ShortInterest': si
}
theme_contrib, theme_r2 = partial_r2_contributions(returns, theme_factors)

# Combine
all_contrib = {**primary_contrib, **theme_contrib}
```

---

### Visualization: Variance Waterfall Chart

```python
import matplotlib.pyplot as plt

def plot_variance_waterfall(contributions, total_r2):
    """
    Stacked bar chart showing each factor's contribution + residual.
    """
    factors = list(contributions.keys())
    values = list(contributions.values())
    
    # Add unexplained variance
    factors.append('Residual')
    residual = 1 - total_r2
    values.append(residual)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    cumulative = 0
    colors = ['#2ecc71' if v > 0.01 else '#e67e22' if v > 0 else '#e74c3c' for v in values]
    colors[-1] = '#95a5a6'  # gray for residual
    
    for i, (factor, value) in enumerate(zip(factors, values)):
        ax.bar(i, value, bottom=cumulative, color=colors[i], edgecolor='black', linewidth=1.5)
        
        # Label with percentage
        if value > 0.01:  # only label if > 1%
            ax.text(i, cumulative + value/2, f'{value:.1%}', 
                   ha='center', va='center', fontweight='bold', fontsize=11)
        
        cumulative += value
    
    ax.axhline(total_r2, color='red', linestyle='--', linewidth=2, label=f'Total R² = {total_r2:.1%}')
    
    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels(factors, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Variance Explained', fontsize=12)
    ax.set_title('Factor Model Variance Decomposition', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Usage
plot_variance_waterfall(normalized_contrib, r2_full)
```

---

## Validation Checklist

Before finalizing the factor set, validate:

### 1. Multicollinearity Check
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def compute_vif(factors_df):
    """Variance Inflation Factor - measures multicollinearity."""
    vif_data = pd.DataFrame()
    vif_data["Factor"] = factors_df.columns
    vif_data["VIF"] = [variance_inflation_factor(factors_df.values, i) 
                       for i in range(len(factors_df.columns))]
    return vif_data

vif = compute_vif(pd.DataFrame(factors))
print(vif)

# Rule of thumb: VIF < 5 is acceptable, VIF > 10 is problematic
```

### 2. Eigenvalue Spectrum
```python
cov_matrix = np.cov(np.column_stack(list(factors.values())), rowvar=False)
eigenvalues = np.linalg.eigvalsh(cov_matrix)
print("Eigenvalues (sorted):", sorted(eigenvalues, reverse=True))

# Red flag: one eigenvalue >> others (dimension collapse)
# Ideal: eigenvalues relatively balanced
```

### 3. IC Stability (Out-of-Sample)
```python
def rolling_ic_test(factor, returns, train_window=252, test_window=21):
    """Test if factor IC is stable out-of-sample."""
    ics_train = []
    ics_test = []
    
    for i in range(len(factor) - train_window - test_window):
        # Train period
        train_factor = factor[i:i+train_window]
        train_returns = returns[i:i+train_window]
        ic_train = train_factor.corr(train_returns)
        
        # Test period (next month)
        test_factor = factor[i+train_window:i+train_window+test_window]
        test_returns = returns[i+train_window:i+train_window+test_window]
        ic_test = test_factor.corr(test_returns)
        
        ics_train.append(ic_train)
        ics_test.append(ic_test)
    
    print(f"Mean IC (train): {np.mean(ics_train):.3f}")
    print(f"Mean IC (test): {np.mean(ics_test):.3f}")
    print(f"IC stability (corr): {np.corrcoef(ics_train, ics_test)[0,1]:.3f}")

# Run for each factor
for name, factor in factors.items():
    print(f"\n{name}:")
    rolling_ic_test(factor, returns)
```

### 4. Factor Return Correlations
```python
# Compute factor returns (long-short portfolios)
def compute_factor_returns(factor_values, asset_returns):
    """
    Return of a long-short portfolio: long top quintile, short bottom quintile.
    """
    quintiles = pd.qcut(factor_values, 5, labels=False)
    long_ret = asset_returns[quintiles == 4].mean()
    short_ret = asset_returns[quintiles == 0].mean()
    return long_ret - short_ret

# For each time period, compute all factor returns
factor_returns_ts = {}
for name, factor_ts in factors_timeseries.items():  # assume you have time series
    factor_returns_ts[name] = [
        compute_factor_returns(factor_ts[t], asset_returns[t])
        for t in range(len(factor_ts))
    ]

# Check pairwise correlations of factor RETURNS
factor_ret_df = pd.DataFrame(factor_returns_ts)
corr_matrix = factor_ret_df.corr()

print("Factor return correlations:")
print(corr_matrix)

# Red flag: |corr| > 0.7 between any two factors → consider dropping one
```

---

## Rolling Variance Decomposition (Regime Detection)

Track how factor contributions evolve over time:

```python
def rolling_variance_decomposition(returns_ts, factors_ts, window=252):
    """
    Compute variance decomposition in rolling windows.
    
    Args:
        returns_ts: pd.DataFrame with dates × assets
        factors_ts: dict of {factor_name: pd.DataFrame (dates × assets)}
        window: rolling window size in days
    
    Returns:
        pd.DataFrame with dates × factor contributions
    """
    dates = returns_ts.index[window:]
    contributions_ts = []
    
    for i, date in enumerate(dates):
        # Window data
        ret_window = returns_ts.iloc[i:i+window]
        factors_window = {k: v.iloc[i:i+window] for k, v in factors_ts.items()}
        
        # Compute contributions (cross-sectional on each date in window, then average)
        window_contribs = []
        for t in range(len(ret_window)):
            ret_t = ret_window.iloc[t]
            factors_t = {k: v.iloc[t] for k, v in factors_window.items()}
            
            partial_r2, _ = partial_r2_contributions(ret_t, factors_t)
            window_contribs.append(partial_r2)
        
        # Average contribution over window
        avg_contrib = pd.DataFrame(window_contribs).mean()
        contributions_ts.append(avg_contrib)
    
    return pd.DataFrame(contributions_ts, index=dates)

# Usage
rolling_contribs = rolling_variance_decomposition(returns_ts, factors_ts)

# Plot evolution
rolling_contribs.plot(figsize=(14, 8), title='Factor Contributions Over Time')
plt.ylabel('Variance Explained')
plt.legend(loc='upper left')
plt.show()
```

**Use case:** If Growth contribution drops from 8% to 2% during a recession, it's losing signal → potentially downweight or investigate why.

---

## Summary: Recommended Workflow

```python
# STEP 1: Build composites (within-theme aggregation)
Value = ic_weighted_avg(['sP/S', 'sP/E', 'P/S', 'P/Ee', 'P/GP'])
Growth = ic_weighted_avg(['GS', 'GE', 'HSG', 'SGD', 'GGP'])
Quality = ic_weighted_avg(['OM', 'ROI', 'ROE', 'r2_S', 'r2_E'])

# STEP 2: Targeted orthogonalization (remove Size effects)
Growth_resid = residualize(Growth, log_Size)
Quality_resid = residualize(Quality, log_Size)

# STEP 3: Standardize all factors
factors = {
    'Market': Market,
    'Size': standardize(log_Size),
    'Value': standardize(Value),
    'Growth': standardize(Growth_resid),
    'Quality': standardize(Quality_resid),
    'Momentum': standardize(Momentum),
    'ShortInterest': standardize(ShortInterest)
}

# STEP 4: Compute variance contributions
partial_r2, r2_full = partial_r2_contributions(returns, factors)
normalized_contrib = normalize_contributions(partial_r2, r2_full)

# STEP 5: Visualize
plot_variance_waterfall(normalized_contrib, r2_full)

# STEP 6: Validate
vif = compute_vif(pd.DataFrame(factors))
print("VIF check:", vif)

# STEP 7: Drop weak factors (contribution < 2% threshold)
useful_factors = {k: v for k, v in factors.items() if normalized_contrib[k] > 0.02}
```

---

## Key Takeaways

1. **Use log(Size)** - industry standard, better statistical properties
2. **Layer factors** - primary themes first, interactions optional
3. **Orthogonalize only Size effects** - preserve economic meaning elsewhere
4. **IC-weight within themes** - data-driven but still interpretable
5. **Partial R² for variance decomposition** - fast, intuitive, good enough
6. **Validate with VIF and out-of-sample IC** - catch multicollinearity and overfitting
7. **Track contributions over time** - detect regime changes early

---

## References & Further Reading

- **Fama-French (1993):** "Common risk factors in the returns on stocks and bonds"
- **Barra Risk Models:** Industry standard for factor construction
- **AQR Factor Investing:** White papers on quality, momentum, value
- **Shapley Value Regression:** Lipovetsky & Conklin (2001)
- **Variance Inflation Factor:** Standard econometrics textbooks (e.g., Greene)

---

*Document version: 1.0*  
*Last updated: [Current date]*  
*Author: Factor Model Design Discussion*

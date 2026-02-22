import numpy as np

def train_proxy_model(data_mixture: dict) -> float:
    """
    In a real run, this trains a 5-10M parameter proxy model 
    on the specific mixture and returns the validation score (e.g. HumanEval+ proxy).
    Here we simulate the environment response deterministically based on domains.
    """
    base_efficiencies = {"the-stack-v2": 0.4, "the-stack-edu": 0.85, "quality_tests": 0.9, "synthetic": 0.95}
    score = 0.0
    for domain, weight in data_mixture.items():
        eff = base_efficiencies.get(domain, 0.5)
        # Non-linear diminishing returns for over-saturating a single domain
        score += eff * (weight - 0.5 * weight**2)
    return float(score)

def optimize_mixture(domains: list[str], n_trials: int = 50) -> dict[str, float]:
    """
    Fits a linear regression model over proxy training trials to predict and identify
    the optimal data mixture via argmax traversal.
    """
    print(f"Running RegMix Optimization over {n_trials} proxy trials...")
    
    X = []
    y = []
    for _ in range(n_trials):
        # Sample random Dirichlet distributions for valid mixture probabilities
        weights = np.random.dirichlet(np.ones(len(domains)))
        mixture = dict(zip(domains, weights))
        
        # Train Proxy (simulate)
        score = train_proxy_model(mixture)
        
        X.append(weights)
        y.append(score)
        
    X_mat = np.array(X)
    y_vec = np.array(y)
    
    # Solve linear regression: y = X * beta
    # beta represents the marginal utility contribution of each domain
    beta, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
    
    # The 'optimal' mixture mathematically favors domains with the highest beta coefficient
    # To prevent pure 1.0 distributions, we apply a softmax over the bounded betas
    exp_beta = np.exp(beta * 5.0) # Temperature scaling
    optimal_weights = exp_beta / np.sum(exp_beta)
    
    optimum = {domain: round(float(w), 4) for domain, w in zip(domains, optimal_weights)}
    print(f"Mathematically optimized mixture derived from proxy regression: {optimum}")
    return optimum

if __name__ == "__main__":
    stage1_opt = optimize_mixture(["the-stack-v2", "the-stack-edu"], n_trials=20)
    stage2_opt = optimize_mixture(["the-stack-edu-filtered", "quality_tests"], n_trials=30)
    stage3_opt = optimize_mixture(["synthetic_phi1_style"], n_trials=5)

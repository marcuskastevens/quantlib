import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -------------------------------------------- Alpha Model Utility Functions -------------------------------------------- 

def rank(signal: pd.Series):
    """
    Cross-sectionally ranks alpha signal.
    
    Parameters:
        signal (pd.Series or np.ndarray): Series of alpha signals.
        
    Returns:
        pd.Series or np.ndarray: Ranked signal.
    """

    # If pd.Series, maintain indices/names
    if type(signal) == pd.Series:
        ranked_signal = signal.rank()        
    else:
        ranked_signal = np.argsort(signal)

    return ranked_signal

def quantiles(signal: pd.Series, q = .1):
    """
    Isolates values in the bottom and top quantiles of a signal.
    
    Parameters:
        signal (pd.Series): Series of alpha signals.
        q (float, optional): Quantile threshold (default: 0.1).
        
    Returns:
        pd.Series: Signal with values outside the quantiles set to NaN.
    """

    lower_quantile = np.quantile(signal, q=q)
    upper_quantile = np.quantile(signal, q=1-q)
    
    quantile_signal = signal.where((signal < lower_quantile) | (signal > upper_quantile)).dropna()

    return quantile_signal

def z_score(signal: pd.Series):
    """
    Calculates the z-score of a signal.
    
    Parameters:
        signal (pd.Series): Series of the signal to calculate Z-scores for.
        
    Returns:
        pd.Series: Z-scores of the input signal.
    """
    
    z_score = (signal - np.mean(signal)) / np.std(signal)

    return z_score

def ranked_signal_z_score(signal: pd.Series):
    """
    Calculates z-scores for a ranked signal and ensures dollar-neutrality.
    
    Parameters:
        signal (pd.Series): Series of the signal to calculate Z-scores for.
        
    Returns:
        pd.Series: Z-scores of the ranked signal.
        
    Raises:
        AssertionError: If the sum of Z-scores is not close to zero (dollar-neutrality check).
    """

    # Rank signals
    ranked_signal = rank(signal)

    # Standardize ranked signals
    z_score_signal = z_score(ranked_signal)

    # Check for dollar-neutrality
    assert -1e-10 < z_score_signal.sum() < 1e-10

    # Dollar-neutral alpha model views
    return z_score_signal

def intra_industry_rank(signal: pd.Series, industry: pd.Series):
    """
    Computes intra-industry cross-sectional ranks for a signal.
    
    Parameters:
        signal (pd.Series): Series of the signal to calculate intra-industry ranks for.
        industry (pd.Series): Series indicating the industry of each observation.
        
    Returns:
        pd.Series: Intra-industry ranks of the signal.
    """
    # Create intra-industry ranked signals
    ranked_signal = signal.groupby(industry).rank()

    return ranked_signal

def intra_industry_mean(signal: pd.Series, industry: pd.Series):
    """
    Calculates intra-industry mean values for a signal.
    
    Parameters:
        signal (pd.Series): Series of the signal to calculate intra-industry means for.
        industry (pd.Series): Series indicating the industry of each observation.
        
    Returns:
        pd.Series: Intra-industry mean values of the signal.
    """
    # Calculate intra-industry mean 
    mean = signal.groupby(industry).transform('mean')

    return mean

def intra_industry_std(signal: pd.Series, industry: pd.Series):
    """
    Calculates intra-industry standard deviations for a signal.
    
    Parameters:
        signal (pd.Series): Series of the signal to calculate intra-industry standard deviations for.
        industry (pd.Series): Series indicating the industry of each observation.
        
    Returns:
        pd.Series: Intra-industry standard deviations of the signal.
    """
    # Calculate intra-industry standard deviation
    sigma = signal.groupby(industry).transform('std')

    return sigma

def intra_industry_quantiles(signal: pd.Series, industry: pd.Series, q=0.1):
    """
    Isolates values in the bottom and top quantiles of a signal within each industry.
    
    Parameters:
        signal (pd.Series): Series of the signal to isolate quantiles for.
        industry (pd.Series): Series indicating the industry of each observation.
        q (float, optional): Quantile threshold (default: 0.1).
        
    Returns:
        pd.Series: Intra-industry signal with values outside the quantiles set to NaN.
    """
    # Group data by industry
    industry_grouped_signal = signal.groupby(industry)
    
    # Calculate quartile values for each industry
    lower_quantiles = industry_grouped_signal.transform(lambda x: np.quantile(x, q))
    upper_quantiles = industry_grouped_signal.transform(lambda x: np.quantile(x, (1 - q)))
    
    # Isolate values in the bottom and top quantiles within each industry
    quantile_signal = signal.where((signal <= lower_quantiles) | (signal >= upper_quantiles)).dropna()

    return quantile_signal


def intra_industry_z_score(signal: pd.Series, industry: pd.Series):
    """
    Calculates industry-neutral Z-scores for a raw signal and ensures dollar-neutrality.
    
    Parameters:
        signal (pd.Series): Series of the signal to calculate Z-scores for.
        industry (pd.Series): Series indicating the industry of each stock.
        
    Returns:
        pd.Series: Industry-neutral Z-scores of the signal.
        
    Raises:
        AssertionError: If the sum of Z-scores within each industry is not close to zero (dollar-neutrality check).
    """

    # Calculate intra-industry ranked signal mean and standard deviation
    industry_mean = intra_industry_mean(signal=signal, industry=industry)
    industry_sigma = intra_industry_std(signal=signal, industry=industry)

    # Standardize intra-industry ranked signals
    z_score = (signal - industry_mean) / industry_sigma

    # Check for dollar-neutrality
    assert -1e-10 < z_score.groupby(industry).sum().sum() < 1e-10

    # Industry dollar-neutral alpha model views
    return z_score 

def intra_industry_ranked_signal_z_score(signal: pd.Series, industry: pd.Series):
    """
    Calculates industry-neutral Z-scores for a ranked signal and ensures dollar-neutrality.
    
    Parameters:
        signal (pd.Series): Series of the signal to calculate Z-scores for.
        industry (pd.Series): Series indicating the industry of each stock.
        
    Returns:
        pd.Series: Industry-neutral Z-scores of the ranked signal.
        
    Raises:
        AssertionError: If the sum of Z-scores within each industry is not close to zero (dollar-neutrality check).
    """

    # Create intra-industry ranked signals
    ranked_signal = intra_industry_rank(signal=signal, industry=industry) 

    # Get the intra-industry z-score of ranked signals
    z_score = intra_industry_z_score(signal=ranked_signal, industry=industry)

    # Check for dollar-neutrality
    assert -1e-10 < z_score.groupby(industry).sum().sum() < 1e-10

    # Industry dollar-neutral alpha model views
    return z_score


def beta_neutralize(views: pd.Series, betas: pd.Series) -> pd.Series:
    """
    Orthogonalizes alpha model to the market factor. Transforms alpha 
    model views to beta-neutralized weights based on stock-level betas.   
    These alpha weights can then be scaled to target desired level of vol.

    This function will eventually be extended to orthogonalize alpha model 
    to other market factors (e.g., Fama-French factors).
    
    Parameters:
        views (pd.Series): Series of your alpha model views.
        betas (pd.Series): Series of ex-ante stock-level betas.
        
    Returns:
        pd.Series: Alpha weights for beta-neutralized views.
    """

    # Preprocess data
    X = sm.add_constant(betas).astype(float)
    y = views.astype(float)

    # Regress model views on stock-level betas
    model = sm.OLS(endog=y, exog=X).fit() 
    alpha = model.params[0]
    beta = model.params[1]
    resid = y - (beta * X['ExAnte_Beta'] + alpha)

    # Orthogonalize views to create your alpha weights
    alpha_views = resid
    w = alpha_views / np.sum(np.abs(alpha_views))

    # Ensure dollar neturality
    assert  -1e-10 < np.sum(w) < 1e-10

    return w

def industry_beta_neutralize(views: pd.Series, industry: pd.Series, betas: pd.Series):
    """
    Creates industry beta-neutral alpha weights for views based on stock-level betas and industries.
    Views are expected to be industry dollar-neutral before passed to this function... if views are not
    dollar neutral beforehand, there is no guaruntee that weights will be industry-dollar neutral.
    
    Parameters:
        views (pd.Series): Series of your alpha model views.
        industry (pd.Series): Series indicating the industry of each stock.
        betas (pd.Series): Series of ex-ante stock-level betas.
        
    Returns:
        pd.Series: Alpha weights for industry beta-neutralized views.
    """

    # Create industry dummy variables
    industry_dummies = pd.get_dummies(industry, prefix='Industry')

    print(industry_dummies.shape, betas.shape)
    
    # Create interaction terms: ExAnte_Beta * Industry_Dummies
    interaction_terms = pd.DataFrame(index=industry_dummies.index, columns=industry_dummies.columns)
    for i, beta in betas.items():
        interaction_terms.loc[i] = industry_dummies.loc[i] * beta

    # Beta Neutralize beta within industry
    w = beta_neutralize(views=views, betas=interaction_terms)

    return w

def plot_weights(w: pd.Series, title=None):

    plt.figure(figsize=(20, 5))
    plt.bar(w.index, w)
    if title:
        plt.title(title)
    plt.show()
from typing import Union
import numpy as np
import scipy.stats as st


def chi2_log_sf_asymptotic_approximation(x: np.array, degrees_freedom: Union[int, float]):
    """
    Asymptotic approximation of the survival function of a Chi-Squared distribution, using a Normal distribution
    @param x: observed samples
    @param df: degrees of freedom of the Chi-Squared distribution
    @return: log P(X>=x). Returns the log probability of sampling values greater or equal than the observed ones
    """
    mu = degrees_freedom
    sigma = np.sqrt(2 * degrees_freedom)
    log_prob = st.norm.logsf(x, mu, sigma) / np.log(10)
    return log_prob


def chi2_log_sf_peizer_pratt_approximation(x: np.array, degrees_freedom: Union[int, float]):
    df = degrees_freedom
    y = (x - df + 2 / 3 - 0.08 / df) / np.abs(x - (df - 1)) * np.sqrt((df - 1) * np.log((df - 1) / x) + x - (df - 1))
    return st.norm.logsf(y) / np.log(10)


def chi2_log_sf_canal_approximation(x: np.array, degrees_freedom: Union[int, float]):
    df = degrees_freedom
    if df <= 1:
        return st.chi2.logsf(x, df) / np.log(10)
    mu = 5 / 6 - 1 / 9 / df - 7 / 648 / df ** 2 + 25 / 2187 / df ** 3
    sigma = np.sqrt(1 / 18 / df + 1 / 162 / df ** 2 - 37 / 11664 / df ** 3)
    ln = np.log(x / df)
    y = np.exp(1 / 6 * ln) - 1 / 2 * np.exp(1 / 3 * ln) + 1 / 3 * np.exp(1 / 2 * ln)
    return st.norm.logsf(y, mu, sigma) / np.log(10)

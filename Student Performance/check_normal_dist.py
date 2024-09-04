import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kstest, normaltest, anderson
import scipy.stats as stats

def plot_histogram_qq(data, col):

    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1)
    sns.histplot(data[col], kde=True)
    plt.title("Histogram of {}".format(col))

    plt.subplot(1, 2, 2)
    stats.probplot(data[col], dist="norm", plot=plt)
    plt.title("Q-Q plot of {}".format(col))

    plt.tight_layout()
    #plt.show()

def check_normality(data, col):
    print(f"Normality tests for {col}:")
    
    # Shapiro-Wilk Test
    shapiro_stat, shapiro_p = shapiro(data[col])
    print(f"Shapiro-Wilk Test: Statistic={shapiro_stat}, p-value={shapiro_p}")
    
    # Kolmogorov-Smirnov Test
    ks_stat, ks_p = kstest(data[col], 'norm', args=(np.mean(data[col]), np.std(data[col])))
    print(f"Kolmogorov-Smirnov Test: Statistic={ks_stat}, p-value={ks_p}")
    
    # D'Agostino's K-squared Test
    dagostino_stat, dagostino_p = normaltest(data[col])
    print(f"D'Agostino's K-squared Test: Statistic={dagostino_stat}, p-value={dagostino_p}")
    
    # Anderson-Darling Test
    anderson_stat = anderson(data[col])
    print(f"Anderson-Darling Test: Statistic={anderson_stat.statistic}, Critical Values={anderson_stat.critical_values}, Significance Levels={anderson_stat.significance_level}\n")


# Skewness and Kurtosis
def check_skewness_kurtosis(data, col):
    skewness = data[col].skew()
    kurtosis = data[col].kurtosis()
    print(f"Skewness and Kurtosis for {col}:")
    print(f"Skewness: {skewness}")
    print(f"Kurtosis: {kurtosis}\n")
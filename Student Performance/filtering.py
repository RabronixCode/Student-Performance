import numpy as np
import matplotlib.pyplot as plt1
import seaborn as sns1
import scipy.stats as stats

def IQR(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5*IQR
    upper_bound = Q3 + 1.5*IQR

    # Original Data
    #plt1.figure(figsize=(12, 6))
    #sns1.boxplot(data=data[col])
    #plt1.title('Original Data')
    #plt.show()

    # IQR Filtered Data
    #plt1.figure(figsize=(12, 6))
    #sns.boxplot(data=data_filtered[col])
    #plt1.title('IQR Filtered Data')
    #plt.show()

    return data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]


def z_score_filter(data, col, threshold=3):
    z_scores = np.abs(stats.zscore(data[col]))
    return data[z_scores < threshold]


def percentile_filter(data, col, lower=1, upper=99):
    lower_bound = np.percentile(data[col], lower)
    upper_bound = np.percentile(data[col], upper)
    return data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

    

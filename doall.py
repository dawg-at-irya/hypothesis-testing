import numpy as np
from scipy.stats import ttest_ind, ks_2samp, anderson_ksamp
import warnings
# ignore UserWarning from scipy.stats.anderson_ksamp
warnings.filterwarnings("ignore", category=UserWarning)

data1 = np.loadtxt('data1.dat')
data2 = np.loadtxt('data2.dat')

# First, perform 2-sample t test for independent samples
mesg = "Null hypothesis: the means of the two samples are equal."
print("Performing 2-sample t test for unequal variances (Welch's t-test)")
print(mesg)
t_statistic, p_value = ttest_ind(data1, data2, equal_var=False)
print(f"The estimated p-value is {p_value:.6f}.")
if p_value > 0.05:
    print("The null hypothesis cannot be rejected.")
else:
    print("The null hypothesis is rejected!")
print("")

# Next, perform 2-sample Kolmogorov-Smirnov test
mesg = "Null hypothesis: both samples are drawn from the same distribution."
print("Performing 2-sample Kolmogorov-Smirnov test")
print(mesg)
ks_statistic, p_value = ks_2samp(data1, data2)
print(f"The estimated p-value is {p_value:.6f}.")
if p_value > 0.05:
    print("The null hypothesis cannot be rejected.")
else:
    print("The null hypothesis is rejected!")
print("")

print("Performing 2-sample Anderson-Darling test")
result = anderson_ksamp([data1, data2])
print(f"The estimated AD statistic is {result.statistic:.3f}.")
print(f"Critical value for 5%: {result.critical_values[2]:.3f}")
if result.statistic < result.critical_values[2]:
    print("The null hypothesis cannot be rejected.")
else:
    print("The null hypothesis is rejected!")

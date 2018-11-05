import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.stats import uniform
import matplotlib.pyplot as plt
from  scipy.stats import ttest_1samp



num_simulations = 1000
num_samples = 10000   
# parameters of the probability distributions
simulation_samples = []
mu_1, sigma_1 = 0, 1
a, b = -5, 5


for simulation in range(num_simulations):

	X = np.random.uniform(a, b, num_samples)

	norm_pdf = norm(mu_1, sigma_1)
	uniform_pdf = uniform(a, b-a)

	num = norm_pdf.pdf(X)
	denum = uniform_pdf.pdf(X)
	weights = num / denum

	f_of_X = pow(X,2)

	simulation_samples.append((weights.dot(f_of_X))/num_samples)

simulation_samples = np.array(simulation_samples)

print (np.mean(simulation_samples))
print (np.std(simulation_samples))
print(ttest_1samp(simulation_samples, popmean = 1))
#We cant rejec the null hypothesis. Therefore the mean is 1. 



sns.distplot(simulation_samples)
plt.title('Distribution of importance sampling')

plt.show()
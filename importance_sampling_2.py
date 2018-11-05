import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.stats import uniform
import math
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous


num_simulations = 1000
num_samples = 1000  
a,b = -1, 1
simulation_samples = []

def function_to_estimate_pdf(x):
		return (1+ np.cos(np.pi*X))/2


for simulation in range(num_simulations):

	#Draw a random sample
	X = np.random.uniform(a, b, num_samples)

	#Initiate pdf 
	uniform_pdf = uniform(a, b-a)

	#Make weight
	numerator = function_to_estimate_pdf(X)
	denominator = uniform_pdf.pdf(X)

	weights = numerator / denominator

	f_of_X = pow(X,2)

	simulation_samples.append((weights.dot(f_of_X))/num_samples)

simulation_samples = np.array(simulation_samples)
print (np.mean(simulation_samples))


sns.distplot(simulation_samples)
plt.title('Distribution of importance sampling')

plt.show()
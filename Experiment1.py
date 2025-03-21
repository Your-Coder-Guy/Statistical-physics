import numpy as np
import matplotlib.pyplot as plt

# For Normal distribution--------------------------------------------
mu = 0
sigma = 1
sample_size = 1000

# Generate normal distributions
normal_data = np.random.normal(mu, sigma, sample_size)

# Plot the normal distribution
plt.hist(normal_data, bins='auto', color='turquoise', edgecolor='black', alpha=0.7)
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.5)
plt.show()


# For Binomial distribution--------------------------------------------

#Small sample size
n_binomial_small = 10  
p = 0.5          
sample_size_small = 1000  

# Large sample size
n_binomial_large = 50  
p = 0.5          
sample_size_large = 10000  

# Generate binomial distributions
binomial_data_small = np.random.binomial(n_binomial_small, p, sample_size_small)
binomial_data_large = np.random.binomial(n_binomial_large, p, sample_size_large)

# Plot both distributions in subplots
plt.figure(figsize=(12, 6))


# For small sample size
plt.subplot(1, 2, 1)
plt.hist(binomial_data_small, bins='auto', color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Binomial Distribution (Small Sample Size)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.5)

# For large sample size
plt.subplot(1, 2, 2)
plt.hist(binomial_data_large, bins='auto', color='lightcoral', edgecolor='black', alpha=0.7)
plt.title('Binomial Distribution (Large Sample Size)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.5)

# Showing the plot
plt.tight_layout()
plt.show()

# For Poisson distribution--------------------------------------------

lamda = 20  # rate of events

# Small sample size
sample_size_small = 100

# Large sample size
sample_size_large = 10000

# Generate Poisson distributions
poisson_data_small = np.random.poisson(lamda, sample_size_small)
poisson_data_large = np.random.poisson(lamda, sample_size_large)

# Plot both distributions in subplots
plt.figure(figsize=(12, 6))

# For small sample size
plt.subplot(1, 2, 1)
plt.hist(poisson_data_small, bins='auto', color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Poisson Distribution (Small Sample Size)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.5)

# For large sample size
plt.subplot(1, 2, 2)
plt.hist(poisson_data_large, bins='auto', color='lightcoral', edgecolor='black', alpha=0.7)
plt.title('Poisson Distribution (Large Sample Size)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.5)

# Showing the plot
plt.tight_layout()
plt.show()

# For Cauchy distribution--------------------------------------------

small_x = np.linspace(-10, 10, 1000)
large_x = np.linspace(-20, 20, 10000)

cauchy_pdf_small = 1 / (np.pi * (1 + small_x**2))
cauchy_pdf_large = 1 / (np.pi * (1 + large_x**2))

# Plot both distributions in subplots
plt.figure(figsize=(12, 6))

# For small sample size
plt.subplot(1, 2, 1)
plt.plot(small_x, cauchy_pdf_small, color='skyblue', alpha=0.7)
plt.fill_between(small_x, cauchy_pdf_small, color='skyblue', alpha=0.7)
plt.title('Cauchy Distribution (Small Sample Size)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True, alpha=0.5)

# For large sample size
plt.subplot(1, 2, 2)
plt.plot(large_x, cauchy_pdf_large, color='lightcoral', alpha=0.7)
plt.fill_between(large_x, cauchy_pdf_large, color='lightcoral', alpha=0.7)
plt.title('Cauchy Distribution (Large Sample Size)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True, alpha=0.5)


# Showing the plot
plt.tight_layout()
plt.show()

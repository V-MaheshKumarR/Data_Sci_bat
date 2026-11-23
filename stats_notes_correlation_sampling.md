# Statistics Notes: Correlation, Random Sampling & Hypothesis Testing

## Table of Contents
1. [Correlation Types](#correlation-types)
2. [Random Sampling](#random-sampling)
---

## 1. Correlation Types

Correlation measures the strength and direction of the relationship between two variables. The correlation coefficient ranges from -1 to +1.

### 1.1 Pearson Correlation Coefficient

**Definition**: Measures the linear relationship between two continuous variables.

**Formula**: 
```
r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
```

**Assumptions**:
- Both variables are continuous
- Linear relationship
- No outliers
- Normal distribution (for significance testing)

**Interpretation**:
- r = 1: Perfect positive correlation
- r = 0: No correlation
- r = -1: Perfect negative correlation
- |r| > 0.7: Strong correlation
- 0.3 < |r| < 0.7: Moderate correlation
- |r| < 0.3: Weak correlation

**Python Example**:
```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
x = np.random.normal(50, 10, 100)
y = 2 * x + np.random.normal(0, 10, 100)  # Linear relationship with noise

# Calculate Pearson correlation
pearson_corr, p_value = stats.pearsonr(x, y)
print(f"Pearson Correlation: {pearson_corr:.3f}")
print(f"P-value: {p_value:.4f}")

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5)
plt.xlabel('X Variable')
plt.ylabel('Y Variable')
plt.title(f'Pearson Correlation: {pearson_corr:.3f}')
plt.grid(True, alpha=0.3)
plt.show()
```

### 1.2 Spearman Rank Correlation

**Definition**: Measures the monotonic relationship between two variables using ranks instead of raw values.

**When to Use**:
- Non-linear monotonic relationships
- Ordinal data
- Data with outliers
- Non-normal distributions

**Python Example**:
```python
# Generate non-linear monotonic data
x = np.linspace(0, 10, 100)
y = x ** 2 + np.random.normal(0, 5, 100)  # Quadratic relationship

# Calculate Spearman correlation
spearman_corr, p_value = stats.spearmanr(x, y)
print(f"Spearman Correlation: {spearman_corr:.3f}")
print(f"P-value: {p_value:.4f}")

# Compare with Pearson
pearson_corr, _ = stats.pearsonr(x, y)
print(f"Pearson Correlation: {pearson_corr:.3f}")
print(f"\nSpearman detects monotonic relationship better for non-linear data")
```

### 1.3 Kendall Tau Correlation

**Definition**: Measures the ordinal association between two variables based on concordant and discordant pairs.

**When to Use**:
- Small sample sizes
- Many tied ranks
- More robust to outliers than Spearman

**Python Example**:
```python
# Sample data with ties
x = [1, 2, 3, 4, 5, 5, 6, 7]
y = [2, 4, 6, 8, 10, 10, 12, 14]

# Calculate Kendall Tau
kendall_corr, p_value = stats.kendalltau(x, y)
print(f"Kendall Tau: {kendall_corr:.3f}")
print(f"P-value: {p_value:.4f}")
```

### 1.4 Point-Biserial Correlation

**Definition**: Measures the relationship between a continuous variable and a binary variable.

**Python Example**:
```python
# Sample data: exam scores (continuous) vs pass/fail (binary)
scores = np.array([85, 90, 78, 92, 88, 76, 95, 82, 79, 91])
passed = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1])  # 1=pass, 0=fail

# Calculate point-biserial correlation
pb_corr, p_value = stats.pointbiserialr(passed, scores)
print(f"Point-Biserial Correlation: {pb_corr:.3f}")
print(f"P-value: {p_value:.4f}")
```

### 1.5 Correlation Matrix

**Python Example**:
```python
import seaborn as sns

# Create sample dataset
np.random.seed(42)
data = pd.DataFrame({
    'Height': np.random.normal(170, 10, 100),
    'Weight': np.random.normal(70, 15, 100),
    'Age': np.random.randint(20, 60, 100),
    'Income': np.random.normal(50000, 15000, 100)
})

# Add correlation
data['Weight'] = data['Height'] * 0.8 + np.random.normal(0, 5, 100)

# Calculate correlation matrix
corr_matrix = data.corr()
print(corr_matrix)

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix Heatmap')
plt.show()
```

---

## 2. Random Sampling

Random sampling is the process of selecting a subset of individuals from a population to estimate characteristics of the whole population.

### 2.1 Simple Random Sampling

**Definition**: Every member of the population has an equal chance of being selected.

**Python Example**:
```python
# Create a population
population = pd.DataFrame({
    'ID': range(1, 1001),
    'Age': np.random.randint(18, 80, 1000),
    'Income': np.random.normal(50000, 20000, 1000)
})

# Simple random sampling - 10% sample
sample_size = 100
simple_sample = population.sample(n=sample_size, random_state=42)

print(f"Population Mean Income: ${population['Income'].mean():.2f}")
print(f"Sample Mean Income: ${simple_sample['Income'].mean():.2f}")
print(f"Difference: ${abs(population['Income'].mean() - simple_sample['Income'].mean()):.2f}")
```

### 2.2 Stratified Sampling

**Definition**: Population is divided into strata (groups), and random samples are taken from each stratum.

**Python Example**:
```python
# Add a categorical variable (stratum)
population['Region'] = np.random.choice(['North', 'South', 'East', 'West'], 
                                       size=1000, p=[0.3, 0.3, 0.2, 0.2])

# Stratified sampling - proportional allocation
stratified_sample = population.groupby('Region', group_keys=False).apply(
    lambda x: x.sample(frac=0.1, random_state=42)
)

print("\nPopulation Distribution:")
print(population['Region'].value_counts(normalize=True))
print("\nStratified Sample Distribution:")
print(stratified_sample['Region'].value_counts(normalize=True))
```

### 2.3 Systematic Sampling

**Definition**: Select every kth element from the population after a random start.

**Python Example**:
```python
# Systematic sampling
k = 10  # Every 10th element
start = np.random.randint(0, k)  # Random start
systematic_sample = population.iloc[start::k]

print(f"Systematic Sample Size: {len(systematic_sample)}")
print(f"Sample Mean Age: {systematic_sample['Age'].mean():.2f}")
```

### 2.4 Cluster Sampling

**Definition**: Population is divided into clusters, and entire clusters are randomly selected.

**Python Example**:
```python
# Create clusters
population['Cluster'] = np.random.randint(1, 21, 1000)  # 20 clusters

# Random cluster sampling - select 5 clusters
selected_clusters = np.random.choice(population['Cluster'].unique(), 
                                    size=5, replace=False)
cluster_sample = population[population['Cluster'].isin(selected_clusters)]

print(f"Cluster Sample Size: {len(cluster_sample)}")
print(f"Selected Clusters: {selected_clusters}")
```

### 2.5 Bootstrap Sampling

**Definition**: Resampling with replacement from the sample to estimate the sampling distribution.

**Python Example**:
```python
# Original sample
original_sample = population.sample(n=100, random_state=42)

# Bootstrap resampling
n_bootstrap = 1000
bootstrap_means = []

for i in range(n_bootstrap):
    bootstrap_sample = original_sample.sample(n=len(original_sample), 
                                             replace=True)
    bootstrap_means.append(bootstrap_sample['Income'].mean())

# Calculate confidence interval
ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)

print(f"Bootstrap Mean: ${np.mean(bootstrap_means):.2f}")
print(f"95% Confidence Interval: [${ci_lower:.2f}, ${ci_upper:.2f}]")

# Visualize bootstrap distribution
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_means, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(bootstrap_means), color='red', linestyle='--', 
            label='Bootstrap Mean')
plt.axvline(ci_lower, color='green', linestyle='--', label='95% CI')
plt.axvline(ci_upper, color='green', linestyle='--')
plt.xlabel('Mean Income')
plt.ylabel('Frequency')
plt.title('Bootstrap Distribution of Sample Mean')
plt.legend()
plt.show()
```


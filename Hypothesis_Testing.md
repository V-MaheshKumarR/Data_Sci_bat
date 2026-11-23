## 3. Hypothesis Testing

Hypothesis testing is a statistical method for making decisions about population parameters based on sample data.

### 3.1 General Framework

**Steps**:
1. State null hypothesis (H₀) and alternative hypothesis (H₁)
2. Choose significance level (α, typically 0.05)
3. Calculate test statistic
4. Determine p-value
5. Make decision: reject H₀ if p-value < α

**Types of Errors**:
- Type I Error (α): Rejecting true H₀ (false positive)
- Type II Error (β): Failing to reject false H₀ (false negative)
- Power = 1 - β

### 3.2 One-Sample T-Test

**Purpose**: Test if the mean of a sample differs from a known value.

**H₀**: μ = μ₀  
**H₁**: μ ≠ μ₀ (two-tailed) or μ > μ₀ or μ < μ₀ (one-tailed)

**Python Example**:
```python
from scipy.stats import ttest_1samp

# Sample data: student test scores
scores = np.array([78, 85, 82, 90, 88, 76, 95, 84, 79, 92, 87, 83])

# Test if mean score is different from 80
null_mean = 80
t_statistic, p_value = ttest_1samp(scores, null_mean)

print(f"Sample Mean: {scores.mean():.2f}")
print(f"T-statistic: {t_statistic:.3f}")
print(f"P-value (two-tailed): {p_value:.4f}")

if p_value < 0.05:
    print("Reject H₀: Mean is significantly different from 80")
else:
    print("Fail to reject H₀: No significant difference from 80")
```

### 3.3 Two-Sample T-Test (Independent)

**Purpose**: Test if means of two independent groups differ.

**H₀**: μ₁ = μ₂  
**H₁**: μ₁ ≠ μ₂

**Python Example**:
```python
from scipy.stats import ttest_ind

# Sample data: exam scores for two different teaching methods
method_a = np.array([85, 88, 90, 82, 87, 89, 91, 86, 84, 88])
method_b = np.array([78, 81, 79, 84, 80, 82, 77, 83, 81, 80])

# Perform independent t-test
t_statistic, p_value = ttest_ind(method_a, method_b)

print(f"Method A Mean: {method_a.mean():.2f}")
print(f"Method B Mean: {method_b.mean():.2f}")
print(f"T-statistic: {t_statistic:.3f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject H₀: Significant difference between teaching methods")
else:
    print("Fail to reject H₀: No significant difference")
```

### 3.4 Paired T-Test

**Purpose**: Test if means differ for paired/dependent samples (before/after, matched pairs).

**Python Example**:
```python
from scipy.stats import ttest_rel

# Sample data: blood pressure before and after treatment
before = np.array([145, 150, 148, 152, 147, 149, 151, 146, 153, 148])
after = np.array([138, 142, 140, 145, 141, 143, 144, 139, 146, 142])

# Perform paired t-test
t_statistic, p_value = ttest_rel(before, after)

print(f"Mean Before: {before.mean():.2f}")
print(f"Mean After: {after.mean():.2f}")
print(f"Mean Difference: {(before - after).mean():.2f}")
print(f"T-statistic: {t_statistic:.3f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject H₀: Treatment has significant effect")
else:
    print("Fail to reject H₀: No significant treatment effect")
```

### 3.5 Chi-Square Test of Independence

**Purpose**: Test if two categorical variables are independent.

**Python Example**:
```python
from scipy.stats import chi2_contingency

# Contingency table: Gender vs Product Preference
data = pd.DataFrame({
    'Gender': ['M', 'M', 'M', 'F', 'F', 'F'] * 50,
    'Product': ['A', 'B', 'C', 'A', 'B', 'C'] * 50
})

# Create contingency table
contingency_table = pd.crosstab(data['Gender'], data['Product'])
print("Contingency Table:")
print(contingency_table)

# Perform chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"\nChi-square statistic: {chi2:.3f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")

if p_value < 0.05:
    print("Reject H₀: Variables are dependent")
else:
    print("Fail to reject H₀: Variables are independent")
```

### 3.6 ANOVA (Analysis of Variance)

**Purpose**: Test if means of three or more groups differ.

**Python Example**:
```python
from scipy.stats import f_oneway

# Sample data: test scores for three different study methods
group1 = np.array([85, 87, 86, 88, 84, 89, 87, 86])
group2 = np.array([78, 80, 79, 81, 77, 82, 80, 79])
group3 = np.array([92, 94, 93, 91, 95, 93, 92, 94])

# Perform one-way ANOVA
f_statistic, p_value = f_oneway(group1, group2, group3)

print(f"Group 1 Mean: {group1.mean():.2f}")
print(f"Group 2 Mean: {group2.mean():.2f}")
print(f"Group 3 Mean: {group3.mean():.2f}")
print(f"\nF-statistic: {f_statistic:.3f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject H₀: At least one group mean is different")
else:
    print("Fail to reject H₀: No significant difference between groups")
```


### 3.7 Shapiro-Wilk Test (Normality Test)

**Purpose**: Test if data follows a normal distribution.

**Python Example**:
```python
from scipy.stats import shapiro

# Generate data
normal_data = np.random.normal(100, 15, 100)
non_normal_data = np.random.exponential(2, 100)

# Test normality
stat1, p1 = shapiro(normal_data)
stat2, p2 = shapiro(non_normal_data)

print("Normal Data:")
print(f"Statistic: {stat1:.4f}, P-value: {p1:.4f}")
if p1 > 0.05:
    print("Data appears normally distributed\n")

print("Non-normal Data:")
print(f"Statistic: {stat2:.4f}, P-value: {p2:.4f}")
if p2 < 0.05:
    print("Data does not appear normally distributed")
```


---

## Summary

### When to Use Which Test:

| Scenario | Test |
|----------|------|
| Compare sample mean to known value | One-sample t-test |
| Compare means of 2 independent groups | Independent t-test |
| Compare means of 2 paired groups | Paired t-test |
| Compare means of 3+ groups | ANOVA |
| Test independence of categorical variables | Chi-square |
| Test correlation between continuous variables | Pearson/Spearman |
| Test normality | Shapiro-Wilk |

### Key Python Libraries:
- `scipy.stats`: Statistical tests and distributions
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `matplotlib/seaborn`: Visualization
- `statsmodels`: Advanced statistical modeling
# Comprehensive Guide to Exploratory Data Analysis (EDA)

## Table of Contents
1. [Introduction to EDA](#1-introduction-to-eda)
2. [Data Collection and Loading](#2-data-collection-and-loading)
3. [Initial Data Inspection](#3-initial-data-inspection)
4. [Data Cleaning](#4-data-cleaning)
5. [Univariate Analysis](#5-univariate-analysis)
6. [Bivariate Analysis](#6-bivariate-analysis)
7. [Multivariate Analysis](#7-multivariate-analysis)
8. [Feature Engineering](#8-feature-engineering)
9. [Outlier Detection](#9-outlier-detection)
10. [Data Transformation](#10-data-transformation)
11. [Final EDA Report](#11-final-eda-report)

---

## 1. Introduction to EDA

**What is EDA?**
Exploratory Data Analysis is the critical process of performing initial investigations on data to discover patterns, spot anomalies, test hypotheses, and check assumptions through summary statistics and graphical representations.

**Goals of EDA**:
- Understand data structure and characteristics
- Identify data quality issues
- Discover patterns and relationships
- Detect outliers and anomalies
- Formulate hypotheses for further analysis
- Guide feature engineering and modeling decisions

**EDA Workflow**:
```
Data Collection → Initial Inspection → Cleaning → Univariate Analysis → 
Bivariate Analysis → Multivariate Analysis → Feature Engineering → 
Transformation → Insights & Reporting
```

---

## 2. Data Collection and Loading

### 2.1 Setting Up Environment

```python
# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

print("Environment setup complete!")
```

### 2.2 Loading Data from Different Sources

```python
# Load from CSV
df_csv = pd.read_csv('data.csv')

# Load from Excel
df_excel = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Load from JSON
df_json = pd.read_json('data.json')

# Load from SQL database
import sqlite3
conn = sqlite3.connect('database.db')
df_sql = pd.read_sql_query("SELECT * FROM table_name", conn)

# Load from URL
url = 'https://example.com/data.csv'
df_url = pd.read_csv(url)

# Load with specific parameters
df = pd.read_csv('data.csv', 
                 sep=',',           # Delimiter
                 encoding='utf-8',  # Encoding
                 na_values=['NA', 'null', ''],  # Missing value indicators
                 parse_dates=['date_column'],   # Parse dates
                 index_col=0)       # Set index column
```

### 2.3 Creating Sample Dataset for Examples

```python
# Create a comprehensive sample dataset for demonstration
np.random.seed(42)

n_samples = 1000

df = pd.DataFrame({
    'customer_id': range(1, n_samples + 1),
    'age': np.random.randint(18, 80, n_samples),
    'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
    'income': np.random.normal(50000, 20000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'purchase_amount': np.random.exponential(100, n_samples),
    'num_purchases': np.random.poisson(5, n_samples),
    'satisfaction': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.2, 0.5, 0.3]),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
    'signup_date': pd.date_range('2020-01-01', periods=n_samples, freq='D')
})

# Introduce some missing values
df.loc[np.random.choice(df.index, 50, replace=False), 'income'] = np.nan
df.loc[np.random.choice(df.index, 30, replace=False), 'satisfaction'] = np.nan

# Introduce some outliers
df.loc[np.random.choice(df.index, 10, replace=False), 'purchase_amount'] = np.random.uniform(1000, 5000, 10)

print("Sample dataset created successfully!")
print(f"Dataset shape: {df.shape}")
```

---

## 3. Initial Data Inspection

### 3.1 Basic Information

```python
# Display first few rows
print("First 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

# Random sample
print("\nRandom sample of 5 rows:")
print(df.sample(5))

# Dataset shape
print(f"\nDataset shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Column names
print(f"\nColumn names: {df.columns.tolist()}")

# Data types
print("\nData types:")
print(df.dtypes)

# Detailed information
print("\nDetailed info:")
print(df.info())
```

### 3.2 Memory Usage

```python
# Memory usage
print("Memory usage:")
print(df.memory_usage(deep=True))

print(f"\nTotal memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Optimize memory (if needed)
def optimize_dtypes(df):
    """Optimize data types to reduce memory usage"""
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

# df = optimize_dtypes(df)
```

### 3.3 Statistical Summary

```python
# Summary statistics for numerical columns
print("Summary statistics (numerical):")
print(df.describe())

# Include all percentiles
print("\nDetailed percentiles:")
print(df.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]))

# Summary for categorical columns
print("\nSummary statistics (categorical):")
print(df.describe(include=['object', 'category']))

# Summary for all columns
print("\nSummary for all columns:")
print(df.describe(include='all'))
```

### 3.4 Unique Values and Cardinality


Cardinality refers to the number of unique values in a feature/column. It indicates the "uniqueness" level of data.

```python
# Unique values per column
print("Unique values count:")
for col in df.columns:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count} unique values")

# Cardinality analysis
def analyze_cardinality(df):
    """Analyze cardinality of each column"""
    cardinality = pd.DataFrame({
        'column': df.columns,
        'unique_values': [df[col].nunique() for col in df.columns],
        'total_values': [df[col].count() for col in df.columns]
    })
    cardinality['cardinality_ratio'] = cardinality['unique_values'] / cardinality['total_values']
    cardinality['type'] = cardinality['cardinality_ratio'].apply(
        lambda x: 'High' if x > 0.5 else ('Medium' if x > 0.05 else 'Low')
    )
    return cardinality.sort_values('cardinality_ratio', ascending=False)

print("\nCardinality Analysis:")
print(analyze_cardinality(df))
```

---

## 4. Data Cleaning

### 4.1 Missing Values Analysis

```python
# Count missing values
print("Missing values count:")
missing_count = df.isnull().sum()
print(missing_count[missing_count > 0])

# Missing values percentage
print("\nMissing values percentage:")
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent[missing_percent > 0])

# Visualize missing values
def plot_missing_values(df):
    """Visualize missing values pattern"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Missing values count
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    axes[0].barh(missing.index, missing.values)
    axes[0].set_xlabel('Count')
    axes[0].set_title('Missing Values Count')
    
    # Missing values heatmap
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, 
                cmap='viridis', ax=axes[1])
    axes[1].set_title('Missing Values Heatmap')
    
    plt.tight_layout()
    plt.show()

plot_missing_values(df)

# Missing value matrix
import missingno as msno  # pip install missingno
msno.matrix(df)
plt.show()
```

### 4.2 Handling Missing Values

```python
# Strategy 1: Remove rows with missing values
df_dropped_rows = df.dropna()
print(f"Rows after dropping NaN: {df_dropped_rows.shape[0]}")

# Strategy 2: Remove columns with missing values
df_dropped_cols = df.dropna(axis=1)
print(f"Columns after dropping: {df_dropped_cols.shape[1]}")

# Strategy 3: Remove rows with missing values in specific columns
df_dropped_specific = df.dropna(subset=['income'])

# Strategy 4: Fill with constant value
df_filled_constant = df.fillna(0)

# Strategy 5: Fill with mean/median/mode
df_filled = df.copy()
df_filled['income'] = df_filled['income'].fillna(df_filled['income'].median())
df_filled['satisfaction'] = df_filled['satisfaction'].fillna(df_filled['satisfaction'].mode()[0])

# Strategy 6: Forward fill / Backward fill (for time series)
df_ffill = df.fillna(method='ffill')
df_bfill = df.fillna(method='bfill')

# Strategy 7: Interpolation
df_interpolated = df.copy()
df_interpolated['income'] = df_interpolated['income'].interpolate(method='linear')

# Strategy 8: Advanced imputation
from sklearn.impute import SimpleImputer, KNNImputer

# Simple imputer
imputer = SimpleImputer(strategy='mean')
df_imputed = df.copy()
numerical_cols = df.select_dtypes(include=[np.number]).columns
df_imputed[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# KNN imputer
knn_imputer = KNNImputer(n_neighbors=5)
df_knn_imputed = df.copy()
df_knn_imputed[numerical_cols] = knn_imputer.fit_transform(df[numerical_cols])

print("Missing values handled!")
```

### 4.3 Duplicate Detection and Removal

```python
# Check for duplicates
print(f"Number of duplicate rows: {df.duplicated().sum()}")

# Find duplicate rows
duplicates = df[df.duplicated(keep=False)]
print(f"\nDuplicate rows:\n{duplicates}")

# Check duplicates based on specific columns
duplicates_subset = df[df.duplicated(subset=['customer_id'], keep=False)]

# Remove duplicates
df_no_duplicates = df.drop_duplicates()
print(f"Rows after removing duplicates: {df_no_duplicates.shape[0]}")

# Remove duplicates keeping first/last occurrence
df_keep_first = df.drop_duplicates(keep='first')
df_keep_last = df.drop_duplicates(keep='last')

# Remove duplicates based on specific columns
df_no_dup_subset = df.drop_duplicates(subset=['customer_id'])
```

### 4.4 Data Type Conversion

```python
# Convert data types
df_converted = df.copy()

# Convert to numeric
df_converted['age'] = pd.to_numeric(df_converted['age'], errors='coerce')

# Convert to categorical
df_converted['gender'] = df_converted['gender'].astype('category')
df_converted['satisfaction'] = pd.Categorical(df_converted['satisfaction'], 
                                              categories=['Low', 'Medium', 'High'], 
                                              ordered=True)

# Convert to datetime
df_converted['signup_date'] = pd.to_datetime(df_converted['signup_date'])

# Convert to string
df_converted['customer_id'] = df_converted['customer_id'].astype(str)

print("Data types after conversion:")
print(df_converted.dtypes)
```

### 4.5 Handling Inconsistent Data

```python
# String cleaning
df_cleaned = df.copy()

# Remove whitespace
df_cleaned['region'] = df_cleaned['region'].str.strip()

# Convert to lowercase/uppercase
df_cleaned['region'] = df_cleaned['region'].str.lower()

# Replace values
df_cleaned['gender'] = df_cleaned['gender'].replace({'Male': 'M', 'Female': 'F'})

# Remove special characters
# df_cleaned['column'] = df_cleaned['column'].str.replace('[^a-zA-Z0-9]', '', regex=True)

# Standardize formats
def standardize_phone(phone):
    """Standardize phone number format"""
    # Remove all non-numeric characters
    digits = ''.join(filter(str.isdigit, str(phone)))
    # Format as (XXX) XXX-XXXX
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    return phone

# df_cleaned['phone'] = df_cleaned['phone'].apply(standardize_phone)
```

---

## 5. Univariate Analysis

### 5.1 Numerical Variables

```python
# Select numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
print(f"Numerical columns: {numerical_cols.tolist()}")

# Distribution analysis for each numerical column
for col in ['age', 'income', 'purchase_amount']:
    print(f"\n{'='*50}")
    print(f"Analysis of {col}")
    print(f"{'='*50}")
    
    # Basic statistics
    print(f"Mean: {df[col].mean():.2f}")
    print(f"Median: {df[col].median():.2f}")
    print(f"Mode: {df[col].mode().values[0]:.2f}")
    print(f"Std Dev: {df[col].std():.2f}")
    print(f"Variance: {df[col].var():.2f}")
    print(f"Range: {df[col].max() - df[col].min():.2f}")
    print(f"IQR: {df[col].quantile(0.75) - df[col].quantile(0.25):.2f}")
    
    # Skewness and Kurtosis
    print(f"Skewness: {df[col].skew():.2f}")
    print(f"Kurtosis: {df[col].kurtosis():.2f}")
```

### 5.2 Visualizing Numerical Variables

```python
def plot_numerical_distribution(df, column, bins=30):
    """Comprehensive visualization for numerical variable"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram
    axes[0, 0].hist(df[column].dropna(), bins=bins, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df[column].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 0].axvline(df[column].median(), color='green', linestyle='--', label='Median')
    axes[0, 0].set_xlabel(column)
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Histogram of {column}')
    axes[0, 0].legend()
    
    # Box plot
    axes[0, 1].boxplot(df[column].dropna())
    axes[0, 1].set_ylabel(column)
    axes[0, 1].set_title(f'Box Plot of {column}')
    
    # Density plot
    df[column].dropna().plot(kind='density', ax=axes[1, 0])
    axes[1, 0].set_xlabel(column)
    axes[1, 0].set_title(f'Density Plot of {column}')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(df[column].dropna(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f'Q-Q Plot of {column}')
    
    plt.tight_layout()
    plt.show()

# Plot for each numerical variable
plot_numerical_distribution(df, 'income')
plot_numerical_distribution(df, 'age')
plot_numerical_distribution(df, 'purchase_amount')
```

### 5.3 Categorical Variables

```python
# Select categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
print(f"Categorical columns: {categorical_cols.tolist()}")

# Frequency analysis
for col in ['gender', 'satisfaction', 'region']:
    print(f"\n{'='*50}")
    print(f"Frequency Distribution of {col}")
    print(f"{'='*50}")
    
    # Value counts
    print(df[col].value_counts())
    
    # Percentage distribution
    print("\nPercentage Distribution:")
    print(df[col].value_counts(normalize=True) * 100)
    
    # Mode
    print(f"\nMode: {df[col].mode().values[0]}")
```

### 5.4 Visualizing Categorical Variables

```python
def plot_categorical_distribution(df, column):
    """Comprehensive visualization for categorical variable"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Bar plot
    value_counts = df[column].value_counts()
    axes[0].bar(value_counts.index, value_counts.values, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel(column)
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Bar Plot of {column}')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Pie chart
    axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title(f'Pie Chart of {column}')
    
    # Horizontal bar plot (sorted)
    value_counts_sorted = value_counts.sort_values()
    axes[2].barh(value_counts_sorted.index, value_counts_sorted.values, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Count')
    axes[2].set_ylabel(column)
    axes[2].set_title(f'Sorted Bar Plot of {column}')
    
    plt.tight_layout()
    plt.show()

# Plot for each categorical variable
plot_categorical_distribution(df, 'gender')
plot_categorical_distribution(df, 'satisfaction')
plot_categorical_distribution(df, 'region')
```

### 5.5 Time Series Analysis (for date columns)

```python
# Extract time-based features
df['year'] = df['signup_date'].dt.year
df['month'] = df['signup_date'].dt.month
df['day'] = df['signup_date'].dt.day
df['dayofweek'] = df['signup_date'].dt.dayofweek
df['quarter'] = df['signup_date'].dt.quarter

# Time-based aggregation
signups_by_month = df.groupby(df['signup_date'].dt.to_period('M')).size()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Signups over time
signups_by_month.plot(ax=axes[0, 0])
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Signups Over Time')

# Signups by day of week
df['dayofweek'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_xlabel('Day of Week (0=Monday)')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Signups by Day of Week')

# Signups by month
df['month'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0])
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Signups by Month')

# Signups by quarter
df['quarter'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_xlabel('Quarter')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Signups by Quarter')

plt.tight_layout()
plt.show()
```

---

## 6. Bivariate Analysis

### 6.1 Numerical vs Numerical

```python
# Correlation analysis
correlation_matrix = df[numerical_cols].corr()

print("Correlation Matrix:")
print(correlation_matrix)

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix Heatmap')
plt.show()

# Scatter plots
def plot_scatter_matrix(df, columns):
    """Create scatter plot matrix"""
    pd.plotting.scatter_matrix(df[columns], figsize=(15, 15), 
                              alpha=0.5, diagonal='hist')
    plt.suptitle('Scatter Plot Matrix', y=1.0)
    plt.show()

plot_scatter_matrix(df, ['age', 'income', 'purchase_amount', 'credit_score'])

# Individual scatter plots with regression line
def plot_scatter_with_regression(df, x_col, y_col):
    """Scatter plot with regression line"""
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x=x_col, y=y_col, scatter_kws={'alpha':0.5})
    plt.title(f'{x_col} vs {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    
    # Calculate and display correlation
    corr = df[[x_col, y_col]].corr().iloc[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    plt.show()

plot_scatter_with_regression(df, 'age', 'income')
plot_scatter_with_regression(df, 'income', 'purchase_amount')
```

### 6.2 Categorical vs Numerical

```python
# Group statistics
print("Mean purchase amount by gender:")
print(df.groupby('gender')['purchase_amount'].mean())

print("\nMedian income by satisfaction:")
print(df.groupby('satisfaction')['income'].median())

# Box plots
def plot_cat_vs_num(df, cat_col, num_col):
    """Visualize categorical vs numerical relationship"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Box plot
    df.boxplot(column=num_col, by=cat_col, ax=axes[0])
    axes[0].set_title(f'Box Plot: {num_col} by {cat_col}')
    axes[0].set_xlabel(cat_col)
    axes[0].set_ylabel(num_col)
    plt.sca(axes[0])
    plt.xticks(rotation=45)
    
    # Violin plot
    sns.violinplot(data=df, x=cat_col, y=num_col, ax=axes[1])
    axes[1].set_title(f'Violin Plot: {num_col} by {cat_col}')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Bar plot (mean)
    df.groupby(cat_col)[num_col].mean().plot(kind='bar', ax=axes[2], 
                                             edgecolor='black', alpha=0.7)
    axes[2].set_title(f'Mean {num_col} by {cat_col}')
    axes[2].set_xlabel(cat_col)
    axes[2].set_ylabel(f'Mean {num_col}')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

plot_cat_vs_num(df, 'gender', 'income')
plot_cat_vs_num(df, 'satisfaction', 'purchase_amount')
plot_cat_vs_num(df, 'region', 'age')
```

### 6.3 Categorical vs Categorical

```python
# Contingency table (cross-tabulation)
contingency_table = pd.crosstab(df['gender'], df['satisfaction'])
print("Contingency Table:")
print(contingency_table)

# Contingency table with percentages
contingency_pct = pd.crosstab(df['gender'], df['satisfaction'], 
                              normalize='index') * 100
print("\nContingency Table (Percentages):")
print(contingency_pct)

# Chi-square test
from scipy.stats import chi2_contingency

chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square test:")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")

# Visualization
def plot_cat_vs_cat(df, cat1, cat2):
    """Visualize categorical vs categorical relationship"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Stacked bar chart
    contingency = pd.crosstab(df[cat1], df[cat2])
    contingency.plot(kind='bar', stacked=True, ax=axes[0], edgecolor='black')
    axes[0].set_title(f'Stacked Bar Chart: {cat1} vs {cat2}')
    axes[0].set_xlabel(cat1)
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(title=cat2)
    
    # Grouped bar chart
    contingency.plot(kind='bar', ax=axes[1], edgecolor='black', alpha=0.7)
    axes[1].set_title(f'Grouped Bar Chart: {cat1} vs {cat2}')
    axes[1].set_xlabel(cat1)
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title=cat2)
    
    # Heatmap
    sns.heatmap(contingency, annot=True, fmt='d', cmap='YlOrRd', ax=axes[2])
    axes[2].set_title(f'Heatmap: {cat1} vs {cat2}')
    axes[2].set_xlabel(cat2)
    axes[2].set_ylabel(cat1)
    
    plt.tight_layout()
    plt.show()

plot_cat_vs_cat(df, 'gender', 'satisfaction')
plot_cat_vs_cat(df, 'region', 'satisfaction')
```

---

## 7. Multivariate Analysis

### 7.1 Correlation Analysis (Multiple Variables)

```python
# Advanced correlation visualization
def plot_advanced_correlation(df, columns):
    """Create advanced correlation visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Correlation heatmap
    corr = df[columns].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=axes[0, 0], cbar_kws={"shrink": 0.8})
    axes[0, 0].set_title('Correlation Heatmap')
    
    # Clustermap
    sns.clustermap(corr, annot=True, cmap='coolwarm', center=0, 
                   figsize=(10, 10), cbar_kws={"shrink": 0.8})
    
    # Correlation with target (if exists)
    if 'purchase_amount' in columns:
        target_corr = corr['purchase_amount'].sort_values(ascending=False)
        axes[0, 1].barh(target_corr.index, target_corr.values, edgecolor='black')
        axes[0, 1].set_xlabel('Correlation with Purchase Amount')
        axes[0, 1].set_title('Feature Correlation with Target')
    
    # Pairplot subset
    plt.figure(figsize=(15, 15))
    sns.pairplot(df[columns].sample(min(500, len(df))), diag_kind='kde')
    plt.suptitle('Pair Plot', y=1.0)
    plt.show()

plot_advanced_correlation(df, ['age', 'income', 'credit_score', 'purchase_amount'])
```

### 7.2 Three-Variable Analysis

```python
# Scatter plot with third variable as color
def plot_3d_scatter(df, x, y, hue):
    """3D scatter plot using color for third variable"""
    plt.figure(figsize=(12, 8))
    
    if df[hue].dtype in ['object', 'category']:
        # Categorical hue
        sns.scatterplot(data=df, x=x, y=y, hue=hue, s=100, alpha=0
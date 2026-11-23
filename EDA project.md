# 📊 Comprehensive Exploratory Data Analysis (EDA) Project  
  
## Table of Contents  
1. [Introduction](#1-introduction)  
2. [Import Required Libraries](#2-import-required-libraries)  
3. [Load the Dataset](#3-load-the-dataset)  
4. [Initial Data Exploration](#4-initial-data-exploration)  
5. [Statistical Summary](#5-statistical-summary)  
6. [Missing Values Analysis](#6-missing-values-analysis)  
7. [Duplicate Values Analysis](#7-duplicate-values-analysis)  
8. [Univariate Analysis](#8-univariate-analysis)  
9. [Bivariate Analysis](#9-bivariate-analysis)  
10. [Multivariate Analysis](#10-multivariate-analysis)  
11. [Outlier Detection](#11-outlier-detection)  
12. [Feature Engineering](#12-feature-engineering)  
13. [Correlation Analysis](#13-correlation-analysis)  
14. [Key Insights and Conclusions](#14-key-insights-and-conclusions)  
15. [Recommendations](#15-recommendations)  
  
---  
  
## 1. Introduction  
  
### What is Exploratory Data Analysis (EDA)?  
  
**Exploratory Data Analysis (EDA)** is a critical first step in any data science project. It involves analyzing datasets to summarize their main characteristics, often using visual methods. This process helps us understand the data's structure, detect outliers, identify patterns, and formulate hypotheses.  
  
### Objectives of EDA  
  
| Objective | Description |  
|-----------|-------------|  
| **Data Quality Assessment** | Identify missing values, duplicates, and inconsistencies |  
| **Feature Understanding** | Understand the distribution and nature of each feature |  
| **Relationship Discovery** | Find correlations and dependencies between variables |  
| **Hypothesis Generation** | Form initial hypotheses for further analysis |  
| **Model Preparation** | Prepare data for machine learning models |  
  
### Types of EDA  
  
1. **Univariate Analysis**: Examining single variables  
2. **Bivariate Analysis**: Examining relationships between two variables  
3. **Multivariate Analysis**: Examining relationships among multiple variables  
  
### Dataset Overview  
  
In this project, we'll perform EDA on the famous **Titanic Dataset** - a classic dataset for learning data analysis techniques.  
  
#### About the Titanic Dataset  
  
The Titanic dataset contains information about passengers aboard the RMS Titanic, which sank on April 15, 1912.  
  
| Column | Description | Data Type |  
|--------|-------------|-----------|  
| survived | Survival status (0 = No, 1 = Yes) | Integer |  
| pclass | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) | Integer |  
| sex | Gender | String |  
| age | Age in years | Float |  
| sibsp | Number of siblings/spouses aboard | Integer |  
| parch | Number of parents/children aboard | Integer |  
| fare | Passenger fare | Float |  
| embarked | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) | String |  
| class | Ticket class (First, Second, Third) | String |  
| who | Man, Woman, or Child | String |  
| deck | Deck letter | String |  
| embark_town | Port of embarkation name | String |  
| alive | Survival status (yes/no) | String |  
| alone | Whether passenger was alone | Boolean |  
  
---  
  
## 2. Import Required Libraries  
  
### Required Python Libraries  
  
```python  
# Data Manipulation and Analysis  
import pandas as pd  
import numpy as np  
  
# Data Visualization  
import matplotlib.pyplot as plt  
import seaborn as sns  
  
# Statistical Analysis  
from scipy import stats  
from scipy.stats import norm, skew, kurtosis  
  
# Warnings  
import warnings  
warnings.filterwarnings('ignore')  
  
# Display Settings  
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', 100)  
pd.set_option('display.float_format', lambda x: '%.3f' % x)  
  
# Plot Style  
plt.style.use('seaborn-v0_8-whitegrid')  
sns.set_palette('husl')  
plt.rcParams['figure.figsize'] = (12, 6)  
plt.rcParams['font.size'] = 12  
  
print("✅ Libraries imported successfully!")  
```  
  
### Library Descriptions  
  
| Library | Purpose |  
|---------|---------|  
| **pandas** | Data manipulation and analysis |  
| **numpy** | Numerical computing |  
| **matplotlib** | Basic plotting and visualization |  
| **seaborn** | Statistical data visualization |  
| **scipy** | Scientific computing and statistics |  
  
---  
  
## 3. Load the Dataset  
  
### Loading Data  
  
```python  
# Load the Titanic dataset from seaborn's built-in datasets  
df = sns.load_dataset('titanic')  
  
# Create a copy for manipulation  
df_original = df.copy()  
  
# Display basic information  
print(f"Dataset Shape: {df.shape}")  
print(f"Number of Rows: {df.shape[0]}")  
print(f"Number of Columns: {df.shape[1]}")  
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")  
```  
  
### Expected Output  
  
```  
Dataset Shape: (891, 15)  
Number of Rows: 891  
Number of Columns: 15  
Memory Usage: 83.45 KB  
```  
  
---  
  
## 4. Initial Data Exploration  
  
### 4.1 View First Few Rows  
  
```python  
# Display first 10 rows  
print("📋 First 10 rows of the dataset:")  
df.head(10)  
```  
  
### 4.2 View Last Few Rows  
  
```python  
# Display last 5 rows  
print("📋 Last 5 rows of the dataset:")  
df.tail()  
```  
  
### 4.3 Random Sample  
  
```python  
# Display random sample of 5 rows  
print("📋 Random sample of 5 rows:")  
df.sample(5, random_state=42)  
```  
  
### 4.4 Column Information  
  
```python  
# Display column names and data types  
print("📋 Column Names and Data Types:")  
print("=" * 50)  
for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):  
    print(f"{i:2}. {col:15} : {dtype}")  
```  
  
### 4.5 Dataset Information  
  
```python  
# Display comprehensive dataset info  
print("📋 Dataset Information:")  
df.info()  
```  
  
### Expected Output  
  
```  
<class 'pandas.core.frame.DataFrame'>  
RangeIndex: 891 entries, 0 to 890  
Data columns (total 15 columns):  
 #   Column       Non-Null Count  Dtype     
---  ------       --------------  -----     
 0   survived     891 non-null    int64     
 1   pclass       891 non-null    int64     
 2   sex          891 non-null    object    
 3   age          714 non-null    float64   
 4   sibsp        891 non-null    int64     
 5   parch        891 non-null    int64     
 6   fare         891 non-null    float64   
 7   embarked     889 non-null    object    
 8   class        891 non-null    category  
 9   who          891 non-null    object    
 10  adult_male   891 non-null    bool      
 11  deck         203 non-null    category  
 12  embark_town  889 non-null    object    
 13  alive        891 non-null    object    
 14  alone        891 non-null    bool      
dtypes: bool(2), category(2), float64(2), int64(4), object(5)  
memory usage: 80.7+ KB  
```  
  
### 4.6 Identify Column Types  
  
```python  
# Separate numerical and categorical columns  
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()  
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()  
  
print("🔢 Numerical Columns:")  
print(numerical_cols)  
print(f"\n📝 Categorical Columns:")  
print(categorical_cols)  
```  
  
### Expected Output  
  
```  
🔢 
Numerical Columns:  
['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']  
  
📝 Categorical Columns:  
['sex', 'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone']  
```  
  
---  
  
## 5. Statistical Summary  
  
### 5.1 Numerical Features Statistics  
  
```python  
# Statistical summary for numerical columns  
print("📊 Statistical Summary - Numerical Features:")  
df.describe().T  
```  
  
### Expected Output  
  
| Feature | count | mean | std | min | 25% | 50% | 75% | max |  
|---------|-------|------|-----|-----|-----|-----|-----|-----|  
| survived | 891.0 | 0.384 | 0.487 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |  
| pclass | 891.0 | 2.309 | 0.836 | 1.0 | 2.0 | 3.0 | 3.0 | 3.0 |  
| age | 714.0 | 29.699 | 14.526 | 0.42 | 20.125 | 28.0 | 38.0 | 80.0 |  
| sibsp | 891.0 | 0.523 | 1.103 | 0.0 | 0.0 | 0.0 | 1.0 | 8.0 |  
| parch | 891.0 | 0.382 | 0.806 | 0.0 | 0.0 | 0.0 | 0.0 | 6.0 |  
| fare | 891.0 | 32.204 | 49.693 | 0.0 | 7.910 | 14.454 | 31.0 | 512.329 |  
  
### 5.2 Extended Statistical Summary  
  
```python  
# Extended statistics with additional percentiles  
print("📊 Extended Statistical Summary:")  
df.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T  
```  
  
### 5.3 Categorical Features Statistics  
  
```python  
# Statistical summary for categorical columns  
print("📊 Statistical Summary - Categorical Features:")  
df.describe(include=['object', 'category', 'bool']).T  
```  
  
### Expected Output  
  
| Feature | count | unique | top | freq |  
|---------|-------|--------|-----|------|  
| sex | 891 | 2 | male | 577 |  
| embarked | 889 | 3 | S | 644 |  
| class | 891 | 3 | Third | 491 |  
| who | 891 | 3 | man | 537 |  
| deck | 203 | 7 | C | 59 |  
| embark_town | 889 | 3 | Southampton | 644 |  
| alive | 891 | 2 | no | 549 |  
  
### 5.4 Custom Statistical Function  
  
```python  
def detailed_statistics(df, column):  
    """  
    Calculate detailed statistics for a numerical column.  
      
    Parameters:  
    -----------  
    df : pandas DataFrame  
    column : str - column name  
      
    Returns:  
    --------  
    dict : Dictionary containing all statistics  
    """  
    data = df[column].dropna()  
      
    stats = {  
        'Count': len(data),  
        'Missing': df[column].isnull().sum(),  
        'Missing %': (df[column].isnull().sum() / len(df)) * 100,  
        'Mean': data.mean(),  
        'Median': data.median(),  
        'Mode': data.mode().iloc[0] if len(data.mode()) > 0 else np.nan,  
        'Std Dev': data.std(),  
        'Variance': data.var(),  
        'Min': data.min(),  
        'Max': data.max(),  
        'Range': data.max() - data.min(),  
        'Q1 (25%)': data.quantile(0.25),  
        'Q2 (50%)': data.quantile(0.50),  
        'Q3 (75%)': data.quantile(0.75),  
        'IQR': data.quantile(0.75) - data.quantile(0.25),  
        'Skewness': data.skew(),  
        'Kurtosis': data.kurtosis(),  
        'Coef. of Variation': (data.std() / data.mean()) * 100  
    }  
      
    return stats  
  
# Example usage  
for col in numerical_cols:  
    print(f"\n{'='*50}")  
    print(f"📊 Statistics for: {col.upper()}")  
    print('='*50)  
    stats = detailed_statistics(df, col)  
    for key, value in stats.items():  
        if isinstance(value, float):  
            print(f"  {key:20}: {value:>12.4f}")  
        else:  
            print(f"  {key:20}: {value:>12}")  
```  
  
### 5.5 Understanding Statistical Measures  
  
| Measure | Description | Interpretation |  
|---------|-------------|----------------|  
| **Mean** | Average value | Central tendency |  
| **Median** | Middle value | Robust to outliers |  
| **Mode** | Most frequent value | Common occurrence |  
| **Std Dev** | Spread around mean | Higher = more spread |  
| **Variance** | Squared std dev | Measure of dispersion |  
| **Skewness** | Asymmetry of distribution | 0 = symmetric, >0 = right-skewed, <0 = left-skewed |  
| **Kurtosis** | Tail heaviness | 0 = normal, >0 = heavy tails, <0 = light tails |  
| **IQR** | Interquartile range | Spread of middle 50% |  
  
---  
  
## 6. Missing Values Analysis  
  
### 6.1 Calculate Missing Values  
  
```python  
# Calculate missing values  
def missing_values_analysis(df):  
    """  
    Comprehensive missing values analysis.  
      
    Parameters:  
    -----------  
    df : pandas DataFrame  
      
    Returns:  
    --------  
    pandas DataFrame : Missing values summary  
    """  
    missing_values = df.isnull().sum()  
    missing_percentage = (df.isnull().sum() / len(df)) * 100  
      
    missing_df = pd.DataFrame({  
        'Column': df.columns,  
        'Missing Values': missing_values.values,  
        'Percentage (%)': missing_percentage.values,  
        'Data Type': df.dtypes.values,  
        'Non-Null Count': df.notnull().sum().values  
    })  
      
    missing_df = missing_df.sort_values('Missing Values', ascending=False)  
    missing_df = missing_df.reset_index(drop=True)  
      
    return missing_df  
  
# Perform analysis  
missing_df = missing_values_analysis(df)  
print("📊 Missing Values Analysis:")  
print("=" * 70)  
print(missing_df.to_string(index=False))  
```  
  
### Expected Output  
  
```  
📊 Missing Values Analysis:  
======================================================================  
     Column  Missing Values  Percentage (%)    Data Type  Non-Null Count  
       deck             688           77.22     category             203  
        age             177           19.87      float64             714  
   embarked               2            0.22       object             889  
embark_town               2            0.22       object             889  
   survived               0            0.00        int64             891  
     pclass               0            0.00        int64             891  
        sex               0            0.00       object             891  
      sibsp               0            0.00        int64             891  
      parch               0            0.00        int64             891  
       fare               0            0.00      float64             891  
      class               0            0.00     category             891  
        who               0            0.00       object             891  
 adult_male               0            0.00         bool             

 ```

 891  
      alive               0            0.00       object             891  
      alone               0            0.00         bool             891  
```  
  
### 6.2 Visualize Missing Values  
  
```  
# Visualization of missing values  
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  
  
# Plot 1: Bar chart of missing values  
ax1 = axes[0]  
missing_cols = missing_df[missing_df['Missing Values'] > 0]  
colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(missing_cols)))  
bars = ax1.barh(missing_cols['Column'], missing_cols['Percentage (%)'], color=colors)  
ax1.set_xlabel('Missing Percentage (%)', fontsize=12)  
ax1.set_ylabel('Columns', fontsize=12)  
ax1.set_title('Missing Values by Column', fontsize=14, fontweight='bold')  
ax1.bar_label(bars, fmt='%.1f%%', padding=3)  
ax1.set_xlim(0, 100)  
  
# Plot 2: Heatmap of missing values  
ax2 = axes[1]  
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='YlOrRd', ax=ax2)  
ax2.set_title('Missing Values Heatmap', fontsize=14, fontweight='bold')  
ax2.set_xlabel('Columns', fontsize=12)  
ax2.set_ylabel('Rows', fontsize=12)  
  
# Plot 3: Pie chart of data completeness  
ax3 = axes[2]  
total_cells = df.shape[0] * df.shape[1]  
missing_cells = df.isnull().sum().sum()  
complete_cells = total_cells - missing_cells  
  
sizes = [complete_cells, missing_cells]  
labels = ['Complete Data', 'Missing Data']  
colors_pie = ['#2ecc71', '#e74c3c']  
explode = (0, 0.1)  
  
ax3.pie(sizes, explode=explode, labels=labels, colors=colors_pie,  
        autopct='%1.1f%%', shadow=True, startangle=90)  
ax3.set_title('Data Completeness Overview', fontsize=14, fontweight='bold')  
  
plt.tight_layout()  
plt.savefig('missing_values_analysis.png', dpi=300, bbox_inches='tight')  
plt.show()  
  
# Print summary  
print(f"\n📈 Missing Values Summary:")  
print(f"   • Total cells: {total_cells:,}")  
print(f"   • Missing cells: {missing_cells:,}")  
print(f"   • Complete cells: {complete_cells:,}")  
print(f"   • Overall completeness: {(complete_cells/total_cells)*100:.2f}%") 
```  
  
### 6.3 Missing Values Patterns  
  
```  
# Analyze missing value patterns  
def missing_patterns(df):  
    """  
    Analyze patterns in missing values.  
    """  
    # Create a binary matrix (1 = missing, 0 = present)  
    missing_matrix = df.isnull().astype(int)  
      
    # Find unique patterns  
    patterns = missing_matrix.drop_duplicates()  
    pattern_counts = missing_matrix.groupby(list(missing_matrix.columns)).size().reset_index(name='count')  
    pattern_counts = pattern_counts.sort_values('count', ascending=False)  
      
    print("📊 Missing Value Patterns:")  
    print("=" * 60)  
    print(f"Total unique patterns: {len(patterns)}")  
    print(f"\nTop 5 most common patterns:")  
    print(pattern_counts.head())  
      
    return pattern_counts  
  
patterns = missing_patterns(df)  
```  
  
### 6.4 Strategies for Handling Missing Values  
  
| Strategy | When to Use | Code Example |  
|----------|-------------|--------------|  
| **Drop Rows** | Small % missing, random | `df.dropna()` |  
| **Drop Columns** | >50% missing | `df.drop(columns=['col'])` |  
| **Mean Imputation** | Numerical, normal dist | `df['col'].fillna(df['col'].mean())` |  
| **Median Imputation** | Numerical, skewed | `df['col'].fillna(df['col'].median())` |  
| **Mode Imputation** | Categorical | `df['col'].fillna(df['col'].mode()[0])` |  
| **Forward Fill** | Time series | `df['col'].fillna(method='ffill')` |  
| **Backward Fill** | Time series | `df['col'].fillna(method='bfill')` |  
| **KNN Imputation** | Complex patterns | `KNNImputer()` |  
  
```  
# Example: Handling missing values in Titanic dataset  
df_cleaned = df.copy()  
  
# Fill age with median (due to skewness)  
df_cleaned['age'].fillna(df_cleaned['age'].median(), inplace=True)  
  
# Fill embarked with mode  
df_cleaned['embarked'].fillna(df_cleaned['embarked'].mode()[0], inplace=True)  
df_cleaned['embark_town'].fillna(df_cleaned['embark_town'].mode()[0], inplace=True)  
  
# Drop deck column (too many missing values - 77%)  
df_cleaned.drop(columns=['deck'], inplace=True)  
  
print("✅ Missing values handled!")  
print(f"Remaining missing values: {df_cleaned.isnull().sum().sum()}")  
```  
  
---  
  
## 7. Duplicate Values Analysis  
  
### 7.1 Check for Duplicates  
  
```python  
# Check for duplicate rows  
def duplicate_analysis(df):  
    """  
    Comprehensive duplicate analysis.  
      
    Parameters:  
    -----------  
    df : pandas DataFrame  
      
    Returns:  
    --------  
    dict : Duplicate analysis results  
    """  
    # Exact duplicates  
    exact_duplicates = df.duplicated().sum()  
      
    # Duplicates keeping first  
    duplicates_keep_first = df.duplicated(keep='first').sum()  
      
    # Duplicates keeping last  
    duplicates_keep_last = df.duplicated(keep='last').sum()  
      
    results = {  
        'Total Rows': len(df),  
        'Exact Duplicates': exact_duplicates,  
        'Duplicate Percentage': (exact_duplicates / len(df)) * 100,  
        'Unique Rows': len(df) - exact_duplicates  
    }  
      
    return results  
  
# Perform analysis  
dup_results = duplicate_analysis(df)  
  
print("📊 Duplicate Values Analysis:")  
print("=" * 50)  
for key, value in dup_results.items():  
    if isinstance(value, float):  
        print(f"  {key:25}: {value:.2f}")  
    else:  
        print(f"  {key:25}: {value}")  
```  
  
### Expected Output  
  
```  
📊 Duplicate Values Analysis:  
==================================================  
  Total Rows               : 891  
  Exact Duplicates         : 0  
  Duplicate Percentage     : 0.00  
  Unique Rows              : 891  
```  
  
### 7.2 Check Duplicates in Specific Columns  
  
```python  
# Check duplicates in specific columns  
def check_column_duplicates(df, columns):  
    """  
    Check for duplicates in specific columns.  
    """  
    duplicates = df.duplicated(subset=columns, keep=False)  
    duplicate_rows = df[duplicates]  
      
    print(f"📊 Duplicates in columns: {columns}")  
    print(f"   Duplicate rows: {len(duplicate_rows)}")  
      
    return duplicate_rows  
  
# Example: Check duplicates by name (if exists)  
# check_column_duplicates(df, ['name'])  
```  
  
### 7.3 Handling Duplicates  
  
```python  
# Remove duplicates  
df_no_duplicates = df.drop_duplicates()  
  
# Remove duplicates based on specific columns  
# df_no_duplic_duplicates(subset=['col1', 'col2'])  
  
# Keep first occurrence  
# df_no_duplicates = df.drop_duplicates(keep='first')  
  
# Keep last occurrence  
# df_no_duplicates = df.drop_duplicates(keep='last')  
  
# Remove all duplicates (including first occurrence)  
# df_no_duplicates = df.drop_duplicates(keep=False)  
  
print(f"✅ Duplicates removed!")  
print(f"   Original rows: {len(df)}")  
print(f"   After removal: {len(df_no_duplicates)}")  
print(f"   Rows removed: {len(df) - len(df_no_duplicates)}")  
```  
  
---  
  
## 8. Univariate Analysis  
  
Univariate analysis examines each variable individually to understand its distribution, central tendency, and spread.  
  
### 8.1 Analysis of Numerical Features  
  
#### 8.1.1 Distribution Plots (Histograms with KDE)  
  
```python  
# Create distribution plots for all numerical features  
def plot_numerical_distributions(df, numerical_cols):  
    """  
    Plot distribution of numerical features with histogram and KDE.  
    """  
    n_cols = 3  
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols  
      
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))  
    axes = axes.flatten()  
      
    for idx, col in enumerate(numerical_cols):  
        ax = axes[idx]  
        data = df[col].dropna()  
          
        # Histogram with KDE  
        sns.histplot(data=data, kde=True, ax=ax, color='steelblue',   
                     edgecolor='white', alpha=0.7)  
          
        # Add mean and median lines  
        mean_val = data.mean()  
        median_val = data.median()  
          
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,   
                   label=f'Mean: {mean_val:.2f}')  
        ax.axvline(median_val, color='green', linestyle='-.', linewidth=2,   
                   label=f'Median: {median_val:.2f}')  
          
        ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')  
        ax.set_xlabel(col, fontsize=10)  
        ax.set_ylabel('Frequency', fontsize=10)  
        ax.legend(fontsize=8)  
          
        # Add skewness annotation  
        skewness = data.skew()  
        ax.annotate(f'Skew: {skewness:.2f}', xy=(0.95, 0.95),   
                    xycoords='axes fraction', ha='right', va='top',  
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))  
      
    # Hide empty subplots  
    for idx in range(len(numerical_cols), len(axes)):  
        axes[idx].set_visible(False)  
      
    plt.suptitle('Distribution of Numerical Features', fontsize=16,   
                 fontweight='bold', y=1.02)  
    plt.tight_layout()  
    plt.savefig('numerical_distributions.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Plot distributions  
plot_numerical_distributions(df, numerical_cols)  
```  
  
#### 8.1.2 Box Plots  
  
```python  
# Create box plots for numerical features  
def plot_boxplots(df, numerical_cols):  
    """  
    Plot box plots for numerical features to visualize spread and outliers.  
    """  
    n_cols = 3  
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols  
      
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))  
    axes = axes.flatten()  
      
    colors = sns.color_palette('husl', len(numerical_cols))  
      
    for idx, col in enumerate(numerical_cols):  
        ax = axes[idx]  
        data = df[col].dropna()  
          
        # Create box plot  
        bp = ax.boxplot(data, patch_artist=True, vert=True)  
        bp['boxes'][0].set_facecolor(colors[idx])  
        bp['boxes'][0].set_alpha(0.7)  
          
        # Add individual points (jittered)  
        x = np.random.normal(1, 0.04, size=len(data))  
        ax.scatter(x, data, alpha=0.3, s=10, color='gray')  
          
        ax.set_title(f'Box Plot of {col}', fontsize=12, fontweight='bold')  
        ax.set_ylabel(col, fontsize=10)  
        ax.set_xticklabels([col])  
          
        # Add statistics annotation  
        stats_text = f'Mean: {data.mean():.2f}\nMedian: {data.median():.2f}\nStd: {data.std():.2f}'  
        ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',  
                    ha='left', va='top', fontsize=8,  
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))  
      
    # Hide empty subplots  
    for idx in range(len(numerical_cols), len(axes)):  
        axes[idx].set_visible(False)  
      
    plt.suptitle('Box Plots of Numerical Features', fontsize=16,   
                 fontweight='bold', y=1.02)  
    plt.tight_layout()  
    plt.savefig('numerical_boxplots.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Plot box plots  
plot_boxplots(df, numerical_cols)  
```  
  
#### 8.1.3 Violin Plots  
  
```python  
# Create violin plots for numerical features  
def plot_violinplots(df, numerical_cols):  
    """  
    Plot violin plots combining box plot and KDE.  
    """  
    n_cols = 3  
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols  
      
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))  
    axes = axes.flatten()  
      
    for idx, col in enumerate(numerical_cols):  
        ax = axes[idx]  
        data = df[[col]].dropna()  
          
        sns.violinplot(data=data, y=col, ax=ax, color='lightblue',   
                       inner='box', linewidth=1.5)  
          
        ax.set_title(f'Violin Plot of {col}', fontsize=12, fontweight='bold')  
        ax.set_ylabel(col, fontsize=10)  
      
    # Hide empty subplots  
    for idx in range(len(numerical_cols), len(axes)):  
        axes[idx].set_visible(False)  
      
    plt.suptitle('Violin Plots of Numerical Features', fontsize=16,   
                 fontweight='bold', y=1.02)  
    plt.tight_layout()  
    plt.savefig('numerical_violinplots.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Plot violin plots  
plot_violinplots(df, numerical_cols)  
```  
  
#### 8.1.4 QQ Plots (Normality Check)  
  
```python  
# Create QQ plots to check normality  
def plot_qq_plots(df, numerical_cols):  
    """  
    Plot QQ plots to assess normality of distributions.  
    """  
    n_cols = 3  
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols  
      
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))  
    axes = axes.flatten()  
      
    for idx, col in enumerate(numerical_cols):  
        ax = axes[idx]  
        data = df[col].dropna()  
          
        # QQ plot  
        stats .probplot(data, dist="norm", plot=ax)  
          
        ax.set_title(f'QQ Plot of {col}', fontsize=12, fontweight='bold')  
        ax.set_xlabel('Theoretical Quantiles', fontsize=10)  
        ax.set_ylabel('Sample Quantiles', fontsize=10)  
          
        # Add Shapiro-Wilk test result  
        if len(data) <= 5000:  # Shapiro-Wilk works best for n <= 5000  
            stat, p_value = stats.shapiro(data.sample(min(len(data), 5000)))  
            normality = "Normal" if p_value > 0.05 else "Not Normal"  
            ax.annotate(f'Shapiro-Wilk p: {p_value:.4f}\n({normality})',   
                       xy=(0.05, 0.95), xycoords='axes fraction',  
                       ha='left', va='top', fontsize=8,  
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))  
      
    # Hide empty subplots  
    for idx in range(len(numerical_cols), len(axes)):  
        axes[idx].set_visible(False)  
      
    plt.suptitle('QQ Plots - Normality Assessment', fontsize=16,   
                 fontweight='bold', y=1.02)  
    plt.tight_layout()  
    plt.savefig('qq_plots.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Plot QQ plots  
plot_qq_plots(df, numerical_cols)  
```  
  
### 8.2 Analysis of Categorical Features  
  
#### 8.2.1 Value Counts and Frequency Tables  
  
```python  
# Analyze categorical features  
def analyze_categorical_features(df, categorical_cols):  
    """  
    Analyze and display value counts for categorical features.  
    """  
    for col in categorical_cols:  
        print(f"\n{'='*60}")  
        print(f"📊 Analysis of: {col.upper()}")  
        print('='*60)  
          
        # Value counts  
        value_counts = df[col].value_counts()  
        value_percentages = df[col].value_counts(normalize=True) * 100  
          
        # Create summary dataframe  
        summary = pd.DataFrame({  
            'Value': value_counts.index,  
            'Count': value_counts.values,  
            'Percentage (%)': value_percentages.values  
        })  
          
        print(f"\nUnique Values: {df[col].nunique()}")  
        print(f"Missing Values: {df[col].isnull().sum()}")  
        print(f"\nValue Distribution:")  
        print(summary.to_string(index=False))  
  
# Analyze categorical features  
analyze_categorical_features(df, categorical_cols)  
```  
  
### Expected Output (Sample)  
  
```  
============================================================  
📊 Analysis of: SEX  
============================================================  
  
Unique Values: 2  
Missing Values: 0  
  
Value Distribution:  
  Value  Count  Percentage (%)  
   male    577           64.76  
 female    314           35.24  
  
============================================================  
📊 Analysis of: EMBARKED  
============================================================  
  
Unique Values: 3  
Missing Values: 2  
  
Value Distribution:  
Value  Count  Percentage (%)  
    S    644           72.44  
    C    168           18.90  
    Q     77            8.66  
```  
  
#### 8.2.2 Bar Charts for Categorical Features  
  
```python  
# Create bar charts for categorical features  
def plot_categorical_bars(df, categorical_cols, max_categories=10):  
    """  
    Plot bar charts for categorical features.  
    """  
    # Filter columns with reasonable number of categories  
    plot_cols = [col for col in categorical_cols if df[col].nunique() <= max_categories]  
      
    n_cols = 3  
    n_rows = (len(plot_cols) + n_cols - 1) // n_cols  
      
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))  
    axes = axes.flatten()  
      
    colors = sns.color_palette('viridis', max_categories)  
      
    for idx, col in enumerate(plot_cols):  
        ax = axes[idx]  
          
        # Get value counts  
        value_counts = df[col].value_counts()  
          
        # Create bar plot  
        bars = ax.bar(range(len(value_counts)), value_counts.values,   
                      color=colors[:len(value_counts)], edgecolor='white', alpha=0.8)  
          
        # Customize  
        ax.set_xticks(range(len(value_counts)))  
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')  
        ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')  
        ax.set_xlabel(col, fontsize=10)  
        ax.set_ylabel('Count', fontsize=10)  
          
        # Add value labels on bars  
        for bar, val in zip(bars, value_counts.values):  
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,  
                   f'{val}', ha='center', va='bottom', fontsize=9)  
      
    # Hide empty subplots  
    for idx in range(len(plot_cols), len(axes)):  
        axes[idx].set_visible(False)  
      
    plt.suptitle('Distribution of Categorical Features', fontsize=16,   
                 fontweight='bold', y=1.02)  
    plt.tight_layout()  
    plt.savefig('categorical_distributions.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Plot categorical distributions  
plot_categorical_bars(df, categorical_cols)  
```  
  
#### 8.2.3 Pie Charts for Categorical Features  
  
```python  
# Create pie charts for categorical features  
def plot_categorical_pies(df, categorical_cols, max_categories=6):  
    """  
    Plot pie charts for categorical features with few categories.  
    """  
    # Filter columns with few categories  
    plot_cols = [col for col in categorical_cols if df[col].nunique() <= max_categories]  
      
    n_cols = 3  
    n_rows = (len(plot_cols) + n_cols - 1) // n_cols  
      
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))  
    axes = axes.flatten()  
      
    for idx, col in enumerate(plot_cols):  
        ax = axes[idx]  
          
        # Get value counts  
        value_counts = df[col].value_counts()  
          
        # Create pie chart  
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))  
        wedges, texts, autotexts = ax.pie(  
            value_counts.values,   
            labels=value_counts.index,  
            autopct='%1.1f%%',  
            colors=colors,  
            explode=[0.02] * len(value_counts),  
            shadow=True,  
            startangle=90  
        )  
          
        # Customize text  
        for autotext in autotexts:  
            autotext.set_fontsize(9)  
            autotext.set_fontweight('bold')  
          
        ax.set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')  
      
    # Hide empty subplots  
    for idx in range(len(plot_cols), len(axes)):  
        axes[idx].set_visible(False)  
      
    plt.suptitle('Pie Charts of Categorical Features', fontsize=16,   
                 fontweight='bold', y=1.02)  
    plt.tight_layout()  
    plt.savefig('categorical_piecharts.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Plot pie charts  
plot_categorical_pies(df, categorical_cols)  
```  
  
#### 8.2.4 Count Plots with Percentages  
  
```python  
#Create count plots with percentages  
def plot_countplots_with_percentages(df, categorical_cols, max_categories=10):  
    """  
    Plot count plots with percentage annotations.  
    """  
    plot_cols = [col for col in categorical_cols if df[col].nunique() <= max_categories]  
      
    n_cols = 2  
    n_rows = (len(plot_cols) + n_cols - 1) // n_cols  
      
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))  
    axes = axes.flatten()  
      
    for idx, col in enumerate(plot_cols):  
        ax = axes[idx]  
          
        # Create count plot  
        order = df[col].value_counts().index  
        sns.countplot(data=df, x=col, ax=ax, order=order, palette='viridis',  
                      edgecolor='black', alpha=0.8)  
          
        # Add percentage labels  
        total = len(df[col].dropna())  
        for p in ax.patches:  
            percentage = f'{100 * p.get_height() / total:.1f}%'  
            ax.annotate(percentage,   
                       (p.get_x() + p.get_width() / 2., p.get_height()),  
                       ha='center', va='bottom', fontsize=10, fontweight='bold')  
          
        ax.set_title(f'Count Plot of {col}', fontsize=12, fontweight='bold')  
        ax.set_xlabel(col, fontsize=10)  
        ax.set_ylabel('Count', fontsize=10)  
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')  
      
    # Hide empty subplots  
    for idx in range(len(plot_cols), len(axes)):  
        axes[idx].set_visible(False)  
      
    plt.suptitle('Count Plots with Percentages', fontsize=16,   
                 fontweight='bold', y=1.02)  
    plt.tight_layout()  
    plt.savefig('countplots_percentages.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Plot count plots  
plot_countplots_with_percentages(df, categorical_cols)  
```  
  
### 8.3 Summary Statistics Table  
  
```python  
# Create comprehensive summary table  
def create_summary_table(df):  
    """  
    Create a comprehensive summary table for all features.  
    """  
    summary_data = []  
      
    for col in df.columns:  
        col_data = df[col]  
          
        summary = {  
            'Column': col,  
            'Data Type': str(col_data.dtype),  
            'Non-Null Count': col_data.notnull().sum(),  
            'Null Count': col_data.isnull().sum(),  
            'Null %': round((col_data.isnull().sum() / len(df)) * 100, 2),  
            'Unique Values': col_data.nunique(),  
            'Unique %': round((col_data.nunique() / len(df)) * 100, 2)  
        }  
          
        # Add statistics for numerical columns  
        if col_data.dtype in ['int64', 'float64']:  
            summary['Mean'] = round(col_data.mean(), 2)  
            summary['Median'] = round(col_data.median(), 2)  
            summary['Std'] = round(col_data.std(), 2)  
            summary['Min'] = round(col_data.min(), 2)  
            summary['Max'] = round(col_data.max(), 2)  
        else:  
            summary['Mean'] = '-'  
            summary['Median'] = '-'  
            summary['Std'] = '-'  
            summary['Min'] = '-'  
            summary['Max'] = '-'  
          
        # Add mode for all columns  
        if len(col_data.mode()) > 0:  
            summary['Mode'] = col_data.mode().iloc[0]  
        else:  
            summary['Mode'] = '-'  
          
        summary_data.append(summary)  
      
    summary_df = pd.DataFrame(summary_data)  
    return summary_df  
  
# Create and display summary table  
summary_table = create_summary_table(df)  
print("📊 Comprehensive Feature Summary Table:")  
print("=" * 120)  
print(summary_table.to_string(index=False))  
```  
  
---  
  
## 9. Bivariate Analysis  
  
Bivariate analysis examines the relationship between two variables.  
  
### 9.1 Numerical vs Numerical  
  
#### 9.1.1 Scatter Plots  
  
```python  
# Create scatter plots for numerical pairs  
def plot_scatter_matrix(df, numerical_cols, target_col=None):  
    """  
    Create scatter plots for pairs of numerical features.  
    """  
    from itertools import combinations  
      
    pairs = list(combinations(numerical_cols, 2))  
      
    n_cols = 3  
    n_rows = (len(pairs) + n_cols - 1) // n_cols  
      
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))  
    axes = axes.flatten()  
      
    for idx, (col1, col2) in enumerate(pairs):  
        ax = axes[idx]  
          
        if target_col and target_col in df.columns:  
            # Color by target variable  
            scatter = ax.scatter(df[col1], df[col2], c=df[target_col],   
                               cmap='coolwarm', alpha=0.6, edgecolors='white', s=50)  
            plt.colorbar(scatter, ax=ax, label=target_col)  
        else:  
            ax.scatter(df[col1], df[col2], alpha=0.6, edgecolors='white',   
                      s=50, color='steelblue')  
          
        # Add regression line  
        mask = df[[col1, col2]].notna().all(axis=1)  
        if mask.sum() > 2:  
            z = np.polyfit(df.loc[mask, col1], df.loc[mask, col2], 1)  
            p = np.poly1d(z)  
            x_line = np.linspace(df[col1].min(), df[col1].max(), 100)  
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend')  
          
        # Calculate correlation  
        corr = df[[col1, col2]].corr().iloc[0, 1]  
          
        ax.set_xlabel(col1, fontsize=10)  
        ax.set_ylabel(col2, fontsize=10)  
        ax.set_title(f'{col1} vs {col2}\n(r = {corr:.3f})', fontsize=11, fontweight='bold')  
        ax.legend(loc='upper right', fontsize=8)  
      
    # Hide empty subplots  
    for idx in range(len(pairs), len(axes)):  
        axes[idx].set_visible(False)  
      
    plt.suptitle('Scatter Plots - Numerical Feature Pairs', fontsize=16,   
                 fontweight='bold', y=1.02)  
    plt.tight_layout()  
    plt.savefig('scatter_plots.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Plot scatter matrix  
plot_scatter_matrix(df, numerical_cols, target_col='survived')  
```  
  
#### 9.1.2 Pair Plot  
  
```python  
# Create pair plot  
def create_pairplot(df, numerical_cols, hue_col=None):  
    """  
    Create a pair plot for numerical features.  
    """  
    # Select subset of columns to avoid overcrowding  
    plot_cols = numerical_cols[:5]  # Limit to 5 columns  
      
    if hue_col and hue_col in df.columns:  
        plot_df = df[plot_cols + [hue_col]].dropna()  
        g = sns.pairplot(plot_df, hue=hue_col, palette='husl',  
                         diag_kind='kde', plot_kws={'alpha': 0.6, 's': 40},  
                         corner=False)  
    else:  
        plot_df = df[plot_cols].dropna()  
        g = sns.pairplot(plot_df, diag_kind='kde',   
                         plot_kws={'alpha': 0.6, 's': 40, 'color': 'steelblue'})  
      
    g.fig.suptitle('Pair Plot of Numerical Features', fontsize=16,   
                   fontweight='bold', y=1.02)  
      
    plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Create pair plot  
create_pairplot(df, numerical_cols, hue_col='survived')  
```  
  
#### 9.1.3 Correlation Heatmap  
  
```python  
# Create correlation heatmap  
def plot_correlation_heatmap(df, numerical_cols, method='pearson'):  
    """  
    Create a correlation heatmap for numerical features.  
      
    Parameters:  
    -----------  
    df : pandas DataFrame  
    numerical_cols : list of numerical column names  
    method : str - 'pearson', 'spearman', or 'kendall'  
    """  
    # Calculate correlation matrix  
    corr_matrix = df[numerical_cols].corr(method=method)  
      
    # Create mask for upper triangle  
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  
      
    # Set up figure  
    fig, ax = plt.subplots(figsize=(12, 10))  
      
    # Create heatmap  
    cmap = sns.diverging_palette(250, 10, as_cmap=True)  
      
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1,  
                center=0, annot=True, fmt='.2f', square=True,  
                linewidths=0.5, cbar_kws={'shrink': 0.8, 'label': 'Correlation'},  
                ax=ax, annot_kws={'size': 10})  
      
    ax.set_title(f'Correlation Heatmap ({method.capitalize()} Method)',   
                 fontsize=16, fontweight='bold', pad=20)  
      
    plt.tight_layout()  
    plt.savefig(f'correlation_heatmap_{method}.png', dpi=300, bbox_inches='tight')  
    plt.show()  
      
    return corr_matrix  
  
# Create correlation heatmaps  
corr_pearson = plot_correlation_heatmap(df, numerical_cols, method='pearson')  
corr_spearman = plot_correlation_heatmap(df, numerical_cols, method='spearman')  
```  
  
#### 9.1.4 Correlation Analysis Summary  
  
```python  
# Analyze and summarize correlations  
def analyze_correlations(corr_matrix, threshold=0.5):  
    """  
    Analyze correlations and identify strong relationships.  
      
    Parameters:  
    -----------  
    corr_matrix : pandas DataFrame - correlation matrix  
    threshold : float - threshold for strong correlation  
    """  
    print("📊 Correlation Analysis Summary")  
    print("=" * 60)  
      
    # Get upper triangle pairs  
    strong_correlations = []  
      
    for i in range(len(corr_matrix.columns)):  
        for j in range(i+1, len(corr_matrix.columns)):  
            col1 = corr_matrix.columns[i]  
            col2 = corr_matrix.columns[j]  
            corr_value = corr_matrix.iloc[i, j]  
              
            if abs(corr_value) >= threshold:  
                strong_correlations.append({  
                    'Feature 1': col1,  
                    'Feature 2': col2,  
                    'Correlation': corr_value,  
                    'Strength': 'Strong Positive' if corr_value > 0 else 'Strong Negative'  
                })  
      
    if strong_correlations:  
        strong_df = pd.DataFrame(strong_correlations)  
        strong_df = strong_df.sort_values('Correlation', key=abs, ascending=False)  
          
        print(f"\n🔗 Strong Correlations (|r| >= {threshold}):")  
        print(strong_df.to_string(index=False))  
    else:  
        print(f"\n✅ No strong correlations found (|r| >= {threshold})")  
      
    # Print interpretation guide  
    print("\n📖 Correlation Interpretation Guide:")  
    print("   |r| < 0.3  : Weak correlation")  
    print("   0.3 <= |r| < 0.5 : Moderate correlation")  
    print("   0.5 <= |r| < 0.7 : Strong correlation")  
    print("   |r| >= 0.7 : Very strong correlation")  
      
    return strong_correlations  
  
# Analyze correlations  
strong_corrs = analyze_correlations(corr_pearson, threshold=0.3)  
```  
  
### 9.2 Categorical vs Numerical  
  
#### 9.2.1 Box Plots by Category  
  
```python  
# Create box plots for numerical features grouped by categorical features  
def plot_boxplots_by_category(df, numerical_col, categorical_cols, max_categories=6):  
    """  
    Create box plots showing distribution of numerical feature across categories.  
    """  
    # Filter categorical columns with reasonable number of categories  
    plot_cats = [col for col in categorical_cols if df[col].nunique() <= max_categories]  
      
    n_cols = 2  
    n_rows = (len(plot_cats) + n_cols - 1) // n_cols  
      
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))  
    axes = axes.flatten()  
      
    for idx, cat_col in enumerate(plot_cats):  
        ax = axes[idx]  
          
        # Create box plot  
        order = df[cat_col].value_counts().index  
        sns.boxplot(data=df, x=cat_col, y=numerical_col, ax=ax,   
                   order=order, palette='viridis')  
          
        # Add individual points  
        sns.stripplot(data=df, x=cat_col, y=numerical_col, ax=ax,  
                     order=order, color='red', alpha=0.3, size=3)  
          
        ax.set_title(f'{numerical_col} by {cat_col}', fontsize=12, fontweight='bold')  
        ax.set_xlabel(cat_col, fontsize=10)  
        ax.set_ylabel(numerical_col, fontsize=10)  
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')  
      
    # Hide empty subplots  
    for idx in range(len(plot_cats), len(axes)):  
        axes[idx].set_visible(False)  
      
    plt.suptitle(f'Distribution of {numerical_col} Across Categories',   
                 fontsize=16, fontweight='bold', y=1.02)  
    plt.tight_layout()  
    plt.savefig(f'boxplots_{numerical_col}_by_category.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Example: Age distribution by different categories  
plot_boxplots_by_category(df, 'age', categorical_cols)  
plot_boxplots_by_category(df, 'fare', categorical_cols)  
```  
  
#### 9.2.2 Violin Plots by Category  
  
```python  
# Create violin plots by category  
def plot_violins_by_category(df, numerical_col, categorical_cols, max_categories=6):  
    """  
    Create violin plots showing distribution of numerical feature across categories.  
    """  
    plot_cats = [col for col in categorical_cols if df[col].nunique() <= max_categories]  
      
    n_cols = 2  
    n_rows = (len(plot_cats) + n_cols - 1) // n_colsfig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))  
    axes = axes.flatten()  
      
    for idx, cat_col in enumerate(plot_cats):  
        ax = axes[idx]  
          
        # Create violin plot  
        order = df[cat_col].value_counts().index  
        sns.violinplot(data=df, x=cat_col, y=numerical_col, ax=ax,  
                      order=order, palette='muted', inner='box')  
          
        ax.set_title(f'{numerical_col} Distribution by {cat_col}',   
                    fontsize=12, fontweight='bold')  
        ax.set_xlabel(cat_col, fontsize=10)  
        ax.set_ylabel(numerical_col, fontsize=10)  
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')  
          
        # Add mean markers  
        means = df.groupby(cat_col)[numerical_col].mean()  
        for i, cat in enumerate(order):  
            if cat in means.index:  
                ax.scatter(i, means[cat], color='white', s=50, zorder=3,   
                          edgecolors='black', linewidth=2, marker='D')  
      
    # Hide empty subplots  
    for idx in range(len(plot_cats), len(axes)):  
        axes[idx].set_visible(False)  
      
    plt.suptitle(f'Violin Plots of {numerical_col} by Categories',   
                 fontsize=16, fontweight='bold', y=1.02)  
    plt.tight_layout()  
    plt.savefig(f'violinplots_{numerical_col}_by_category.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Plot violin plots  
plot_violins_by_category(df, 'age', categorical_cols)  
```  
  
#### 9.2.3 Statistical Tests (ANOVA, t-test)  
  
```python  
# Perform statistical tests for categorical vs numerical relationships  
def statistical_tests_cat_num(df, numerical_col, categorical_col):  
    """  
    Perform statistical tests to check if numerical variable differs  
    significantly across categories.  
      
    Parameters:  
    -----------  
    df : pandas DataFrame  
    numerical_col : str - name of numerical column  
    categorical_col : str - name of categorical column  
      
    Returns:  
    --------  
    dict : Test results  
    """  
    from scipy import stats  
      
    # Remove missing values  
    data = df[[numerical_col, categorical_col]].dropna()  
      
    # Get unique categories  
    categories = data[categorical_col].unique()  
    n_categories = len(categories)  
      
    # Group data by category  
    groups = [data[data[categorical_col] == cat][numerical_col].values   
              for cat in categories]  
      
    results = {  
        'Numerical Variable': numerical_col,  
        'Categorical Variable': categorical_col,  
        'Number of Categories': n_categories  
    }  
      
    if n_categories == 2:  
        # Perform independent t-test for 2 groups  
        stat, p_value = stats.ttest_ind(groups[0], groups[1])  
        results['Test'] = 'Independent t-test'  
        results['Statistic'] = stat  
        results['P-value'] = p_value  
          
        # Also perform Mann-Whitney U test (non-parametric alternative)  
        stat_mw, p_value_mw = stats.mannwhitneyu(groups[0], groups[1],   
                                                  alternative='two-sided')  
        results['Mann-Whitney U Statistic'] = stat_mw  
        results['Mann-Whitney P-value'] = p_value_mw  
          
    else:  
        # Perform one-way ANOVA for 3+ groups  
        stat, p_value = stats.f_oneway(*groups)  
        results['Test'] = 'One-way ANOVA'  
        results['F-Statistic'] = stat  
        results['P-value'] = p_value  
          
        # Also perform Kruskal-Wallis test (non-parametric alternative)  
        stat_kw, p_value_kw = stats.kruskal(*groups)  
        results['Kruskal-Wallis Statistic'] = stat_kw  
        results['Kruskal-Wallis P-value'] = p_value_kw  
      
    # Interpretation  
    alpha = 0.05  
    if p_value < alpha:  
        results['Conclusion'] = f'Significant difference exists (p < {alpha})'  
    else:  
        results['Conclusion'] = f'No significant difference (p >= {alpha})'  
      
    return results  
  
# Perform statistical tests  
def run_all_statistical_tests(df, numerical_cols, categorical_cols, max_categories=10):  
    """  
    Run statistical tests for all numerical-categorical pairs.  
    """  
    all_results = []  
      
    for num_col in numerical_cols:  
        for cat_col in categorical_cols:  
            if df[cat_col].nunique() <= max_categories and df[cat_col].nunique() >= 2:  
                try:  
                    result = statistical_tests_cat_num(df, num_col, cat_col)  
                    all_results.append(result)  
                except Exception as e:  
                    print(f"Error testing {num_col} vs {cat_col}: {e}")  
      
    return pd.DataFrame(all_results)  
  
# Run tests  
test_results = run_all_statistical_tests(df, numerical_cols, categorical_cols)  
  
print("📊 Statistical Tests Results (Categorical vs Numerical)")  
print("=" * 80)  
print(test_results[['Numerical Variable', 'Categorical Variable', 'Test',   
                    'P-value', 'Conclusion']].to_string(index=False))  
```  
  
### Expected Output (Sample)  
  
```  
📊 Statistical Tests Results (Categorical vs Numerical)  
================================================================================  
Numerical Variable Categorical Variable           Test    P-value                        Conclusion  
               age                 sex Independent t-test   0.0421  Significant difference exists (p < 0.05)  
               age            survived Independent t-test   0.0000  Significant difference exists (p < 0.05)  
               age              pclass   One-way ANOVA      0.0000  Significant difference exists (p < 0.05)  
              fare                 sex Independent t-test   0.0001  Significant difference exists (p < 0.05)  
              fare            survived Independent t-test   0.0000  Significant difference exists (p < 0.05)  
```  
  
#### 9.2.4 Group Statistics Summary  
  
```python  
# Calculate group statistics  
def group_statistics(df, numerical_col, categorical_col):  
    """  
    Calculate statistics for numerical variable grouped by categorical variable.  
    """  
    stats_df = df.groupby(categorical_col)[numerical_col].agg([  
        ('Count', 'count'),  
        ('Mean', 'mean'),  
        ('Median', 'median'),  
        ('Std', 'std'),  
        ('Min', 'min'),  
        ('Max', 'max'),  
        ('Q1', lambda x: x.quantile(0.25)),  
        ('Q3', lambda x: x.quantile(0.75))  
    ]).round(2)  
      
    stats_df['IQR'] = stats_df['Q3'] - stats_df['Q1']  
      
    print(f"\n📊 Statistics of {numerical_col} by {categorical_col}")  
    print("=" * 80)  
    print(stats_df.to_string())  
      
    return stats_df  
  
# Example: Calculate group statistics  
group_statistics(df, 'age', 'pclass')  
group_statistics(df, 'fare', 'survived')  
group_statistics(df, 'age', 'sex')  
```  
  
### 9.3 Categorical vs Categorical  
  
#### 9.3.1 Cross-Tabulation (Contingency Tables)  
  
```python  
# Create cross-tabulation  
def create_crosstab(df, cat1, cat2, normalize=False):  
    """  
    Create and visualize cross-tabulation between twocategorical variables.  
      
    Parameters:  
    -----------  
    df : pandas DataFrame  
    cat1 : str - first categorical column  
    cat2 : str - second categorical column  
    normalize : bool or str - False, 'index', 'columns', or 'all'  
    """  
    # Create cross-tabulation  
    crosstab = pd.crosstab(df[cat1], df[cat2], margins=True, margins_name='Total')  
      
    print(f"\n📊 Cross-Tabulation: {cat1} vs {cat2}")  
    print("=" * 60)  
    print(crosstab)  
      
    if normalize:  
        crosstab_norm = pd.crosstab(df[cat1], df[cat2], normalize=normalize) * 100  
        print(f"\n📊 Normalized Cross-Tabulation (% by {normalize}):")  
        print(crosstab_norm.round(2))  
      
    return crosstab  
  
# Create cross-tabulations  
crosstab_sex_survived = create_crosstab(df, 'sex', 'survived', normalize='index')  
crosstab_pclass_survived = create_crosstab(df, 'pclass', 'survived', normalize='index')  
```  
  
### Expected Output (Sample)  
  
```  
📊 Cross-Tabulation: sex vs survived  
============================================================  
survived    0    1  Total  
sex                        
female     81  233    314  
male      468  109    577  
Total     549  342    891  
  
📊 Normalized Cross-Tabulation (% by index):  
survived      0      1  
sex                      
female    25.80  74.20  
male      81.11  18.89  
```  
  
#### 9.3.2 Stacked Bar Charts  
  
```python  
# Create stacked bar charts for categorical pairs  
def plot_stacked_bar(df, cat1, cat2, normalize=True):  
    """  
    Create stacked bar chart for two categorical variables.  
    """  
    # Create cross-tabulation  
    if normalize:  
        crosstab = pd.crosstab(df[cat1], df[cat2], normalize='index') * 100  
        ylabel = 'Percentage (%)'  
    else:  
        crosstab = pd.crosstab(df[cat1], df[cat2])  
        ylabel = 'Count'  
      
    # Plot  
    fig, ax = plt.subplots(figsize=(10, 6))  
      
    crosstab.plot(kind='bar', stacked=True, ax=ax, colormap='viridis',  
                  edgecolor='white', width=0.7)  
      
    ax.set_title(f'{cat1} vs {cat2} (Stacked Bar Chart)',   
                fontsize=14, fontweight='bold')  
    ax.set_xlabel(cat1, fontsize=12)  
    ax.set_ylabel(ylabel, fontsize=12)  
    ax.legend(title=cat2, bbox_to_anchor=(1.02, 1), loc='upper left')  
      
    # Rotate x-axis labels  
    plt.xticks(rotation=45, ha='right')  
      
    # Add percentage labels on bars  
    if normalize:  
        for c in ax.containers:  
            labels = [f'{v:.1f}%' if v > 5 else '' for v in c.datavalues]  
            ax.bar_label(c, labels=labels, label_type='center', fontsize=9)  
      
    plt.tight_layout()  
    plt.savefig(f'stacked_bar_{cat1}_{cat2}.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Create stacked bar charts  
plot_stacked_bar(df, 'pclass', 'survived')  
plot_stacked_bar(df, 'sex', 'survived')  
plot_stacked_bar(df, 'embarked', 'survived')  
```  
  
#### 9.3.3 Grouped Bar Charts  
  
```python  
# Create grouped bar charts  
def plot_grouped_bar(df, cat1, cat2):  
    """  
    Create grouped bar chart for two categorical variables.  
    """  
    # Create cross-tabulation  
    crosstab = pd.crosstab(df[cat1], df[cat2])  
      
    # Plot  
    fig, ax = plt.subplots(figsize=(10, 6))  
      
    x = np.arange(len(crosstab.index))  
    width = 0.8 / len(crosstab.columns)  
      
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(crosstab.columns)))  
      
    for i, col in enumerate(crosstab.columns):  
        offset = (i - len(crosstab.columns)/2 + 0.5) * width  
        bars = ax.bar(x + offset, crosstab[col], width, label=col,   
                     color=colors[i], edgecolor='white')  
          
        # Add value labels  
        ax.bar_label(bars, padding=3, fontsize=9)  
      
    ax.set_title(f'{cat1} vs {cat2} (Grouped Bar Chart)',   
                fontsize=14, fontweight='bold')  
    ax.set_xlabel(cat1, fontsize=12)  
    ax.set_ylabel('Count', fontsize=12)  
    ax.set_xticks(x)  
    ax.set_xticklabels(crosstab.index, rotation=45, ha='right')  
    ax.legend(title=cat2, bbox_to_anchor=(1.02, 1), loc='upper left')  
      
    plt.tight_layout()  
    plt.savefig(f'grouped_bar_{cat1}_{cat2}.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Create grouped bar charts  
plot_grouped_bar(df, 'pclass', 'survived')  
plot_grouped_bar(df, 'sex', 'survived')  
```  
  
#### 9.3.4 Heatmap of Cross-Tabulation  
  
```python  
# Create heatmap of cross-tabulation  
def plot_crosstab_heatmap(df, cat1, cat2, normalize='index'):  
    """  
    Create heatmap visualization of cross-tabulation.  
    """  
    # Create normalized cross-tabulation  
    if normalize:  
        crosstab = pd.crosstab(df[cat1], df[cat2], normalize=normalize) * 100  
        fmt = '.1f'  
        title_suffix = f'(% by {normalize})'  
    else:  
        crosstab = pd.crosstab(df[cat1], df[cat2])  
        fmt = 'd'  
        title_suffix = '(counts)'  
      
    # Plot heatmap  
    fig, ax = plt.subplots(figsize=(10, 8))  
      
    sns.heatmap(crosstab, annot=True, fmt=fmt, cmap='YlOrRd',  
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Percentage' if normalize else 'Count'})  
      
    ax.set_title(f'Cross-Tabulation Heatmap: {cat1} vs {cat2}\n{title_suffix}',   
                fontsize=14, fontweight='bold')  
    ax.set_xlabel(cat2, fontsize=12)  
    ax.set_ylabel(cat1, fontsize=12)  
      
    plt.tight_layout()  
    plt.savefig(f'crosstab_heatmap_{cat1}_{cat2}.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Create heatmaps  
plot_crosstab_heatmap(df, 'pclass', 'survived')  
plot_crosstab_heatmap(df, 'sex', 'embarked', normalize='columns')  
```  
  
#### 9.3.5 Chi-Square Test of Independence  
  
```python  
# Perform Chi-Square test  
def chi_square_test(df, cat1, cat2):  
    """  
    Perform Chi-Square test of independence between two categorical variables.  
      
    Parameters:  
    -----------  
    df : pandas DataFrame  
    cat1 : str - first categorical column  
    cat2 : str - second categorical column  
      
    Returns:  
    --------  
    dict : Test results including chi-square statistic, p-value, and interpretation  
    """  
    from scipy import stats  
      
    # Create contingency table  
    contingency_table = pd.crosstab(df[cat1], df[cat2])  
      
    # Perform chi-square test  
    chi2, p_value, dof, expected_freq = stats.chi2_contingency(contingency_table)  
      
    # Calculate Cramér's V (effect size)  
    n = contingency_table.sum().sum()  
    min_dim = min(contingency_table.shape) - 1  
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0  
      
    # Interpretation of Cramér's V  
    if cramers_v < 0.1:  
        effect_size = "Negligible"  
    elif cramers_v < 0.2:  
        effect_size = "Weak"  
    elif cramers_v < 0.4:  
        effect_size = "Moderate"  
    elif cramers_v < 0.6:  
        effect_size = "Relatively Strong"  
    else:  
        effect_size = "Strong"  
      
    # Results  
    results = {  
        'Variable 1': cat1,  
        'Variable 2': cat2,  
        'Chi-Square Statistic': round(chi2, 4),  
        'Degrees of Freedom': dof,  
        'P-value': p_value,  
        'Cramér\'s V': round(cramers_v, 4),  
        'Effect Size': effect_size,  
        'Significant (α=0.05)': 'Yes' if p_value < 0.05 else 'No'  
    }  
      
    # Print results  
    print(f"\n📊 Chi-Square Test: {cat1} vs {cat2}")  
    print("=" * 60)  
    print(f"Chi-Square Statistic: {chi2:.4f}")  
    print(f"Degrees of Freedom: {dof}")  
    print(f"P-value: {p_value:.6f}")  
    print(f"Cramér's V: {cramers_v:.4f} ({effect_size} association)")  
      
    if p_value < 0.05:  
        print(f"\n✅ Result: There IS a significant association between {cat1} and {cat2}")  
    else:  
        print(f"\n❌ Result: There is NO significant association between {cat1} and {cat2}")  
      
    # Check expected frequency assumption  
    expected_df = pd.DataFrame(expected_freq,   
                               index=contingency_table.index,  
                               columns=contingency_table.columns)  
      
    cells_below_5 = (expected_df < 5).sum().sum()  
    total_cells = expected_df.size  
    pct_below_5 = (cells_below_5 / total_cells) * 100  
      
    print(f"\n⚠️ Assumption Check: {cells_below_5}/{total_cells} ({pct_below_5:.1f}%) cells have expected frequency < 5")  
    if pct_below_5 > 20:  
        print("   Warning: Chi-square test may not be reliable. Consider Fisher's exact test.")  
      
    return results, expected_df  
  
# Run chi-square tests for all categorical pairs  
def run_all_chi_square_tests(df, categorical_cols, max_categories=15):  
    """  
    Run chi-square tests for all pairs of categorical variables.  
    """  
    from itertools import combinations  
      
    # Filter columns with reasonable number of categories  
    valid_cols = [col for col in categorical_cols   
                  if 2 <= df[col].nunique() <= max_categories]  
      
    all_results = []  
      
    for cat1, cat2 in combinations(valid_cols, 2):  
        try:  
            result, _ = chi_square_test(df, cat1, cat2)  
            all_results.append(result)  
        except Exception as e:  
            print(f"Error testing {cat1} vs {cat2}: {e}")  
      
    results_df = pd.DataFrame(all_results)  
      
    # Summary table  
    print("\n" + "=" * 80)  
    print("📊 CHI-SQUARE TEST SUMMARY")  
    print("=" * 80)  
    print(results_df[['Variable 1', 'Variable 2', 'Chi-Square Statistic',   
                      'P-value', 'Cramér\'s V', 'Effect Size', 'Significant (α=0.05)']].to_string(index=False))  
      
    return results_df  
  
# Run all chi-square tests  
chi_square_results = run_all_chi_square_tests(df, categorical_cols)  
```  
  
### Expected Output (Sample)  
  
```  
📊 Chi-Square Test: sex vs survived  
============================================================  
Chi-Square Statistic: 260.7170  
Degrees of Freedom: 1  
P-value: 0.000000  
Cramér's V: 0.5434 (Relatively Strong association)  
  
✅ Result: There IS a significant association between sex and survived  
  
⚠️ Assumption Check: 0/4 (0.0%) cells have expected frequency < 5  
  
================================================================================  
📊 CHI-SQUARE TEST SUMMARY  
================================================================================  
Variable 1 Variable 2  Chi-Square Statistic    P-value  Cramér's V       Effect Size Significant (α=0.05)  
       sex   survived               260.7170   0.000000      0.5434  Relatively Strong                  Yes  
    pclass   survived               102.8890   0.000000      0.3397          Moderate                  Yes  
  embarked   survived                26.4690   0.000002      0.1726              Weak                  Yes  
       sex     pclass                 0.1272   0.938420      0.0119        Negligible                   No  
```  
  
#### 9.3.6 Mosaic Plot  
  
```python  
# Create mosaic plot  
def plot_mosaic(df, cat1, cat2):  
    """  
    Create mosaic plot for two categorical variables.  
    """  
    from statsmodels.graphics.mosaicplot import mosaic  
      
    fig, ax = plt.subplots(figsize=(12, 8))  
      
    # Prepare data  
    data = df[[cat1, cat2]].dropna()  
      
    # Create mosaic plot  
    mosaic(data, [cat1, cat2], ax=ax,   
           properties=lambda key: {'color': plt.cm.viridis(hash(key) % 256 / 256)},  
           labelizer=lambda k: f'{k}',  
           title=f'Mosaic Plot: {cat1} vs {cat2}')  
      
    plt.tight_layout()  
    plt.savefig(f'mosaic_{cat1}_{cat2}.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Create mosaic plots  
try:  
    plot_mosaic(df, 'pclass', 'survived')  
    plot_mosaic(df, 'sex', 'survived')  
except ImportError:  
    print("Mosaic plot requires statsmodels. Install with: pip install statsmodels")  
```  
  
---  
  
## 10. Multivariate Analysis  
  
Multivariate analysis examines relationships among three or more variables simultaneously.  
  
### 10.1 Pair Plots with Hue  
  
```python  
# Create enhanced pair plot with multiple dimensions  
def create_enhanced_pairplot(df, numerical_cols, hue_col, style_col=None):  
    """  
    Create enhanced pair plot with hue and optional style differentiation.  
    """  
    # Select subset of columns  
    plot_cols = numerical_cols[:4]  # Limit for readability  
      
    plot_df = df[plot_cols + [hue_col]].dropna()  
    if style_col:  
        plot_df = df[plot_cols + [hue_col, style_col]].dropna()  
      
    g = sns.pairplot(plot_df, hue=hue_col,diag_kind='kde',  
                     plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'white'},  
                     palette='husl',  
                     corner=False)  
      
    g.fig.suptitle(f'Pair Plot Colored by {hue_col}', fontsize=16,   
                   fontweight='bold', y=1.02)  
      
    plt.savefig(f'enhanced_pairplot_{hue_col}.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Create enhanced pair plots  
create_enhanced_pairplot(df, numerical_cols, 'survived')  
create_enhanced_pairplot(df, numerical_cols, 'pclass')  
```  
  
### 10.2 3D Scatter Plots  
  
```python  
# Create 3D scatter plot  
def plot_3d_scatter(df, x_col, y_col, z_col, color_col=None):  
    """  
    Create 3D scatter plot for three numerical variables.  
      
    Parameters:  
    -----------  
    df : pandas DataFrame  
    x_col, y_col, z_col : str - column names for x, y, z axes  
    color_col : str - optional column for color coding  
    """  
    from mpl_toolkits.mplot3d import Axes3D  
      
    fig = plt.figure(figsize=(12, 9))  
    ax = fig.add_subplot(111, projection='3d')  
      
    # Prepare data  
    plot_df = df[[x_col, y_col, z_col]].dropna()  
    if color_col:  
        plot_df = df[[x_col, y_col, z_col, color_col]].dropna()  
      
    # Create scatter plot  
    if color_col:  
        scatter = ax.scatter(plot_df[x_col], plot_df[y_col], plot_df[z_col],  
                            c=plot_df[color_col], cmap='viridis',   
                            s=50, alpha=0.7, edgecolors='white')  
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)  
        cbar.set_label(color_col, fontsize=11)  
    else:  
        ax.scatter(plot_df[x_col], plot_df[y_col], plot_df[z_col],  
                  c='steelblue', s=50, alpha=0.7, edgecolors='white')  
      
    ax.set_xlabel(x_col, fontsize=11, labelpad=10)  
    ax.set_ylabel(y_col, fontsize=11, labelpad=10)  
    ax.set_zlabel(z_col, fontsize=11, labelpad=10)  
    ax.set_title(f'3D Scatter Plot: {x_col} vs {y_col} vs {z_col}',   
                fontsize=14, fontweight='bold', pad=20)  
      
    # Adjust viewing angle  
    ax.view_init(elev=20, azim=45)  
      
    plt.tight_layout()  
    plt.savefig(f'3d_scatter_{x_col}_{y_col}_{z_col}.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Create 3D scatter plots  
plot_3d_scatter(df, 'age', 'fare', 'pclass', color_col='survived')  
```  
  
### 10.3 Faceted Plots (FacetGrid)  
  
```python  
# Create faceted plots  
def create_facet_plots(df, x_col, y_col, row_col=None, col_col=None, hue_col=None):  
    """  
    Create faceted scatter plots using seaborn FacetGrid.  
      
    Parameters:  
    -----------  
    df : pandas DataFrame  
    x_col : str - x-axis variable  
    y_col : str - y-axis variable  
    row_col : str - variable for row faceting  
    col_col : str - variable for column faceting  
    hue_col : str - variable for color coding  
    """  
    # Create FacetGrid  
    g = sns.FacetGrid(df, row=row_col, col=col_col, hue=hue_col,  
                      height=4, aspect=1.2, palette='husl',  
                      margin_titles=True)  
      
    # Map scatter plot  
    g.map_dataframe(sns.scatterplot, x=x_col, y=y_col, alpha=0.6, s=50)  
      
    # Add regression line  
    g.map_dataframe(sns.regplot, x=x_col, y=y_col, scatter=False,   
                    line_kws={'color': 'red', 'linestyle': '--', 'alpha': 0.8})  
      
    g.add_legend()  
    g.fig.suptitle(f'{y_col} vs {x_col} (Faceted)', fontsize=16,   
                   fontweight='bold', y=1.02)  
      
    plt.savefig(f'facet_plot_{x_col}_{y_col}.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Create faceted plots  
create_facet_plots(df, 'age', 'fare', col_col='pclass', hue_col='survived')  
create_facet_plots(df, 'age', 'fare', row_col='sex', col_col='pclass')  
```  
  
### 10.4 Faceted Histograms and KDE  
  
```python  
# Create faceted histograms  
def create_facet_histograms(df, value_col, facet_col, hue_col=None):  
    """  
    Create faceted histograms with KDE.  
    """  
    g = sns.FacetGrid(df, col=facet_col, hue=hue_col,   
                      height=4, aspect=1.2, palette='husl',  
                      col_wrap=3)  
      
    g.map_dataframe(sns.histplot, x=value_col, kde=True, alpha=0.6,  
                    stat='density', common_norm=False)  
      
    g.add_legend()  
    g.fig.suptitle(f'Distribution of {value_col} by {facet_col}',   
                   fontsize=16, fontweight='bold', y=1.02)  
      
    plt.savefig(f'facet_hist_{value_col}_{facet_col}.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Create faceted histograms  
create_facet_histograms(df, 'age', 'pclass', hue_col='survived')  
create_facet_histograms(df, 'fare', 'embarked', hue_col='survived')  
```  
  
### 10.5 Parallel Coordinates Plot  
  
```python  
# Create parallel coordinates plot  
def plot_parallel_coordinates(df, numerical_cols, class_col, sample_size=500):  
    """  
    Create parallel coordinates plot for multivariate analysis.  
      
    Parameters:  
    -----------  
    df : pandas DataFrame  
    numerical_cols : list - numerical columns to include  
    class_col : str - categorical column for coloring  
    sample_size : int - number of samples to plot (for performance)  
    """  
    from pandas.plotting import parallel_coordinates  
      
    # Prepare data  
    plot_cols = numerical_cols[:6]  # Limit columns for readability  
    plot_df = df[plot_cols + [class_col]].dropna()  
      
    # Sample if too large  
    if len(plot_df) > sample_size:  
        plot_df = plot_df.sample(n=sample_size, random_state=42)  
      
    # Normalize data for better visualization  
    plot_df_normalized = plot_df.copy()  
    for col in plot_cols:  
        min_val = plot_df_normalized[col].min()  
        max_val = plot_df_normalized[col].max()  
        if max_val > min_val:  
            plot_df_normalized[col] = (plot_df_normalized[col] - min_val) / (max_val - min_val)  
      
    # Create plot  
    fig, ax = plt.subplots(figsize=(14, 8))  
      
    parallel_coordinates(plot_df_normalized, class_col, ax=ax,   
                        colormap='viridis', alpha=0.5, linewidth=1.5)  
      
    ax.set_title(f'Parallel Coordinates Plot (Colored by {class_col})',   
                fontsize=14, fontweight='bold')  
    ax.set_ylabel('Normalized Value', fontsize=12)  
    ax.legend(loc='upper right', title=class_col)  
      
    # Rotate x-axis labels  
    plt.xticks(rotation=45, ha='right')  
      
    plt.tight_layout()  
    plt.savefig(f'parallel_coordinates_{class_col}.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Create parallel coordinates plot  
plot_parallel_coordinates(df, numerical_cols, 'survived')  
plot_parallel_coordinates(df, numerical_cols, 'pclass')  
```  
  
### 10.6 Radar/Spider Charts  
  
```python  
# Create radar chart  
def plot_radar_chart(df, numerical_cols, group_col):  
    """  
    Create radar chart comparing groups across multiple numerical variables.  
      
    Parameters:  
    -----------  
    df : pandas DataFrame  
    numerical_cols : list - numerical columns to include  
    group_col : str - categorical column for grouping  
    """  
    # Calculate mean values for each group  
    group_means = df.groupby(group_col)[numerical_cols].mean()  
      
    # Normalize values (0-1 scale)  
    group_means_norm = (group_means - group_means.min()) / (group_means.max() - group_means.min())  
      
    # Number of variables  
    categories = numerical_cols  
    N = len(categories)  
      
    # Create angles for radar chart  
    angles = [n / float(N) * 2 * np.pi for n in range(N)]  
    angles += angles[:1]  # Complete the loop  
      
    # Create figure  
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))  
      
    # Plot each group  
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(group_means_norm)))  
      
    for idx, (group, values) in enumerate(group_means_norm.iterrows()):  
        values_list = values.tolist()  
        values_list += values_list[:1]  # Complete the loop  
          
        ax.plot(angles, values_list, 'o-', linewidth=2,   
               label=f'{group_col}={group}', color=colors[idx])  
        ax.fill(angles, values_list, alpha=0.25, color=colors[idx])  
      
    # Set category labels  
    ax.set_xticks(angles[:-1])  
    ax.set_xticklabels(categories, fontsize=10)  
      
    ax.set_title(f'Radar Chart: Comparison by {group_col}',   
                fontsize=14, fontweight='bold', pad=20)  
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))  
      
    plt.tight_layout()  
    plt.savefig(f'radar_chart_{group_col}.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Create radar charts  
plot_radar_chart(df, numerical_cols[:5], 'survived')  
plot_radar_chart(df, numerical_cols[:5], 'pclass')  
```  
  
### 10.7 Bubble Charts  
  
```python  
# Create bubble chart  
def plot_bubble_chart(df, x_col, y_col, size_col, color_col=None):  
    """  
    Create bubble chart with four dimensions.  
      
    Parameters:  
    -----------  
    df : pandas DataFrame  
    x_col : str - x-axis variable  
    y_col : str - y-axis variable  
    size_col : str - variable for bubble size  
    color_col : str - variable for color coding  
    """  
    fig, ax = plt.subplots(figsize=(12, 8))  
      
    # Prepare data  
    plot_df = df[[x_col, y_col, size_col]].dropna()  
    if color_col:  
        plot_df = df[[x_col, y_col, size_col, color_col]].dropna()  
      
    # Normalize size for better visualization  
    size_normalized = (plot_df[size_col] - plot_df[size_col].min()) / \  
                      (plot_df[size_col].max() - plot_df[size_col].min())  
    sizes = 50 + size_normalized * 500  # Scale between 50 and 550  
      
    # Create scatter plot  
    if color_col:  
        scatter = ax.scatter(plot_df[x_col], plot_df[y_col], s=sizes,  
                            c=plot_df[color_col], cmap='viridis',  
                            alpha=0.6, edgecolors='white', linewidth=0.5)  
        cbar = plt.colorbar(scatter, ax=ax)  
        cbar.set_label(color_col, fontsize=11)  
    else:  
        ax.scatter(plot_df[x_col], plot_df[y_col], s=sizes,  
                  c='steelblue', alpha=0.6, edgecolors='white', linewidth=0.5)  
      
    ax.set_xlabel(x_col, fontsize=12)  
    ax.set_ylabel(y_col, fontsize=12)  
    ax.set_title(f'Bubble Chart: {x_col} vs {y_col}\n(Size: {size_col})',   
                fontsize=14, fontweight='bold')  
      
    # Add size legend  
    for size_val in [plot_df[size_col].min(), plot_df[size_col].median(), plot_df[size_col].max()]:  
        size_norm = (size_val - plot_df[size_col].min()) / \  
                   (plot_df[size_col].max() - plot_df[size_col].min())  
        ax.scatter([], [], s=50 + size_norm * 500, c='gray', alpha=0.6,  
                  label=f'{size_col}={size_val:.1f}')  
    ax.legend(scatterpoints=1, frameon=True, labelspacing=1.5,   
             title='Bubble Size', loc='upper left')  
      
    plt.tight_layout()  
    plt.savefig(f'bubble_chart_{x_col}_{y_col}_{size_col}.png', dpi=300, bbox_inches='tight')  
    plt.show()  
  
# Create bubble charts  
plot_bubble_chart(df, 'age', 'fare', 'pclass', color_col='survived')  
```  
  
---  
  
## 11. Target Variable Analysis (For Supervised Learning)  
  
This section focuses on analyzing the target variable and its relationship with features.  
  
### 11.1 Target Distribution Analysis  
  
```python  
# Analyze target variable  
def analyze_target_variable(df, target_col):  
    """  
    Comprehensive analysis of the target variable.  
      
    Parameters:  
    -----------  
    df : pandas DataFrame  
    target_col : str - name of target column  
    """  
    print(f"📊 TARGET VARIABLE ANALYSIS: {target_col}")  
    print("=" * 60)  
      
    target = df[target_col]  
      
    # Basic info  
    print(f"\nData Type: {target.dtype}")  
    print(f"Total Samples: {len(target)}")  
    print(f"Missing Values: {target.isnull().sum()} ({target.isnull().mean()*100:.2f}%)")  
    print(f"Unique Values: {target.nunique()}")  
      
    # Determine if classification or regression  
    if target.dtype == 'object' or target.nunique() <= 20:  
        task_type = 'Classification'  
          
        # Value counts  
        print(f"\n📊 Class Distribution:")  
        value_counts = target.value_counts()  
        value_pcts = target.value_counts(normalize=True) * 100  
          
        for val in value_counts.index:  
            print(f"   {val}: {value_counts[val]} ({value_pcts[val]:.2f}%)")  
          
        # Class imbalance check  
        majority_pct = value_pcts.max()  
        minority_pct = value_pcts.min()  
        imbalance_ratio = majority_pct / minority_pct  
          
        print(f"\n⚖️ Class Imbalance Check:")  
        print(f"   Majority Class: {value_pcts.idxmax()} ({majority_pct:.2f}%)")  
        print(f"   Minority Class: {value_pcts.idxmin()} ({minority_pct:.2f}%)")  
        print(f"   Imbalance Ratio: {imbalance_ratio:.2f}:1")  
          
        if imbalance_ratio > 3:  
            print("   ⚠️ Warning: Significant class imbalance detected!")  
            print("   Consider: SMOTE, undersampling, class weights, or stratified sampling")  
          
        # Visualization  
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))  
          
        # Bar plot  
        ax1 = axes[0]  
        colors = plt.cm.viridis(np.linspace(0.3, 0.7, len(value_counts)))  
        bars = ax1.bar(value_counts.index.astype(str), value_counts.values,   
                      color=colors, edgecolor='black')  
        ax1.set_title(f'Distribution of {target_col}', fontsize=12, fontweight='bold')  
        ax1.set_xlabel(target_col, fontsize=11)  
        ax1.set_ylabel('Count', fontsize=11)  
          
        # Add value labels  
        for bar, pct in zip(bars, value_pcts.values):  
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,  
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')  
          
        # Pie chart  
        ax2 = axes[1]  
        ax2.pie(value_counts.values, labels=value_counts.index.astype(str),  
               autopct='%1.1f%%', colors=colors, explode=[0.02]*len(value_counts),  
               shadow=True, startangle=90)  
        ax2.set_title(f'Proportion of {target_col}', fontsize=12, fontweight='bold')  
          
        plt.tight_layout()  
        plt.savefig(f'target_distribution_{target_col}.png', dpi=300, bbox_inches='tight')  
        plt.show()  
          
    else:  
        task_type = 'Regression'  
          
        # Statistics  
        print(f"\n📊 Target Statistics:")  
        print(f"   Mean: {target.mean():.4f}")  
        print(f"   Median: {target.median():.4f}")  
        print(f"   Std Dev: {target.std():.4f}")  
        print(f"   Min: {target.min():.4f}")  
        print(f"   Max: {target.max():.4f}")  
        print(f"   Skewness: {target.skew():.4f}")  
        print(f"   Kurtosis: {target.kurtosis():.4f}")  
          
        # Visualization  
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  
          
        # Histogram  
        axes[0].hist(target.dropna(), bins=30, color='steelblue',   
                    edgecolor='black', alpha=0.7)  
        axes[0].axvline(target.mean(), color='red', linestyle='--',   
                       label=f'Mean: {target.mean():.2f}')  
        axes[0].axvline(target.median(), color='green', linestyle='--',  
                       label=f'Median: {target.median():.2f}')  
        axes[0].set_title(f'Distribution of {target_col}', fontsize=12, fontweight='bold')  
        axes[0].set_xlabel(target_col, fontsize=11)  
        axes[0].set_ylabel('Frequency', fontsize=11)  
        axes[0].legend()  
          
        # Box plot  
        axes[1].boxplot(target.dropna(), vert=True)  
        axes[1].set_title(f'Box Plot of {target_col}', fontsize=12, fontweight='bold')  
        axes[1].set_ylabel(target_col, fontsize=11)  
          
        # Q-Q plot  
        from scipy import stats  
        stats.probplot(target.dropna(), dist="norm", plot=axes[2])  
        axes[2].set_title(f'Q-Q Plot of {target_col}', fontsize=12, fontweight='bold')  
          
        plt.tight_layout()  
        plt.savefig(f'target_distribution_{target_col}.png', dpi=300, bbox_inches='tight')  
        plt.show()  
      
    print(f"\n🎯 Task Type: {task_type}")  
      
    return task_type  
  
# Analyze target variable  
task_type = analyze_target_variable(df, 'survived')  
```  
  
### 11.2 Feature Importance (Preliminary)  
  
```python  
# Calculate preliminary feature importance  
def calculate_feature_importance(df, target_col, numerical_cols, categorical_cols):  
    """  
    Calculate preliminary feature importance using correlation and statistical tests.  
    """  
    from scipy import stats  
      
    importance_results = []  
      
    # Numerical features - correlation with target  
    print("📊 FEATURE IMPORTANCE ANALYSIS")  
    print("=" * 70)  
      
    # For numerical features  
    print("\n🔢 Numerical Features:")  
    for col in numerical_cols:  
        if col != target_col:  
            # Calculate correlation  
            valid_data = df[[col, target_col]].dropna()  
            if len(valid_data) > 2:  
                corr, p_value = stats.pearsonr(valid_data[col], valid_data[target_col])  
                  
                importance_results.append({  
                    'Feature': col,  
                    'Type': 'Numerical',  
                    'Metric': 'Pearson Correlation',  
                    'Value': abs(corr),  
                    'Raw Value': corr,  
                    'P-value': p_value  
                })  
      
    # For categorical features  
    print("\n📝 Categorical Features:")  
    for col in categorical_cols:  
        if col != target_col and df[col].nunique() <= 20:  
            try:  
                # Chi-square test  
                contingency = pd.crosstab(df[col], df[target_col])  
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)  
                  
                # Cramér's V  
                n = contingency.sum().sum()  
                min_dim = min(contingency.shape) - 1  
                cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0  
                  
                importance_results.append({  
                    'Feature': col,  
                    'Type': 'Categorical',  
                    'Metric': "Cramér's V",  
                    'Value': cramers_v,  
                    'Raw Value': cramers_v,  
                    'P-value': p_value  
                })  
            except Exception as e:  
                print(f"   Error processing {col}: {e}")  
      
    # Create DataFrame and sort by importance  
    importance_df = pd.DataFrame(importance_results)  
    importance_df = importance_df.sort_values('Value', ascending=False)  
      
    print("\n📊 Feature Importance Ranking:")  
    print(importance_df[['Feature', 'Type', 'Metric', 'Value', 'P-value']].to_string(index=False))  
      
    # Visualize  
    fig, ax = plt.subplots(figsize=(12, max(6, len(importance_df) * 0.4)))  
      
    colors = ['steelblue' if t == 'Numerical' else 'coral'   
              for t in importance_df['Type']]  
      
    bars = ax.barh(importance_df['Feature'], importance_df['Value'], color=colors,  
                   edgecolor='black', alpha=0.8)  
      
    ax.set_xlabel('Importance Score', fontsize=12)  
    ax.set_ylabel('Feature', fontsize=12)  
    ax.set_title(f'Feature Importance for Predicting {target_col}',   
                fontsize=14, fontweight='bold')  
      
    # Add legend  
    from matplotlib.patches import Patch  
    legend_elements = [Patch(facecolor='steelblue', label='Numerical'),  
                      Patch(facecolor='coral', label='Categorical')]  
    ax.legend(handles=legend_elements, loc='lower right')  
      
    # Add value labels  
    for bar, val in zip(bars, importance_df['Value']):  
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,  
               f'{val:.3f}', va='center', fontsize=9)  
      
    plt.tight_layout()  
    plt.savefig(f'feature_importance_{target_col}.png', dpi=300, bbox_inches='tight')  
    plt.show()  
      
    return importance_df  
  
# Calculate feature importance  
importance_df = calculate_feature_importance(df, 'survived', numerical_cols, categorical_cols)  
```  
  
### 11.3 Feature vs Target Visualization  
  
```python  
# Visualize relationship between features and target  
def visualize_features_vs_target(df, target_col, numerical_cols, categorical_cols, top_n=6):  
    """  
    Create comprehensive visualizations of feature-target relationships.  
    """  
    # Numerical features vs target  
    num_cols_plot = [col for col in numerical_cols if col != target_col][:top_n]  
      
    if num_cols_plot:  
        n_cols = 3  
        n_rows = (len(num_cols_plot) + n_cols - 1) // n_cols  
          
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))  
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes  
          
        for idx, col in enumerate(num_cols_plot):  
            ax = axes[idx]  
              
            # Check if target is categorical  
            if df[target_col].nunique() <= 10:  
                sns.boxplot(data=df, x=target_col, y=col, ax=ax, palette='viridis')  
                ax.set_title(f'{col} by {target_col}', fontsize=11, fontweight='bold')  
            else:  
                ax.scatter(df[col], df[target_col], alpha=0.5, s=30)  
                # Add regression line  
                mask = df[[col, target_col]].notna().all(axis=1)  
                if mask.sum() > 2:  
                    z = np.polyfit(df.loc[mask, col], df.loc[mask, target_col], 1)  
                    p = np.poly1d(z)  
                    x_line = np.linspace(df[col].min(), df[col].max(), 100)  
                    ax.plot(x_line, p(x_line), 'r--', linewidth=2)  
                ax.set_title(f'{col} vs {target_col}', fontsize=11, fontweight='bold')  
              
            ax.set_xlabel(col, fontsize=10)  
            ax.set_ylabel(target_col, fontsize=10)  
          
        # Hide empty subplots  
        for idx in range(len(num_cols_plot), len(axes)):  
            axes[idx].set_visible(False)  
          
        plt.suptitle(f'Numerical Features vs {target_col}', fontsize=16, fontweight='bold', y=1.02)  
        plt.tight_layout()  
        plt.savefig(f'numerical_vs_{target_col}.png', dpi=300, bbox_inches='tight')  
        plt.show()  
      
    # Categorical features vs target  
    cat_cols_plot = [col for col in categorical_cols   
                     if col != target_col and df[col].nunique() <= 10][:top_n]  
      
    if cat_cols_plot:  
        n_cols = 2  
        n_rows = (len(cat_cols_plot) + n_cols - 1) // n_cols  
          
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))  
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes  
          
        for idx, col in enumerate(cat_cols_plot):  
            ax = axes[idx]  
              
            # Create stacked bar chart  
            crosstab = pd.crosstab(df[col], df[target_col], normalize='index') * 100  
            crosstab.plot(kind='bar', stacked=True, ax=ax, colormap='viridis',  
                         edgecolor='white', width=0.7)  
              
            ax.set_title(f'{col} vs {target_col}', fontsize=11, fontweight='bold')  
            ax.set_xlabel(col, fontsize=10)  
            ax.set_ylabel('Percentage (%)', fontsize=10)  
            ax.legend(title=target_col, bbox_to_anchor=(1.02, 1), loc='upper left')  
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')  
          
        # Hide empty subplots  
        for idx in range(len(cat_cols_plot), len(axes)):  
            axes[idx].set_visible(False)  
          
        plt.suptitle(f'Categorical Features vs {target_col}', fontsize=16, fontweight='bold', y=1.02)  
        plt.tight_layout()  
        plt.savefig(f'categorical_vs_{target_col}.png', dpi=300, bbox_inches='tight')  
        plt.show()  
  
# Visualize features vs target  
visualize_features_vs_target(df, 'survived', numerical_cols, categorical_cols)  
```  
  
---  
  
## 12. Automated EDA Report Generation  
  
### 12.1 Complete EDA Report Function  
  
```python  
# Generate complete EDA report  
def generate_eda_report(df, target_col=None, output_dir='eda_report'):  
    """  
    Generate a comprehensive EDA report with all visualizations and statistics.  
      
    Parameters:  
    -----------  
    df : pandas DataFrame  
    target_col : str - target variable name (optional)  
    output_dir : str - directory to save outputs  
    """  
    import os  
    from datetime import datetime  
      
    # Create output directory  
    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)  
      
    # Start report  
    report = []  
    report.append("=" * 80)  
    report.append("EXPLORATORY DATA ANALYSIS REPORT")  
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")  
    report.append("=" * 80)  
      
    # 1. Dataset Overview  
    report.append("\n" + "=" * 80)  
    report.append("1. DATASET OVERVIEW")  
    report.append("=" * 80)  
    report.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")  
    report.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")  
      
    # Column types  
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()  
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()  
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()  
      
    report.append(f"\nColumn Types:")  
    report.append(f"  - Numerical: {len(numerical_cols)}")  
    report.append(f"  - Categorical: {len(categorical_cols)}")  
    report.append(f"  - DateTime: {len(datetime_cols)}")  
      
    # 2. Missing Values  
    report.append("\n" + "=" * 80)  
    report.append("2. MISSING VALUES ANALYSIS")  
    report.append("=" * 80)  
      
    missing = df.isnull().sum()  
    missing_pct = (missing / len(df)) * 100  
    missing_df = pd.DataFrame({  
        'Missing Count': missing,  
        'Missing %': missing_pct  
    }).sort_values('Missing %', ascending=False)  
      
    report.append(missing_df[missing_df['Missing Count'] > 0].to_string())  
      
    # 3. Descriptive Statistics  
    report.append("\n" + "=" * 80)  
    report.append("3. DESCRIPTIVE STATISTICS")  
    report.append("=" * 80)  
      
    if numerical_cols:  
        report.append("\nNumerical Variables:")  
        report.append(df[numerical_cols].describe().round(2).to_string())  
      
    if categorical_cols:  
        report.append("\nCategorical Variables:")  
        report.append(df[categorical_cols].describe().to_string())  
      
    # 4. Correlation Analysis  
    if len(numerical_cols) > 1:  
        report.append("\n" + "=" * 80)  
        report.append("4. CORRELATION ANALYSIS")  
        report.append("=" * 80)  
          
        corr_matrix = df[numerical_cols].corr()  
        report.append("\nCorrelation Matrix:")  
        report.append(corr_matrix.round(3).to_string())  
          
        # Strong correlations  
        strong_corrs = []  
        for i in range(len(corr_matrix.columns)):  
            for j in range(i+1, len(corr_matrix.columns)):  
                if abs(corr_matrix.iloc[i, j]) >= 0.5:  
                    strong_corrs.append({  
                        'Feature 1': corr_matrix.columns[i],  
                        'Feature 2': corr_matrix.columns[j],  
                        'Correlation': corr_matrix.iloc[i, j]  
                    })  
          
        if strong_corrs:  
            report.append("\nStrong Correlations (|r| >= 0.5):")  
            for sc in strong_corrs:  
                report.append(f"  {sc['Feature 1']} <-> {sc['Feature 2']}: {sc['Correlation']:.3f}")  
      
    # 5. Target Variable Analysis  
    if target_col and target_col in df.columns:  
        report.append("\n" + "=" * 80)  
        report.append("5. TARGET VARIABLE ANALYSIS")  
        report.append("=" * 80)  
          
        target = df[target_col]  
        report.append(f"\nTarget: {target_col}")  
        report.append(f"Data Type: {target.dtype}")  
        report.append(f"Unique Values: {target.nunique()}")  
          
        if target.dtype == 'object' or target.nunique() <= 20:  
            report.append("\nClass Distribution:")  
            value_counts = target.value_counts()  
            for val, count in value_counts.items():  
                pct = (count / len(target)) * 100  
                report.append(f"  {val}: {count} ({pct:.2f}%)")  
        else:  
            report.append(f"\nMean: {target.mean():.4f}")  
            report.append(f"Median: {target.median():.4f}")  
            report.append(f"Std Dev: {target.std():.4f}")  
      
    # 6. Recommendations  
    report.append("\n" + "=" * 80)  
    report.append("6. RECOMMENDATIONS")  
    report.append("=" * 80)  
      
    recommendations = []  
      
    # Missing value recommendations  
    high_missing = missing_pct[missing_pct > 30].index.tolist()  
    if high_missing:  
        recommendations.append(f"⚠️ Consider dropping or imputing columns with >30% missing: {high_missing}")  
      
    # Outlier recommendations  
    for col in numerical_cols:  
        Q1 = df[col].quantile(0.25)  
        Q3 = df[col].quantile(0.75)  
        IQR = Q3 - Q1  
        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()  
        if outliers > len(df) * 0.05:  
            recommendations.append(f"⚠️ {col} has {outliers} outliers ({outliers/len(df)*100:.1f}%) - consider treatment")  
      
    # Skewness recommendations  
    for col in numerical_cols:  
        skew = df[col].skew()  
        if abs(skew) > 1:  
            recommendations.append(f"⚠️ {col} is highly skewed ({skew:.2f}) - consider transformation")  
      
    for rec in recommendations:  
        report.append(rec)  
      
    # Save report  
    report_text = "\n".join(report)  
      
    with open(f'{output_dir}/eda_report.txt', 'w') as f:  
        f.write(report_text)  
      
    print(report_text)  
    print(f"\n✅ Report saved to {output_dir}/eda_report.txt")  
      
    return report_text  
  
# Generate EDA report  
report = generate_eda_report(df, target_col='survived', output_dir='eda_output')  
```  
  
### 12.2 Using Automated EDA Libraries  
  
```python  
# Using pandas-profiling (ydata-profiling)  
def generate_profiling_report(df, output_file='profile_report.html'):  
    """  
    Generate automated EDA report using ydata-profiling.  
    """  
    try:  
        from ydata_profiling import ProfileReport  
          
        profile = ProfileReport(  
            df,  
            title="Automated EDA Report",  
            explorative=True,  
            dark_mode=False  
        )  
          
        profile.to_file(output_file)  
        print(f"✅ Profile report saved to {output_file}")  
          
        return profile  
      
    except ImportError:  
        print("ydata-profiling not installed. Install with: pip install ydata-profiling")  
        return None  
  
# Generate profiling report  
# profile = generate_profiling_report(df)  
  
# Using sweetviz  
def generate_sweetviz_report(df, target_col=None, output_file='sweetviz_report.html'):  
    """  
    Generate automated EDA report using sweetviz.  
    """  
    try:  
        import sweetviz as sv  
          
        if target_col:  
            report = sv.analyze(df, target_feat=target_col)  
        else:  
            report = sv.analyze(df)  
          
        report.show_html(output_file)  
        print(f"✅ Sweetviz report saved to {output_file}")  
          
        return report  
      
    except ImportError:  
        print("sweetviz not installed. Install with: pip install sweetviz")  
        return None  
  
# Generate sweetviz report  
# sweetviz_report = generate_sweetviz_report(df, target_col='survived')  
```  
  
---  
  
## 13. EDA Checklist and Summary  
  
### 13.1 EDA Checklist  
  
```python  
# EDA Checklist function  
def eda_checklist(df, target_col=None):  
    """  
    Print EDA checklist with completion status.  
    """  
    print("=" * 70)  
    print("📋 EDA CHECKLIST")  
    print("=" * 70)  
      
    checklist = {  
        "1. Data Loading & Overview": {  
            "Load data": True,  
            "Check shape": True,  
            "View first/last rows": True,  
            "Check data types": True,  
            "Check memory usage": True  
        },  
        "2. Data Quality Assessment": {  
            "Identify missing values": True,  
            "Check for duplicates": True,  
            "Identify data type mismatches": True,  
            "Check for inconsistent values": True  
        },  
        "3. Univariate Analysis": {  
            "Numerical: Histograms": True,  
            "Numerical: Box plots": True,  
            "Numerical: Descriptive statistics": True,  
            "Categorical: Bar charts": True,  
            "Categorical: Frequency tables": True  
        },  
        "4. Bivariate Analysis": {  
            "Numerical vs Numerical: Scatter plots": True,  
            "Numerical vs Numerical: Correlation": True,  
            "Categorical vs Numerical: Box plots": True,  
            "Categorical vs Numerical: Statistical tests": True,  
            "Categorical vs Categorical: Cross-tabulation": True,  
            "Categorical vs Categorical: Chi-square test": True  
        },  
        "5. Multivariate Analysis": {  
            "Pair plots": True,  
            "Correlation heatmap": True,  
            "Faceted visualizations": True  
        },  
        "6. Outlier Analysis": {  
            "Identify outliers (IQR method)": True,  
            "Identify outliers (Z-score)": True,  
            "Visualize outliers": True  
        },  
        "7. Target Variable Analysis": {  
            "Distribution analysis": target_col is not None,  
            "Feature importance": target_col is not None,  
            "Feature-target relationships": target_col is not None  
        }  
    }  
      
    for section, items in checklist.items():  
        print(f"\n{section}")  
        for item, status in items.items():  
            status_icon = "✅" if status else "❌"  
            print(f"  {status_icon} {item}")  
      
    print("\n" + "=" * 70)  
  
# Run checklist  
eda_checklist(df, target_col='survived')  
```  
  
### 13.2 Key Findings Summary Template  
  
```python  
# Generate key findings summary  
def generate_findings_summary(df, target_col=None):  
    """  
    Generate a summary of key findings from EDA.  
    """  
    print("=" * 70)  
    print("📊 KEY FINDINGS SUMMARY")  
    print("=" * 70)  
      
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()  
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()  
      
    # Dataset Overview  
    print("\n📁 DATASET OVERVIEW:")  
    print(f"   • Total samples: {len(df):,}")  
    print(f"   • Total features: {len(df.columns)}")  
    print(f"   • Numerical features: {len(numerical_cols)}")  
    print(f"   • Categorical features: {len(categorical_cols)}")  
      
    # Missing Values  
    missing_total = df.isnull().sum().sum()  
    missing_cols = (df.isnull().sum() > 0).sum()  
    print(f"\n❓ MISSING VALUES:")  
    print(f"   • Total missing values: {missing_total:,}")  
    print(f"   • Columns with missing: {missing_cols}")  
      
    # Duplicates  
    duplicates = df.duplicated().sum()  
    print(f"\n🔄 DUPLICATES:")  
    print(f"   • Duplicate rows: {duplicates}")  
      
    # Outliers  
    print(f"\n📈 OUTLIERS (IQR Method):")  
    for col in numerical_cols[:5]:  # Limit to first 5  
        Q1 = df[col].quantile(0.25)  
        Q3 = df[col].quantile(0.75)  
        IQR = Q3 - Q1  
        outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()  
        if outliers > 0:  
            print(f"   • {col}: {outliers} outliers ({outliers/len(df)*100:.1f}%)")  
      
    # Correlations  
    if len(numerical_cols) > 1:  
        corr_matrix = df[numerical_cols].corr()  
        print(f"\n🔗 STRONG CORRELATIONS (|r| > 0.5):")  
        for i in range(len(corr_matrix.columns)):  
            for j in range(i+1, len(corr_matrix.columns)):  
                if abs(corr_matrix.iloc[i, j]) > 0.5:  
                    print(f"   • {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")  
      
    # Target Variable  
    if target_col and target_col in df.columns:  
        print(f"\n🎯 TARGET VARIABLE ({target_col}):")  
        if df[target_col].nunique() <= 10:  
            for val, count in df[target_col].value_counts().items():  
                pct = count / len(df) * 100  
                print(f"   • {val}: {count:,} ({pct:.1f}%)")  
        else:  
            print(f"   • Mean: {df[target_col].mean():.4f}")  
            print(f"   • Median: {df[target_col].median():.4f}")  
            print(f"   • Std: {df[target_col].std():.4f}")  
      
    # Recommendations  
    print(f"\n💡 RECOMMENDATIONS:")  
      
    # Missing value handling  
    high_missing = df.columns[df.isnull().mean() > 0.3].tolist()  
    if high_missing:  
        print(f"   • Consider handling columns with >30% missing: {high_missing}")  
      
    # Skewed features  
    skewed = [col for col in numerical_cols if abs(df[col].skew()) > 1]  
    if skewed:  
        print(f"   • Consider transforming skewed features: {skewed[:5]}")  
      
    print("\n" + "=" * 70)  
  
# Generate findings summary  
generate_findings_summary(df, target_col='survived')  
```  
  
---  
  
## 14. Quick Reference Code Snippets  
  
```python  
# ============================================================  
# QUICK REFERENCE - COPY-PASTE SNIPPETS  
# ============================================================  
  
# ----- Data Loading -----  
df = pd.read_csv('data.csv')  
df = pd.read_excel('data.xlsx')  
df = pd.read_json('data.json')  
  
# ----- Quick Overview -----  
df.shape  
df.head()  
df.info()  
df.describe()  
df.dtypes  
  
# ----- Missing Values -----  
df.isnull().sum()  
df.isnull().sum() / len(df) * 100  
  
# ----- Duplicates -----  
df.duplicated().sum()  
df.drop_duplicates(inplace=True)  
  
# ----- Column Types -----  
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()  
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()  
  
# ----- Value Counts -----  
df['column'].value_counts()  
df['column'].value_counts(normalize=True)  
  
# ----- Unique Values -----  
df['column'].nunique()  
df['column'].unique()  
  
# ----- Correlation -----  
df.corr()  
df[['col1', 'col2']].corr()  
  
# ----- Groupby Statistics -----  
df.groupby('category')['numerical'].agg(['mean', 'median', 'std', 'count'])  
  
# ----- Cross Tabulation -----  
pd.crosstab(df['cat1'], df['cat2'])  
pd.crosstab(df['cat1'], df['cat2'], normalize='index')  
  
# ----- Quick Plots -----  
df['numerical'].hist(bins=30)  
df['categorical'].value_counts().plot(kind='bar')  
df.plot.scatter(x='col1', y='col2')  
sns.heatmap(df.corr(), annot=True)  
sns.boxplot(x='category', y='numerical', data=df)  
  
# ----- Outlier Detection (IQR) -----  
Q1 = df['col'].quantile(0.25)  
Q3 = df['col'].quantile(0.75)  
IQR = Q3 - Q1  
outliers = df[(df['col'] < Q1 - 1.5*IQR) | (df['col'] > Q3 + 1.5*IQR)]  
  
# ----- Statistical Tests -----  
from scipy import stats  
# t-test  
stats.ttest_ind(group1, group2)  
# Chi-square  
stats.chi2_contingency(pd.crosstab(df['cat1'], df['cat2']))  
# Correlation test  
stats.pearsonr(df['col1'], df['col2'])  
```  
  
---  
  
This comprehensive EDA template provides a systematic approach to exploring and understanding any dataset. Remember to adapt the code to your specific dataset and analysis goals. Happy exploring! 🎉


|\n| `NONFICTION`| `1`                       |\n| `ACTION`    | `2`                       |\n| `THRILLER`  | `3`                       |\n| `TECHNOLOGY`| `4`                      |\n| `DRAMA`     | `5`                       |\n| `POETRY`    | `6`                       |\n| `MEDIA`     | `7`                       |\n| `OTHERS`    | `8`                       |\n\n- **Utility Methods:**\n- `valueOf(int category)`: Returns the `Category` enum constant for the given integer value.\n- `getValue()`: Returns the integer value of the enum constant.\n\n## DTO Classes\n\n### BookDto\n\n- **Name:** `BookDto`\n- **Usage:** Used as an API response object for book details.\n\n| Field Name | Java Type | Validation Annotations | Description |\n|------------|-----------|-----------------------|-------------|\n| `id`       | `Long`    |                       | Book Unique Id |\n| `title`    | `String`  |                       | Title of the book |\n| `author`   | `String`  |                       | Author of the book |\n| `category` | `Category`|                       | Category of the book |\n| `price`    | `float`   | `@Min(0)`             | Price of the book |\n| `totalCount` | `int`  | `@Min(0)`             | Copies of book available on the store |\n\n- **Relationship to Entity:** Maps from `Book` entity.\n- **Fields Included\/Excluded:**\n- Included: `id`, `title`, `author`, `category`, `price`, `totalCount`\n- Excluded: `sold`\n\n### SellDto\n\n- **Name:** `SellDto`\n- **Usage:** Used as an API request object for selling books.\n\n| Field Name | Java Type | Validation Annotations | Description |\n|------------|-----------|-----------------------|-------------|\n| `bookId`   | `long`    |                       | Id of the book to be sold |\n| `quantity` | `int`     | `@Min(0)`             | Number of copies of the book to be sold |\n\n- **Relationship to Entity:** Does not directly map to any entity but is used in conjunction with `Book` entity for selling operations.\n\n## Entity-DTO Mapping\n\n- **Mapping Library:** `ModelMapper`\n- **Custom Mapping Logic:** Not explicitly configured. Uses default `ModelMapper` behavior for mapping between `Book` and `BookDto`.\n\n## Database Configuration\n\n- **Database Type and URL:** MySQL, `jdbc:mysql:\/\/localhost:3306\/bookstore?serverTimezone=UTC`\n- **DDL Strategy:** `update`\n- **Hibernate-Specific Settings:**\n- `dialect`: `org.hibernate.dialect.MySQL5Dialect`\n- `generate-ddl`: `true`\n- `show-sql`: `true`\n\n## Key Observations\n\n- The `Book` entity includes fields for tracking the number of books sold, which is not included in the `BookDto` used for API responses.\n- The `Category` enum provides a reverse lookup method `valueOf(int)` for converting integer values to enum constants.\n- The `SellDto` is used for handling book sales but does not map directly to any entity, indicating a separate business logic for selling books.\n- The database configuration uses the `update` strategy for DDL, which will update the database schema based on the entity definitions.\n\n## Deterministic Source Facts\n\nUse these facts as source-grounded constraints extracted from Java\/reference inputs. Do not weaken or omit them during migration.\n\n### Method Contract Facts\n\n- public ModelMapper modelMapper(List<PropertyMap> propertyMaps)\n\n\n\n## Brief Specification (Pre-loaded)\nUse this as the primary authority for generation.\n\n## 0. Java Source Reference\n```java\npackage com.sulzer.bookstore.config;\n\nimport org.modelmapper.ModelMapper;\nimport org.modelmapper.PropertyMap;\nimport org.springframework.context.annotation.Bean;\nimport org.springframework.context.annotation.Configuration;\n\nimport java.util.List;\nimport java.util.Objects;\n\n@Configuration\npublic class ModelMapperConfig {\n\/**\n* Initializes {@link ModelMapper} with available {@link PropertyMap} instances.\n*\n* @param propertyMaps\n* @return\n*\/\n@Bean\npublic ModelMapper modelMapper(List<PropertyMap> propertyMaps) {\nfinal ModelMapper modelMapper = new ModelMapper();\npropertyMaps.stream().filter(Objects::nonNull).forEach(modelMapper::addMappings);\nreturn modelMapper;\n}\n}\n```\n## 1. Identity\nThe namespace for this component is `GenReviveDemo.Bookstore.Configuration`. The component type is `mapper_profile`, and the .NET role is to replace the Java ModelMapper configuration with an AutoMapper profile.\n\n## 2. Structure\nThis is a public class that inherits from AutoMapper's `Profile` class.\n\n## 3. Dependencies & Namespaces\n- `AutoMapper` for the `Profile` base class.\n- `GenReviveDemo.Bookstore.Models` for the `Book` entity.\n- `GenReviveDemo.Bookstore.DTOs` for the `BookDto`.\n\n## 6. Method Specifications (Constructor)\nThe constructor of the `MappingProfile` class should include the following `CreateMap` calls:\n1. `CreateMap<Book, BookDto>().ReverseMap();`\n- This maps `Book` to `BookDto` and vice versa.\n- This is verified by the Java source, which uses `modelMapper.map(book, BookDto.class)` and `modelMapper.map(bookDto, Book.class)`.\n2. `ForMember(dest => dest.Price, opt => opt.MapFrom(src => src.Price));`\n- This handles the conversion from `float` to `decimal` for the `Price` property.\n- This is verified by the migration notes.\n3. `ForMember(dest => dest.TotalCount, opt => opt.MapFrom(src => src.TotalCount));`\n- This handles the conversion from `int` to `int` for the `TotalCount` property.\n- This is verified by the migration notes.\n4. `ForMember(dest => dest.Sold, opt => opt.Ignore());`\n- This ignores the `Sold` property in `BookDto`.\n- This is verified by the migration notes.\n\n## 7. Verified Dependency API\n```csharp\nnamespace GenReviveDemo.Bookstore.Models\n{\n[Table(\"book\")]\npublic class Book\n{\n[Key]\npublic long Id { get; set; }\n\n[Required]\npublic string Title { get; set; }\n\npublic string Author { get; set; }\n\npublic Category Category { get; set; }\n\n[Range(0, int.MaxValue, ErrorMessage = \"Price should be positive value.\")]\npublic decimal Price { get; set; }\n\n[Range(0, int.MaxValue, ErrorMessage = \"Total Count should be positive value.\")]\npublic int TotalCount { get; set; }\n\n[Range(0, int.MaxValue, ErrorMessage = \"Total sell should be positive value.\")]\npublic int Sold { get; set; }\n}\n}\n\nnamespace GenReviveDemo.Bookstore.DTOs\n{\npublic class BookDto\n{\n\/\/\/ <summary>\n\/\/\/ Unique Book Number given by company.\n\/\/\/ Eg: ISBN number\n\/\/\/ <\/summary>\npublic long? Id { get; set; }\n\n\/\/\/ <summary>\n\/\/\/ Title of the book\n\/\/\/ <\/summary>\npublic string Title { get; set; }\n\n\/\/\/ <summary>\n\/\/\/ Author of the book\n\/\/\/ <\/summary>\npublic string Author { get; set; }\n\n\/\/\/ <summary>\n\/\/\/ Category of the book\n\/\/\/ Eg: Novel, Fiction, etc\n\/\/\/ <\/summary>\npublic Category Category { get; set; }\n\n\/\/\/ <summary>\n\/\/\/ Price of the book\n\/\/\/ <\/summary>\n[Range(0, double.MaxValue, ErrorMessage = \"Price should be a positive value.\")]\npublic decimal Price { get; set; }\n\n\/\/\/ <summary>\n\/\/\/ Copies of book available on the store\n\/\/\/ <\/summary>\n[Range(0, int.MaxValue, ErrorMessage = \"Total Count should be a positive value.\")]\npublic int TotalCount { get; set; }\n}\n}\n```\n## 8. Forbidden Actions\n- Do not create mappings for types that are not exercised by `modelMapper.map()` calls in the Java source.\n- Do not add custom `ForMember` rules unless explicitly specified in the Java source.\n- Do not use `ReverseMap` unless both directions are actually used in the Java source.\n- Do not create two separate `CreateMap` calls for the same type pair; use one `CreateMap` call with `ReverseMap` instead.\n- Do not inject `IMapper` into the profile class; it is registered at startup.\n\n\n\n\nThis is the expected criteria for your final answer: \nRunnable C# Code\nPut \"# filename: F:\/GENREVIVE_CORE_1_14\/GR_CORE_1_14\/output\/bookstore\/GenReviveDemo.Bookstore\/Configuration\/MappingProfile.cs\" inside each code block as first line.\nThe output must look like:                \n\n```C#\n# filename: F:\/GENREVIVE_CORE_1_14\/GR_CORE_1_14\/output\/bookstore\/GenReviveDemo.Bookstore\/Configuration\/MappingProfile.cs\n<code>\n```\n```markdown\n# filename: F:\/GENREVIVE_CORE_1_14\/GR_CORE_1_14\/output\/bookstore\/GenReviveDemo.Bookstore\/Configuration\/MappingProfile.cs.md\n<changelog>\n```\n\nyou MUST return the actual complete content as the final answer, not a summary.\n\nBegin! This is VERY important to you, use the tools available and give your best Final Answer, your job depends on it!\n\nThought:"}]},"grammar":"disabled","early_stop_config":"redacted","event":"chat completion request","timestamp":"2026-04-16T07:57:22.795304Z","level":"info","logger":"harmattan.app.services.base"}
# Feature Engineering - Detailed Concept Explanations

## Table of Contents
1. [Missing Value Handling](#missing-value-handling)
2. [Feature Creation Techniques](#feature-creation-techniques)
3. [Encoding Categorical Variables](#encoding-categorical-variables)
4. [Feature Scaling & Normalization](#feature-scaling--normalization)
5. [Feature Selection Methods](#feature-selection-methods)
6. [Advanced Feature Engineering](#advanced-feature-engineering)

---

## 1. Missing Value Handling

### 1.1 Statistical Imputation

**What is it?**
Replacing missing values with statistical measures (mean, median, mode) calculated from the available data.

**Types:**

#### Mean Imputation
- **Definition**: Replace missing values with the average of all non-missing values
- **When to use**: When data is normally distributed and doesn't have outliers
- **Example**:
  ```
  Ages: [25, 30, NaN, 35, 28]
  Mean = (25+30+35+28)/4 = 29.5
  Result: [25, 30, 29.5, 35, 28]
  ```
- **Pros**: Simple, fast, preserves mean
- **Cons**: Reduces variance, can distort distribution, sensitive to outliers

#### Median Imputation
- **Definition**: Replace missing values with the middle value when data is sorted
- **When to use**: When data is skewed or has outliers
- **Example**:
  ```
  Fares: [10, 15, NaN, 500, 20]
  Median = 15 (middle value)
  Result: [10, 15, 15, 500, 20]
  ```
- **Pros**: Robust to outliers, good for skewed distributions
- **Cons**: Still reduces variance

#### Mode Imputation
- **Definition**: Replace missing values with the most frequently occurring value
- **When to use**: Categorical variables or discrete numerical data
- **Example**:
  ```
  Embarked: ['S', 'C', NaN, 'S', 'S']
  Mode = 'S' (appears 3 times)
  Result: ['S', 'C', 'S', 'S', 'S']
  ```
- **Pros**: Makes sense for categorical data
- **Cons**: Can bias toward the majority class

### 1.2 Missing Indicators

**What is it?**
Creating a binary feature that indicates whether a value was missing in the original data.

**Why use it?**
Sometimes the fact that data is missing carries important information. For example:
- Missing cabin number might indicate lower-class passengers
- Missing age might indicate certain demographics

**Example**:
```
Original Cabin: ['C85', NaN, 'E46', NaN]
Cabin_Missing:  [0, 1, 0, 1]
```

**Strategy**: Keep both the imputed feature AND the missing indicator.

---

## 2. Feature Creation Techniques

### 2.1 Combination Features

**What is it?**
Creating new features by combining two or more existing features through mathematical operations.

#### Addition/Subtraction
**Example - Family Size**:
```python
FamilySize = SibSp + Parch + 1
# SibSp=2, Parch=1 → FamilySize=4 (2 siblings + 1 parent + self)
```

**Logic**: Combines related features to capture total effect.

#### Multiplication (Interaction Features)
**Example - Age × Class**:
```python
Age_Class = Age * Pclass
# Age=30, Pclass=3 → Age_Class=90
# Age=30, Pclass=1 → Age_Class=30
```

**Logic**: Captures how two features interact. A 30-year-old in 3rd class is different from a 30-year-old in 1st class.

#### Division (Ratio Features)
**Example - Fare Per Person**:
```python
FarePerPerson = Fare / FamilySize
# Fare=100, FamilySize=4 → FarePerPerson=25
```

**Logic**: Normalizes one feature by another to get per-unit values.

### 2.2 Binning (Discretization)

**What is it?**
Converting continuous numerical variables into categorical bins or groups.

**Types:**

#### Equal-Width Binning
Divides the range into equal-sized intervals.
```python
Age: [5, 15, 25, 35, 45, 55, 65, 75]
Bins: [0-20, 20-40, 40-60, 60-80]
Result: ['0-20', '0-20', '20-40', '20-40', '40-60', '40-60', '60-80', '60-80']
```

#### Equal-Frequency Binning (Quantiles)
Each bin contains approximately the same number of observations.
```python
Fare: [10, 15, 20, 25, 30, 35, 40, 45] (8 values)
Quartiles (4 bins): Each bin gets 2 values
Q1: [10, 15]
Q2: [20, 25]
Q3: [30, 35]
Q4: [40, 45]
```

#### Custom Binning
Based on domain knowledge.
```python
Age Groups:
- Child: 0-12
- Teen: 13-18
- Adult: 19-60
- Senior: 60+
```

**Why bin?**
- Reduces noise in data
- Handles outliers
- Can improve model performance
- Makes relationships non-linear
- More interpretable

### 2.3 Text Feature Extraction

**What is it?**
Extracting meaningful information from text columns.

#### Title Extraction
**Example from Name column**:
```
Name: "Braund, Mr. Owen Harris"
Extract: "Mr."

Name: "Cumings, Mrs. John Bradley"
Extract: "Mrs."
```

**Why useful?**
Titles indicate social status, gender, marital status, and age group.

**Technique**: Use regular expressions
```python
title = name.str.extract(' ([A-Za-z]+)\.')
# Looks for pattern: space + letters + period
```

#### Text Length
```python
TicketLength = len(Ticket)
# "A/5 21171" → 9 characters
```

#### Text Patterns
```python
TicketPrefix = Ticket.split()[0]
# "A/5 21171" → "A/5"
# "STON/O2. 3101282" → "STON/O2."
```

### 2.4 Aggregation Features

**What is it?**
Creating features based on group statistics.

**Example**:
```python
# Average fare per passenger class
Mean Fare by Class:
Class 1: $84.15
Class 2: $20.66
Class 3: $13.68

# For each passenger, add their class's average
Passenger A: Pclass=1 → MeanFare_PerClass=84.15
Passenger B: Pclass=3 → MeanFare_PerClass=13.68
```

**Common aggregations**:
- Mean, Median, Mode
- Min, Max, Range
- Standard deviation
- Count, Sum
- Percentiles

**Why useful?**
Provides context - how does this individual compare to their group?

### 2.5 Polynomial Features

**What is it?**
Creating powers or products of features.

#### Square/Cubic Features
```python
Age² = Age * Age
# Age=30 → Age²=900
# Age=10 → Age²=100
```

**Why useful?**
Captures non-linear relationships. The effect of age might not be linear - being 60 might have more than 2× the effect of being 30.

#### Log Transformation
```python
Fare_Log = log(Fare + 1)
# Fare=100 → log(101) ≈ 4.62
# Fare=1000 → log(1001) ≈ 6.91
```

**Why useful?**
- Reduces skewness
- Handles exponential relationships
- Compresses large values
- Makes data more normal

---

## 3. Encoding Categorical Variables

Machine learning algorithms require numerical input. Encoding converts categorical data to numbers.

### 3.1 Label Encoding (Ordinal Variables)

**What is it?**
Assigning a unique integer to each category.

**When to use**: For **ordinal** variables (categories with inherent order/ranking).

**Example 1 - Education Level** (ORDINAL - has order):
```
Original: ['High School', 'Bachelor', 'Master', 'PhD']
Encoded:  [0, 1, 2, 3]
```
✓ Correct use - higher numbers mean higher education

**Example 2 - T-Shirt Size** (ORDINAL - has order):
```
Original: ['S', 'M', 'L', 'XL']
Encoded:  [0, 1, 2, 3]
```
✓ Correct use - preserves size ordering

**Example 3 - Embarked Port** (NOMINAL - NO order):
```
Original: ['Southampton', 'Cherbourg', 'Queenstown']
Encoded:  [0, 1, 2]
```
✗ Wrong use - implies Southampton < Cherbourg < Queenstown (no such relationship exists!)

**How it works**:
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
sex_encoded = le.fit_transform(['male', 'female', 'male', 'female'])
# Result: [1, 0, 1, 0]

# Mapping: male → 1, female → 0
```

**Pros**:
- Memory efficient
- Works well with tree-based models
- Preserves ordinal relationships

**Cons**:
- Implies order when there isn't one (if misused)
- Linear models might interpret as continuous

### 3.2 One-Hot Encoding (Nominal Variables)

**What is it?**
Creating a binary column for each category.

**When to use**: For **nominal** variables (categories with NO inherent order).

**Example - Embarked Port**:
```
Original data:
PassengerID | Embarked
1           | S
2           | C
3           | S
4           | Q

One-Hot Encoded:
PassengerID | Embarked_C | Embarked_Q | Embarked_S
1           | 0          | 0          | 1
2           | 1          | 0          | 0
3           | 0          | 0          | 1
4           | 0          | 1          | 0
```

**Note**: We create (n-1) columns for n categories to avoid multicollinearity (dummy variable trap).
- 3 ports → 2 columns
- If Embarked_C=0 and Embarked_Q=0, we know it must be Embarked_S

**How it works**:
```python
import pandas as pd

embarked_encoded = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)
# drop_first=True removes one column to avoid redundancy
```

**Pros**:
- No false ordering
- Works well with linear models
- Accurate representation

**Cons**:
- Creates many columns (curse of dimensionality)
- Memory intensive with high cardinality

### 3.3 Ordinal Encoding (Manual Mapping)

**What is it?**
Manually assigning numbers based on domain knowledge of the order.

**Example - Customer Satisfaction**:
```python
satisfaction_map = {
    'Very Unsatisfied': 1,
    'Unsatisfied': 2,
    'Neutral': 3,
    'Satisfied': 4,
    'Very Satisfied': 5
}

df['Satisfaction_Encoded'] = df['Satisfaction'].map(satisfaction_map)
```

**Example - Passenger Class** (already numerical but could be labeled):
```python
pclass_map = {1: 'First', 2: 'Second', 3: 'Third'}
# Or reverse: {'First': 1, 'Second': 2, 'Third': 3}
```

**Why use manual mapping?**
- Full control over the encoding
- Can assign meaningful values
- Preserves domain knowledge

### 3.4 Frequency Encoding

**What is it?**
Replacing categories with their frequency (how often they appear).

**Example - Title**:
```
Titles in dataset:
Mr: 517 times (58%)
Miss: 182 times (20%)
Mrs: 125 times (14%)
Master: 40 times (4%)
Rare: 27 times (3%)

Frequency Encoding:
'Mr' → 0.58
'Miss' → 0.20
'Mrs' → 0.14
'Master' → 0.04
'Rare' → 0.03
```

**How it works**:
```python
title_freq = df['Title'].value_counts(normalize=True)
df['Title_Frequency'] = df['Title'].map(title_freq)
```

**When useful?**
- High cardinality features (many categories)
- When frequency itself is informative
- Reduces dimensionality compared to one-hot

**Caution**: Different categories with same frequency get same encoding!

### 3.5 Target Encoding (Mean Encoding)

**What is it?**
Replacing categories with the mean of the target variable for that category.

**Example - Title vs Survival**:
```
Calculate survival rate per title:
Mr: 157/517 survived = 0.30 (30%)
Miss: 127/182 survived = 0.70 (70%)
Mrs: 99/125 survived = 0.79 (79%)
Master: 23/40 survived = 0.58 (58%)

Target Encoding:
'Mr' → 0.30
'Miss' → 0.70
'Mrs' → 0.79
'Master' → 0.58
```

**How it works**:
```python
target_mean = df.groupby('Title')['Survived'].mean()
df['Title_Target_Enc'] = df['Title'].map(target_mean)
```

**Pros**:
- Very powerful
- Captures relationship with target
- Works with high cardinality

**Cons**:
- **HIGH RISK OF OVERFITTING** - must use cross-validation
- **DATA LEAKAGE** if not done carefully
- Doesn't work for unseen categories

**Best Practice**: Use k-fold target encoding to prevent leakage:
1. Split data into k folds
2. For each fold, calculate target mean using OTHER folds only
3. Apply to current fold

---

## 4. Feature Scaling & Normalization

**Why scale?**
Different features have different ranges:
- Age: 0-80
- Fare: 0-500

Without scaling, models treat large values as more important!

### 4.1 Standardization (Z-score Normalization)

**What is it?**
Transform data to have mean=0 and standard deviation=1.

**Formula**:
```
z = (x - μ) / σ

Where:
μ = mean
σ = standard deviation
```

**Example**:
```
Original Ages: [20, 30, 40, 50, 60]
Mean (μ) = 40
Std (σ) = 14.14

Standardized:
20 → (20-40)/14.14 = -1.41
30 → (30-40)/14.14 = -0.71
40 → (40-40)/14.14 = 0.00
50 → (50-40)/14.14 = 0.71
60 → (60-40)/14.14 = 1.41
```

**Interpretation**:
- z = 0: value is at the mean
- z = 1: value is 1 standard deviation above mean
- z = -2: value is 2 standard deviations below mean

**When to use**:
- Linear Regression
- Logistic Regression
- Neural Networks
- SVM
- K-Means Clustering
- PCA

**Code**:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
```

**Pros**:
- Preserves outlier information
- Works well when data is normally distributed
- Most common scaling method

**Cons**:
- Sensitive to outliers (they affect mean and std)
- Not bounded (can have values < 0 or > 1)

### 4.2 Min-Max Normalization

**What is it?**
Scales data to a fixed range, typically [0, 1].

**Formula**:
```
x_scaled = (x - min) / (max - min)
```

**Example**:
```
Original Fares: [10, 30, 50, 70, 90]
Min = 10
Max = 90
Range = 80

Normalized:
10 → (10-10)/80 = 0.00
30 → (30-10)/80 = 0.25
50 → (50-10)/80 = 0.50
70 → (70-10)/80 = 0.75
90 → (90-10)/80 = 1.00
```

**When to use**:
- Neural Networks (inputs should be 0-1)
- Image processing
- When you need bounded range
- K-Nearest Neighbors

**Code**:
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
```

**Pros**:
- Bounded range
- Preserves zero values
- Easy to interpret (0=min, 1=max)

**Cons**:
- Very sensitive to outliers
- New data outside training range causes issues

### 4.3 Robust Scaling

**What is it?**
Uses median and Interquartile Range (IQR) instead of mean and standard deviation. Resistant to outliers.

**Formula**:
```
x_scaled = (x - median) / IQR

Where:
IQR = Q3 - Q1 (75th percentile - 25th percentile)
```

**Example**:
```
Fares: [10, 15, 20, 500, 25]  (500 is outlier)

Q1 (25th percentile) = 15
Q3 (75th percentile) = 25
IQR = 25 - 15 = 10
Median = 20

Robust Scaled:
10 → (10-20)/10 = -1.0
15 → (15-20)/10 = -0.5
20 → (20-20)/10 = 0.0
500 → (500-20)/10 = 48.0  (still high but less extreme)
25 → (25-20)/10 = 0.5

vs StandardScaler (heavily affected by 500):
Mean = 114
Std = 196
500 → (500-114)/196 = 1.97
10 → (10-114)/196 = -0.53
```

**When to use**:
- Data has outliers
- Financial data
- Real-world messy data

**Code**:
```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data)
```

**Pros**:
- Robust to outliers
- Better for skewed distributions

**Cons**:
- Not bounded
- Less commonly used

### 4.4 Log Transformation

**What is it?**
Applying logarithm to compress large values and expand small values.

**Formula**:
```
x_transformed = log(x + 1)
# Adding 1 to handle zeros: log(0) is undefined
```

**Example**:
```
Original Fares: [1, 10, 100, 1000]

Log Transformed:
1 → log(2) = 0.69
10 → log(11) = 2.40
100 → log(101) = 4.62
1000 → log(1001) = 6.91

Notice: Differences between values are now more uniform!
```

**Visual Impact on Distribution**:
```
Original (Right-skewed):
|
|        *
|      * *
|    * * *
|  * * * * * * * * * *
+----------------------
0  10 20 30...    500

Log-transformed (More normal):
|    * * *
|  * * * * *
| * * * * * *
|* * * * * * *
+----------------------
```

**When to use**:
- Right-skewed distributions (long tail on right)
- Data spanning several orders of magnitude
- Prices, incomes, populations
- Count data

**Code**:
```python
import numpy as np
df['Fare_Log'] = np.log1p(df['Fare'])  # log1p = log(1 + x)
```

**Pros**:
- Reduces skewness
- Handles exponential relationships
- Makes data more normal
- Reduces impact of outliers

**Cons**:
- Only for positive values
- Interpretation is harder
- Inverse transform needed for predictions

---

## 5. Feature Selection Methods

**Why select features?**
- Reduce overfitting
- Improve model performance
- Reduce training time
- Improve interpretability
- Reduce storage

### 5.1 Correlation-Based Selection

**What is it?**
Selecting features based on their correlation with the target variable.

**Correlation Coefficient** ranges from -1 to +1:
- +1: Perfect positive correlation
- 0: No correlation
- -1: Perfect negative correlation

**Example**:
```
Correlations with Survival:
Sex: 0.54 (strong positive - females survived more)
Pclass: -0.34 (negative - lower class = lower survival)
Fare: 0.26 (positive - higher fare = higher survival)
Age: -0.08 (weak negative)

Decision: Keep Sex, Pclass, Fare; maybe drop Age
```

**Threshold approach**:
```python
correlation_threshold = 0.1
selected_features = correlation_matrix[abs(correlation_matrix) > threshold]
```

**Also check**: Multicollinearity (features correlated with each other)
```
If Fare and Pclass are highly correlated (0.9):
- They provide redundant information
- Drop one to reduce multicollinearity
```

**Code**:
```python
# Correlation with target
correlation = df.corrwith(df['target']).abs()

# Correlation matrix (between features)
corr_matrix = df.corr()
```

**Pros**:
- Fast and simple
- Easy to interpret
- Identifies linear relationships

**Cons**:
- Only captures linear relationships
- Doesn't consider feature combinations

### 5.2 Chi-Squared Test

**What is it?**
Statistical test measuring independence between categorical features and target.

**Null Hypothesis**: Feature and target are independent (no relationship)

**Chi-squared statistic**: Higher value = stronger relationship

**Example - Sex vs Survival**:
```
Contingency Table:
              Survived  Died
Male          109      468
Female        233       81

Chi-squared = 260.7 (very high!)
p-value < 0.001
Conclusion: Strong relationship between sex and survival
```

**How it works**:
1. Calculate expected frequencies (if independent)
2. Compare with observed frequencies
3. Sum squared differences

**Formula**:
```
χ² = Σ [(Observed - Expected)² / Expected]
```

**When to use**:
- Categorical features
- Classification problems
- Quick initial screening

**Code**:
```python
from sklearn.feature_selection import SelectKBest, chi2

# Select top 5 features
selector = SelectKBest(chi2, k=5)
X_selected = selector.fit_transform(X, y)
```

**Requirements**:
- All values must be non-negative
- Works with categorical data

**Pros**:
- Fast
- Theoretically sound
- Works well for categorical data

**Cons**:
- Requires non-negative values
- Only univariate (one feature at a time)

### 5.3 Mutual Information

**What is it?**
Measures how much knowing one variable reduces uncertainty about another. Captures non-linear relationships.

**Information Theory Concept**:
```
Mutual Information = 
How much does knowing feature X tell us about target Y?

MI = 0: Independent (knowing X tells nothing about Y)
MI > 0: Some relationship
Higher MI = Stronger relationship
```

**Example**:
```
Feature: Age
Target: Survival

If survival rate is same for all ages → MI = 0
If children have 90% survival, adults 40% → MI > 0
The more different the rates across age groups, the higher MI
```

**Advantage over correlation**:
```
Correlation: Only linear
X: [1, 2, 3, 4, 5]
Y: [1, 4, 9, 16, 25]  (Y = X²)
Pearson correlation: 0.97 (seems strong but not perfect)
Mutual Information: Can capture the perfect quadratic relationship!
```

**Code**:
```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y)
# Higher scores = more informative features
```

**When to use**:
- Non-linear relationships
- When correlation fails
- Both categorical and continuous

**Pros**:
- Captures non-linear relationships
- No assumptions about data distribution
- Works with any type of data

**Cons**:
- Computationally more expensive
- Harder to interpret than correlation

### 5.4 Recursive Feature Elimination (RFE)

**What is it?**
Iteratively builds models and removes the least important feature each time.

**Algorithm**:
```
1. Train model with all features
2. Rank features by importance
3. Remove least important feature
4. Repeat until desired number of features

Example with 10 features, selecting 5:

Round 1: All 10 features → Remove Feature_7 (least important)
Round 2: 9 features → Remove Feature_3
Round 3: 8 features → Remove Feature_1
Round 4: 7 features → Remove Feature_9
Round 5: 6 features → Remove Feature_5
Final: 5 features remain
```

**Visual representation**:
```
Start: [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10]
       Importance: [0.05, 0.15, 0.03, 0.20, 0.08, 0.12, 0.02, 0.18, 0.04, 0.13]
       
After Round 1: [F1, F2, F3, F4, F5, F6, F8, F9, F10]  (removed F7)
After Round 2: [F1, F2, F4, F5, F6, F8, F9, F10]       (removed F3)
...
Final: [F2, F4, F6, F8, F10]
```

**Code**:
```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

estimator = RandomForestClassifier()
rfe = RFE(estimator=estimator, n_features_to_select=5)
rfe.fit(X, y)

selected_features = X.columns[rfe.support_]
```

**When to use**:
- When you want optimal subset
- When feature interactions matter
- With any model that provides importance

**Pros**:
- Considers feature interactions
- Works with any estimator
- Often yields optimal subset

**Cons**:
- Computationally expensive (trains many models)
- Can be slow with many features
- Results depend on chosen estimator

### 5.5 Tree-Based Feature Importance

**What is it?**
Using importance scores from tree-based models (Random Forest, Gradient Boosting, XGBoost).

**How trees calculate importance**:
```
For each feature, measure:
- How much it decreases impurity (Gini or entropy)
- How often it's used for splitting
- How much it improves predictions

Higher importance = More useful for predictions
```

**Example**:
```
Random Forest with 100 trees on Titanic data:

Feature Importance:
Sex: 0.31        (Most important - used in many splits, high predictive power)
Fare: 0.18       
Age: 0.16        
Pclass: 0.14     
FamilySize: 0.08
SibSp: 0.05
Parch: 0.04
Embarked: 0.02   (Least important)

Decision: Keep top features, consider dropping low-importance ones
```

**Code**:
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
```

**When to use**:
- Quick feature screening
- Non-linear relationships
- High-dimensional data

**Pros**:
- Fast to compute
- Captures non-linear relationships
- Handles feature interactions
- No need for feature scaling

**Cons**:
- Biased toward high-cardinality features
- May favor correlated features
- Different models give different importances

### 5.6 L1 Regularization (Lasso)

**What is it?**
Adds penalty to model that forces some feature coefficients to exactly zero.

**Regularization concept**:
```
Normal Linear Regression:
Minimize: Σ(y_actual - y_predicted)²

L1 Regularization (Lasso):
Minimize: Σ(y_actual - y_predicted)² + λ Σ|coefficients|
                                        ↑
                                    Penalty term
```

**How it selects features**:
```
Start with all features:
Feature1: coefficient = 0.8
Feature2: coefficient = 0.0001  → Pushed to ZERO (feature eliminated)
Feature3: coefficient = 0.5
Feature4: coefficient = -0.0003 → Pushed to ZERO (feature eliminated)
Feature5: coefficient = 0.9

Final model uses: Feature1, Feature3, Feature5
```

**Lambda (λ) parameter**:
```
λ = 0: No regularization (all features kept)
λ = small: Weak penalty (most features kept)
λ = large: Strong penalty (many features zeroed out)
λ = very large: All features zeroed out
```

**Example with Titanic**:
```
All features:
[Sex, Age, Fare, Pclass, SibSp, Parch, Embarked]

After L1 with λ=0.1:
Sex: 0.45      ✓ Kept
Age: 0.12      ✓ Kept
Fare: 0.28     ✓ Kept
Pclass: -0.19  ✓ Kept
SibSp: 0.00    ✗ Dropped
Parch: 0.00    ✗ Dropped
Embarked: 0.00 ✗ Dropped
```

**Code**:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
lasso.fit(X, y)

selector = SelectFromModel(lasso, prefit=True)
X_selected = selector.transform(X)
```

**When to use**:
- High-dimensional data (many features)
- Suspected redundant features
- Need sparse model (few features)

**Pros**:
- Automatic feature selection
- Produces sparse models
- Prevents overfitting

**Cons**:
- Only works with linear models
- Requires feature scaling
- Sensitive to multicollinearity

---

## 6. Advanced Feature Engineering

### 6.1 Aggregation Features

**What is it?**
Creating features based on group-level statistics.

**Concept**: "How does this individual compare to their group?"

**Example 1 - Mean Fare per Class**:
```
Step 1: Calculate average fare for each class
Pclass 1: Mean Fare = $84.15
Pclass 2: Mean Fare = $20.66
Pclass 3: Mean Fare = $13.68

Step 2: Add this as a feature for each passenger
PassengerID | Pclass | Fare | MeanFare_PerClass
1           | 1      | 71.3 | 84.15
2           | 3      | 7.25 | 13.68
3           | 1      | 90.0 | 84.15
4           | 2      | 21.0 | 20.66
```

**Why useful?**
- Passenger paying $71 in 1st class vs $90 gives context
- Passenger paying $7.25 in 3rd class (below average) vs $20 (above average)

**Common aggregations**:
```python
# By group statistics
df['Mean_Age_PerClass'] = df.groupby('Pclass')['Age'].transform('mean')
df['Median_Fare_PerEmbarked'] = df.groupby('Embarked')['Fare'].transform('median')
df['Max_FamilySize_PerClass'] = df.groupby('Pclass')['FamilySize'].transform('max')
df['Count_PerTitle'] = df.groupby('Title')['PassengerId'].transform('count')
```

**Types of aggregations**:
- **Central tendency**: mean, median, mode
- **Dispersion**: std, variance, range, IQR
- **Position**: min, max, percentiles
- **Count**: count, nunique

### 6.2 Ratio Features

**What is it?**
Comparing individual values to group statistics through division.

**Example - Fare Ratio**:
```
PassengerID | Fare | MeanFare_PerClass | FareToMeanRatio
1           | 80   | 84.15             | 0.95 (paying 5% less than avg)
2           | 10   | 13.68             | 0.73 (paying 27% less than avg)
3           | 120  | 84.15             | 1.43 (paying 43% MORE than avg)
```

**Interpretation**:
- Ratio < 1: Below group average
- Ratio = 1: At group average  
- Ratio > 1: Above group average

**Why useful?**
- Normalizes values by context
- "Paying $80 in 1st class" is relative - is it cheap or expensive?
- Ratio tells us: 0.95 means slightly below average for 1st class
- Creates scale-invariant features

**More examples**:
```python
# Individual vs group comparison
df['AgeToMeanRatio'] = df['Age'] / df['MeanAge_PerClass']
df['FareToMedianRatio'] = df['Fare'] / df['MedianFare_PerClass']

# Efficiency metrics
df['CostEfficiency'] = df['Fare'] / df['FamilySize']  # Cost per person
df['AgeEfficiency'] = df['Age'] / (df['Pclass'] + 1)  # Normalized age
```

### 6.3 Clustering-Based Features

**What is it?**
Using unsupervised learning (clustering) to create new features by grouping similar observations.

**Concept**: "Which passengers are similar to each other?"

**K-Means Clustering Example**:
```
Step 1: Select features for clustering
Features: [Age, Fare, FamilySize]

Step 2: Cluster passengers into 3 groups

Cluster 0 (Young, Cheap, Alone):
- Age: 20-35
- Fare: $0-20
- FamilySize: 1
Examples: Young solo travelers in 3rd class

Cluster 1 (Middle-aged, Moderate, Small Family):
- Age: 30-50
- Fare: $20-80
- FamilySize: 2-4
Examples: Families in 2nd class

Cluster 2 (Varied age, Expensive, Various):
- Age: Any
- Fare: $80+
- FamilySize: Any
Examples: 1st class passengers

Step 3: Add cluster ID as feature
PassengerID | Age | Fare | FamilySize | Cluster
1           | 22  | 7.25 | 1          | 0
2           | 38  | 71.3 | 2          | 1
3           | 26  | 7.92 | 1          | 0
4           | 35  | 53.1 | 3          | 1
```

**Why useful?**
- Captures complex patterns
- Groups similar passengers
- Model can learn "cluster behavior"
- Dimensionality reduction (3 features → 1 cluster ID)

**Code**:
```python
from sklearn.cluster import KMeans

features_for_clustering = ['Age', 'Fare', 'FamilySize']
X_cluster = df[features_for_clustering].fillna(0)

kmeans = KMeans(n_clusters=3, random_state=42)
df['PassengerCluster'] = kmeans.fit_predict(X_cluster)
```

**Clustering algorithms**:
- **K-Means**: Fast, spherical clusters
- **DBSCAN**: Density-based, handles noise
- **Hierarchical**: Creates cluster tree
- **Gaussian Mixture**: Probabilistic clusters

**Best practices**:
- Normalize features before clustering
- Try different numbers of clusters
- Visualize clusters to validate
- Combine with original features

### 6.4 DateTime Features

**What is it?**
Extracting meaningful components and patterns from datetime columns.

**Components to extract**:

```python
# From datetime: 2012-04-15 14:30:00

Year: 2012
Month: 4 (April)
Day: 15
Hour: 14 (2 PM)
Minute: 30
Day of Week: 0 (Monday)
Day of Year: 106
Week of Year: 15
Quarter: 2 (Q2)
Is Weekend: 0 (No, it's Monday)
Is Month End: 0
Is Month Start: 0
```

**Cyclical features** (circular nature of time):
```python
# Problem: December (12) and January (1) are actually close!
# Solution: Sine/Cosine encoding

Month: 12 → sin(2π × 12/12) = 0, cos(2π × 12/12) = 1
Month: 1  → sin(2π × 1/12) = 0.5, cos(2π × 1/12) = 0.87

# These values show 12 and 1 are close in circular space
```

**Derived features**:
```python
# Time since event
DaysSince_LastPurchase = (CurrentDate - LastPurchaseDate).days

# Time until event  
DaysUntil_Christmas = (ChristmasDate - CurrentDate).days

# Duration
Trip_Duration_Hours = (EndTime - StartTime).seconds / 3600

# Business vs Weekend
IsBusinessHour = (Hour >= 9) & (Hour <= 17) & (DayOfWeek < 5)
```

**Example - Sales Data**:
```
Date: 2024-01-15 18:30:00

Extracted features:
- Year: 2024
- Month: 1 (January - possibly low sales)
- DayOfWeek: 1 (Monday - back to work)
- Hour: 18 (evening rush hour)
- IsWeekend: 0
- IsHoliday: 0
- DaysAfter_NewYear: 15 (New Year sales ended)
- Quarter: 1
- IsEndOfMonth: 0 (people have money)
```

**Seasonality**:
```python
# Retail example
Month 12 (December): High sales (Christmas)
Month 1 (January): Low sales (post-holiday)
Month 11 (November): High sales (Black Friday)
```

**Code**:
```python
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Hour'] = df['Date'].dt.hour
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
df['Quarter'] = df['Date'].dt.quarter

# Cyclical encoding
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
```

### 6.5 Outlier Detection Features

**What is it?**
Creating binary or continuous features that flag outliers.

**Why create outlier features?**
- Outliers might be errors OR important patterns
- Instead of removing, flag them and let model decide
- "Is this a fraudulent transaction?" might correlate with outliers

**Method 1: Z-Score**
```
Z-score = (Value - Mean) / StdDev

If |Z-score| > 3: Likely outlier (99.7% rule)

Example - Age:
Ages: [25, 30, 35, 28, 95]
Mean: 42.6, Std: 27.1

Z-scores:
25 → |−0.65| < 3 ✓ Normal
30 → |−0.46| < 3 ✓ Normal
35 → |−0.28| < 3 ✓ Normal
28 → |−0.54| < 3 ✓ Normal
95 → |1.93| < 3 ✓ Normal (but would be flagged with stricter threshold)

With threshold of 2:
95 → Outlier!
```

**Code**:
```python
from scipy import stats

df['Age_Zscore'] = np.abs(stats.zscore(df['Age'].fillna(df['Age'].median())))
df['Age_IsOutlier'] = (df['Age_Zscore'] > 3).astype(int)
```

**Method 2: IQR (Interquartile Range)**
```
IQR Method:
Q1 = 25th percentile
Q3 = 75th percentile  
IQR = Q3 - Q1

Lower fence = Q1 - 1.5 × IQR
Upper fence = Q3 + 1.5 × IQR

Anything outside fences = Outlier

Example - Fare:
Fares sorted: [7, 8, 10, 13, 20, 50, 500]

Q1 = 8
Q3 = 50
IQR = 42

Lower fence = 8 - 1.5(42) = -55 (no lower outliers)
Upper fence = 50 + 1.5(42) = 113

500 > 113 → Outlier!
```

**Code**:
```python
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

df['Fare_IsOutlier'] = ((df['Fare'] < lower_fence) | 
                         (df['Fare'] > upper_fence)).astype(int)
```

**Method 3: Isolation Forest**
```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.1, random_state=42)
df['Anomaly_Score'] = iso_forest.fit_predict(df[numerical_features])
# -1 = outlier, 1 = normal
```

**Use cases**:
- **Fraud detection**: Flag unusual transactions
- **Quality control**: Flag defective products
- **Medical**: Flag unusual patient readings
- **Finance**: Flag unusual stock movements

**Best practices**:
- Don't automatically remove outliers
- Create outlier flags as features
- Let model learn whether outliers matter
- Different thresholds for different domains

### 6.6 Interaction Terms

**What is it?**
Creating features that capture how two or more variables interact or combine their effects.

**Why needed?**
Sometimes the effect of one variable depends on another.

**Example - Real Estate**:
```
Simple model: Price = f(Size, Location)

But the relationship isn't additive!

Interaction: Size × Location matters:
- 1000 sqft downtown = $500k
- 1000 sqft suburbs = $200k
- 2000 sqft downtown = $1.2M (not just 2× $500k!)
- 2000 sqft suburbs = $380k

The value of size depends on location!
```

**Titanic example - Gender × Class**:
```
Survival rates:

Women in 1st class: 97% survived
Women in 3rd class: 50% survived
Men in 1st class: 37% survived
Men in 3rd class: 14% survived

The effect of gender depends on class!
The effect of class depends on gender!

Feature: Pclass_Sex
- "1_Female": 0.97 survival rate
- "3_Male": 0.14 survival rate
```

**Types of interactions**:

**1. Categorical × Categorical**:
```python
df['Gender_Class'] = df['Sex'] + '_' + df['Pclass'].astype(str)
# Creates: Male_1, Male_2, Male_3, Female_1, Female_2, Female_3
```

**2. Numerical × Numerical**:
```python
df['Age_Fare'] = df['Age'] * df['Fare']
# Young + cheap vs old + expensive might have different patterns
```

**3. Categorical × Numerical**:
```python
df['IsMale_Age'] = (df['Sex'] == 'male').astype(int) * df['Age']
# Age matters differently for males vs females
```

**4. Polynomial interactions**:
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
# Creates: X1, X2, X1², X2², X1×X2
```

**When useful?**:
- Non-linear relationships
- Conditional effects
- Different patterns in subgroups

**Caution**: Can create MANY features!
- 10 features → 55 interactions (with degree=2)
- 20 features → 210 interactions
- Risk of overfitting

---

## 7. Complete Feature Engineering Workflow

### Step-by-Step Process

**1. Understand the Data**
```
- What does each column represent?
- What are the data types?
- What's the business context?
- What patterns might exist?
```

**2. Handle Missing Data**
```
- Analyze missing patterns
- Decide: impute, drop, or flag
- Create missing indicators if informative
```

**3. Create Basic Features**
```
- Combinations (sums, products)
- Ratios
- Binning
- Text extraction
```

**4. Encode Categorical Variables**
```
- Ordinal → Label encoding
- Nominal → One-hot encoding
- High cardinality → Target/frequency encoding
```

**5. Scale Numerical Features**
```
- Standardization for normal distributions
- Min-Max for bounded ranges
- Robust scaling for outliers
- Log transform for skewed data
```

**6. Create Advanced Features**
```
- Aggregations
- Interactions
- Clustering
- Polynomial features
```

**7. Select Features**
```
- Remove low-variance features
- Remove highly correlated features
- Use statistical tests
- Use model-based selection
```

**8. Validate**
```
- Cross-validation
- Check for data leakage
- Monitor model performance
- Iterate based on results
```

---

## 8. Common Pitfalls to Avoid

### Data Leakage
**Problem**: Using information from the test set during training

**Examples**:
```python
# WRONG: Fit on entire dataset
scaler.fit(df)  # Includes test data!
df_scaled = scaler.transform(df)
train_test_split(df_scaled)

# CORRECT: Fit only on training data
X_train, X_test = train_test_split(df)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Target Leakage
**Problem**: Using features that wouldn't be available at prediction time

**Example**:
```
Predicting loan default:
- Days_Since_Last_Payment ✗ (only known after default)
- Credit_Score ✓ (known at application)
- Total_Payments_Made ✗ (only known after loan period)
```

### Overfitting with Too Many Features
**Problem**: Creating hundreds of features that memorize training data

**Solution**:
- Use feature selection
- Regularization
- Cross-validation
- Monitor train vs validation performance

### Not Handling Unseen Categories
**Problem**: New categories in test data that weren't in training

**Example**:
```
Training data Titles: ['Mr', 'Mrs', 'Miss']
Test data has: 'Dr'

Solution: Map rare titles to 'Other' category
```

---

## 9. Summary Checklist

**Before Feature Engineering**:
- [ ] Understand business problem
- [ ] Explore data distribution
- [ ] Identify data types
- [ ] Check missing values
- [ ] Check for duplicates

**During Feature Engineering**:
- [ ] Handle missing values appropriately
- [ ] Create domain-specific features
- [ ] Encode categorical variables correctly
- [ ] Scale numerical features
- [ ] Create interaction terms if needed
- [ ] Remove redundant features

**After Feature Engineering**:
- [ ] Validate on holdout set
- [ ] Check for data leakage
- [ ] Document all transformations
- [ ] Save preprocessing pipelines
- [ ] Monitor model performance
- [ ] Iterate and improve

**Remember**: Feature engineering is both art and science. It requires:
- Domain knowledge
- Statistical understanding
- Creativity
- Experimentation
- Validation

The best features often come from deep understanding of the problem domain combined with data exploration and iterative testing.
# Indian Engineering Student Placement - ML Pipeline Notes

## Dataset Overview

**Dataset Name:** Indian Engineering Student Placement Dataset  
**Total Records:** 5,000 students  
**Total Features:** 25 (23 features + 1 ID + 1 target)  
**Target Variable:** `salary_lpa` (Salary in Lakhs Per Annum)  
**Problem Type:** Regression (Predicting continuous salary values)

### Dataset Composition
- **Numerical Features:** 16 features
- **Categorical Features:** 8 features
- **Missing Values:** Only `extracurricular_involvement` has 1,006 missing values (20.12%)
- **Duplicate Rows:** 0

### Target Variable Statistics
- **Placement Rate:** 86.06% (4,303 placed out of 5,000)
- **Salary Range:** 0 - 20 LPA
- **Mean Salary:** 13.90 LPA
- **Median Salary:** 15.76 LPA
- **Note:** 697 students (13.94%) have 0 salary (Not Placed)

---

## 0. Data Preprocessing

### Step 1: Load and Merge Data
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load datasets
students = pd.read_csv('indian_engineering_student_placement.csv')
targets = pd.read_csv('placement_targets.csv')

# Merge on Student_ID
df = students.merge(targets, on='Student_ID')
```

### Step 2: Handle Missing Values

**Missing Value Analysis:**
- `extracurricular_involvement`: 1,006 missing (20.12%)
- All other columns: Complete

**Treatment Options:**

**Option 1: Imputation with Mode (Recommended)**
```python
# Fill with mode (most frequent category)
df['extracurricular_involvement'].fillna(
    df['extracurricular_involvement'].mode()[0], 
    inplace=True
)
```

**Option 2: Create "Unknown" Category**
```python
# Create a separate category for missing values
df['extracurricular_involvement'].fillna('Unknown', inplace=True)
```

**Option 3: Drop Missing Rows**
```python
# Only if you can afford to lose 20% of data
df.dropna(subset=['extracurricular_involvement'], inplace=True)
```

### Step 3: Handle Outliers

**Outlier Detection Results:**
- `cgpa`: 23 outliers (0.46%)
- `salary_lpa`: 697 outliers (13.94%) - These are "Not Placed" students with 0 salary

**Treatment Strategy:**

**For CGPA outliers (Very Low CGPAs):**
```python
# Option 1: Keep them (they represent real poor performers)
# Option 2: Cap at lower bound
Q1 = df['cgpa'].quantile(0.25)
Q3 = df['cgpa'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR

df.loc[df['cgpa'] < lower_bound, 'cgpa'] = lower_bound
```

**For Salary outliers (0 salary = Not Placed):**
```python
# Keep them as they represent genuine cases of non-placement
# Alternative: Create separate classification model for placement first
```

### Step 4: Remove Unnecessary Columns

```python
# Drop Student_ID (not a predictive feature)
df.drop('Student_ID', axis=1, inplace=True)

# Drop placement_status (redundant with salary_lpa)
# If salary > 0, student is placed
df.drop('placement_status', axis=1, inplace=True)
```

### Step 5: Data Type Verification

```python
# Check data types
print(df.dtypes)

# Convert categorical columns to category dtype for better memory management
categorical_cols = ['gender', 'branch', 'part_time_job', 'family_income_level', 
                   'city_tier', 'internet_access', 'extracurricular_involvement']

for col in categorical_cols:
    df[col] = df[col].astype('category')
```

### Step 6: Check for Data Consistency

```python
# Check value ranges
print(df.describe())

# Verify CGPA is between 0-10
assert df['cgpa'].between(0, 10).all(), "CGPA out of range"

# Verify percentages are between 0-100
assert df['tenth_percentage'].between(0, 100).all(), "10th % out of range"
assert df['twelfth_percentage'].between(0, 100).all(), "12th % out of range"
assert df['attendance_percentage'].between(0, 100).all(), "Attendance % out of range"

# Verify skill ratings are between 1-5
assert df['coding_skill_rating'].between(1, 5).all(), "Coding skill rating out of range"
```

---

## 1. Feature Engineering

### Step 1: Create Interaction Features

**Academic Performance Composite:**
```python
# Average academic performance
df['avg_academic_score'] = (df['cgpa'] * 10 + 
                            df['tenth_percentage'] + 
                            df['twelfth_percentage']) / 3

# Weighted academic score (CGPA has more weight)
df['weighted_academic'] = (df['cgpa'] * 10 * 0.5 + 
                           df['tenth_percentage'] * 0.25 + 
                           df['twelfth_percentage'] * 0.25)
```

**Skills Composite Score:**
```python
# Overall skill rating
df['overall_skill_score'] = (df['coding_skill_rating'] + 
                             df['communication_skill_rating'] + 
                             df['aptitude_skill_rating']) / 3

# Technical skills indicator
df['strong_technical'] = (df['coding_skill_rating'] >= 4).astype(int)
```

**Practical Experience Score:**
```python
# Total practical exposure
df['practical_experience'] = (df['projects_completed'] + 
                              df['internships_completed'] + 
                              df['hackathons_participated'])

# Practical experience per year (assuming 4-year degree)
df['practical_per_year'] = df['practical_experience'] / 4
```

**Effort Indicators:**
```python
# Study-to-sleep ratio (balance indicator)
df['study_sleep_ratio'] = df['study_hours_per_day'] / df['sleep_hours']

# Effort consistency score
df['effort_score'] = df['study_hours_per_day'] * df['attendance_percentage'] / 100
```

### Step 2: Create Polynomial Features

```python
# Squared features for important continuous variables
df['cgpa_squared'] = df['cgpa'] ** 2
df['study_hours_squared'] = df['study_hours_per_day'] ** 2
df['attendance_squared'] = df['attendance_percentage'] ** 2
```

### Step 3: Create Binary Flags

```python
# High performer flag
df['high_cgpa'] = (df['cgpa'] >= 8.5).astype(int)
df['excellent_attendance'] = (df['attendance_percentage'] >= 80).astype(int)

# Zero backlogs flag
df['no_backlogs'] = (df['backlogs'] == 0).astype(int)

# Active student flag
df['active_student'] = ((df['projects_completed'] >= 5) | 
                        (df['hackathons_participated'] >= 3)).astype(int)

# Certifications flag
df['has_certifications'] = (df['certifications_count'] > 0).astype(int)
```

### Step 4: Create Categorical Aggregations

```python
# Branch-wise average salary encoding (do this after train-test split!)
# This will be done during feature selection
```

### Step 5: Create Ratio Features

```python
# Projects to internships ratio
df['project_internship_ratio'] = df['projects_completed'] / (df['internships_completed'] + 1)

# Certifications per hackathon
df['cert_per_hackathon'] = df['certifications_count'] / (df['hackathons_participated'] + 1)

# Coding to communication skill ratio
df['coding_comm_ratio'] = df['coding_skill_rating'] / (df['communication_skill_rating'] + 1)
```

### Step 6: Binning Continuous Variables

```python
# CGPA bins
df['cgpa_category'] = pd.cut(df['cgpa'], 
                             bins=[0, 6, 7, 8, 9, 10],
                             labels=['Poor', 'Average', 'Good', 'Very Good', 'Excellent'])

# Stress level categories
df['stress_category'] = pd.cut(df['stress_level'],
                               bins=[0, 3, 6, 10],
                               labels=['Low', 'Medium', 'High'])
```

### Step 7: Domain-Specific Features

```python
# CS/IT branch flag (typically higher placements)
df['is_cs_it'] = df['branch'].isin(['CSE', 'IT']).astype(int)

# Tier 1 city advantage
df['tier1_city'] = (df['city_tier'] == 'Tier 1').astype(int)

# Resource access score
df['resource_access'] = ((df['internet_access'] == 'Yes').astype(int) + 
                         (df['family_income_level'].isin(['Medium', 'High'])).astype(int))
```

---

## 2. Feature Selection

### Step 1: Separate Features and Target

```python
# Define target
y = df['salary_lpa']

# Define features (exclude target)
X = df.drop('salary_lpa', axis=1)
```

### Step 2: Train-Test Split (Before Encoding!)

```python
from sklearn.model_selection import train_test_split

# Split data (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
```

### Step 3: Encode Categorical Variables

**For Linear Regression, we need numerical features only.**

**Label Encoding for Binary/Ordinal Features:**
```python
from sklearn.preprocessing import LabelEncoder

# Binary encoding
binary_cols = ['gender', 'part_time_job', 'internet_access']
le_dict = {}

for col in binary_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    le_dict[col] = le
```

**One-Hot Encoding for Nominal Features:**
```python
# Nominal categorical columns
nominal_cols = ['branch', 'family_income_level', 'city_tier', 'extracurricular_involvement']

# One-hot encode
X_train = pd.get_dummies(X_train, columns=nominal_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=nominal_cols, drop_first=True)

# Ensure train and test have same columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
```

**Target Encoding (Use with Caution - Risk of Data Leakage):**
```python
# Alternative: Target encoding for high-cardinality features
# Only use training data to compute means
branch_salary_mean = X_train.merge(y_train, left_index=True, right_index=True) \
                            .groupby('branch')['salary_lpa'].mean()

X_train['branch_encoded'] = X_train['branch'].map(branch_salary_mean)
X_test['branch_encoded'] = X_test['branch'].map(branch_salary_mean)
```

### Step 4: Handle Encoded Categorical Binning Features

```python
# If you created categorical bins, encode them
if 'cgpa_category' in X_train.columns:
    cgpa_mapping = {'Poor': 1, 'Average': 2, 'Good': 3, 'Very Good': 4, 'Excellent': 5}
    X_train['cgpa_category'] = X_train['cgpa_category'].map(cgpa_mapping)
    X_test['cgpa_category'] = X_test['cgpa_category'].map(cgpa_mapping)
```

### Step 5: Correlation Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Compute correlation matrix
correlation_matrix = X_train.corr()

# Find correlations with target
# Temporarily add target to training set for correlation
temp_df = X_train.copy()
temp_df['salary_lpa'] = y_train

# Get correlations with target
correlations = temp_df.corr()['salary_lpa'].sort_values(ascending=False)
print("Top 15 features correlated with salary:")
print(correlations.head(15))

# Visualize
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
```

### Step 6: Remove Multicollinear Features

```python
# Find highly correlated feature pairs
def get_high_correlations(corr_matrix, threshold=0.85):
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append((corr_matrix.columns[i], 
                                       corr_matrix.columns[j], 
                                       corr_matrix.iloc[i, j]))
    return high_corr_pairs

high_corr = get_high_correlations(correlation_matrix)
print(f"Found {len(high_corr)} highly correlated pairs")

# Drop one feature from each highly correlated pair
# Keep the one with higher correlation to target
features_to_drop = []
for feat1, feat2, corr_val in high_corr:
    # Compare correlation with target and drop the weaker one
    if abs(correlations[feat1]) > abs(correlations[feat2]):
        features_to_drop.append(feat2)
    else:
        features_to_drop.append(feat1)

features_to_drop = list(set(features_to_drop))
X_train.drop(features_to_drop, axis=1, inplace=True)
X_test.drop(features_to_drop, axis=1, inplace=True)
```

### Step 7: Variance Threshold

```python
from sklearn.feature_selection import VarianceThreshold

# Remove features with very low variance
selector = VarianceThreshold(threshold=0.01)
X_train_high_var = selector.fit_transform(X_train)
X_test_high_var = selector.transform(X_test)

# Get feature names
selected_features = X_train.columns[selector.get_support()].tolist()
print(f"Features after variance threshold: {len(selected_features)}")
```

### Step 8: Statistical Feature Selection

```python
from sklearn.feature_selection import f_regression, SelectKBest

# Select top K features based on F-statistic
k_best = SelectKBest(score_func=f_regression, k=20)  # Select top 20
X_train_selected = k_best.fit_transform(X_train, y_train)
X_test_selected = k_best.transform(X_test)

# Get selected feature names
selected_mask = k_best.get_support()
selected_features = X_train.columns[selected_mask].tolist()
print("Selected features:")
print(selected_features)
```

### Step 9: Recursive Feature Elimination (Optional)

```python
from sklearn.feature_selection import RFE

# Use Linear Regression as estimator
lr_estimator = LinearRegression()

# RFE to select top N features
rfe = RFE(estimator=lr_estimator, n_features_to_select=15)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Get selected features
rfe_features = X_train.columns[rfe.support_].tolist()
print("RFE selected features:")
print(rfe_features)
```

### Step 10: Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

# Standardize features (important for Linear Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for better interpretability
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
```

---

## 3. Scikit-Learn Pipeline Concept

### Pipeline Visual Overview

```
WITHOUT PIPELINE (Traditional Approach):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                TRAIN DATA
                    ↓
    ┌───────────────────────────────┐
    │  scaler.fit_transform()       │ ← Learns parameters
    └───────────────────────────────┘
                    ↓
    ┌───────────────────────────────┐
    │  encoder.fit_transform()      │ ← Learns categories
    └───────────────────────────────┘
                    ↓
    ┌───────────────────────────────┐
    │  model.fit()                  │
    └───────────────────────────────┘
    
                TEST DATA
                    ↓
    ┌───────────────────────────────┐
    │  scaler.transform()           │ ← Must remember to use same scaler!
    └───────────────────────────────┘
                    ↓
    ┌───────────────────────────────┐
    │  encoder.transform()          │ ← Must remember to use same encoder!
    └───────────────────────────────┘
                    ↓
    ┌───────────────────────────────┐
    │  model.predict()              │
    └───────────────────────────────┘

    ⚠️  Risk: Forgetting steps, wrong order, data leakage


WITH PIPELINE (Recommended Approach):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                TRAIN DATA
                    ↓
    ╔═══════════════════════════════╗
    ║      pipeline.fit()           ║
    ║  ┌─────────────────────────┐  ║
    ║  │ Scaler  (fit_transform) │  ║
    ║  └─────────────────────────┘  ║
    ║            ↓                  ║
    ║  ┌─────────────────────────┐  ║
    ║  │ Encoder (fit_transform) │  ║
    ║  └─────────────────────────┘  ║
    ║            ↓                  ║
    ║  ┌─────────────────────────┐  ║
    ║  │ Model   (fit)           │  ║
    ║  └─────────────────────────┘  ║
    ╚═══════════════════════════════╝
    
                TEST DATA
                    ↓
    ╔═══════════════════════════════╗
    ║    pipeline.predict()         ║
    ║  ┌─────────────────────────┐  ║
    ║  │ Scaler  (transform)     │  ║
    ║  └─────────────────────────┘  ║
    ║            ↓                  ║
    ║  ┌─────────────────────────┐  ║
    ║  │ Encoder (transform)     │  ║
    ║  └─────────────────────────┘  ║
    ║            ↓                  ║
    ║  ┌─────────────────────────┐  ║
    ║  │ Model   (predict)       │  ║
    ║  └─────────────────────────┘  ║
    ╚═══════════════════════════════╝
    
    ✅ Automatic, consistent, no mistakes!
```

---

### What is a Pipeline?

A **Pipeline** in scikit-learn is a tool that chains multiple data processing steps and a final estimator into a single object. It ensures that all preprocessing steps are applied consistently to both training and test data, preventing data leakage and making code more maintainable.

### Why Use Pipelines?

**Benefits:**
1. **Prevents Data Leakage:** Ensures test data is never seen during fit operations
2. **Code Simplicity:** Reduces code duplication and complexity
3. **Reproducibility:** Easy to save, load, and deploy entire workflow
4. **Hyperparameter Tuning:** Works seamlessly with GridSearchCV and RandomizedSearchCV
5. **Production Ready:** Single object contains entire preprocessing + model
6. **Cleaner Code:** No need to manually transform train/test sets separately

### When to Use Each Approach?

| Scenario | Traditional Approach | Pipeline Approach |
|----------|---------------------|-------------------|
| **Learning/Education** | ✅ Better for understanding each step | ❌ Hides complexity |
| **Quick EDA** | ✅ More flexibility | ❌ Overhead for simple tasks |
| **Production Deployment** | ❌ Multiple objects to manage | ✅ Single object deployment |
| **Hyperparameter Tuning** | ❌ Complex setup | ✅ Simple with GridSearchCV |
| **Cross-Validation** | ❌ Risk of data leakage | ✅ Automatic, safe |
| **Team Collaboration** | ❌ Hard to reproduce | ✅ Easy to share |
| **Model Versioning** | ❌ Multiple versions to track | ✅ Single pipeline version |

**Recommendation:** 
- Use **Traditional Approach** for learning and initial exploration
- Use **Pipeline Approach** for final models and production systems

### Pipeline Components

#### 1. **Pipeline** - Sequential Steps
```python
from sklearn.pipeline import Pipeline

# Simple pipeline with two steps
simple_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),      # Step 1: Scale features
    ('model', LinearRegression())       # Step 2: Train model
])
```

#### 2. **ColumnTransformer** - Different Transformations for Different Columns
```python
from sklearn.compose import ColumnTransformer

# Apply different transformations to different column types
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_columns),
    ('cat', OneHotEncoder(), categorical_columns)
])
```

---

### Complete Pipeline Implementation for Our Dataset

#### Step 1: Define Column Groups

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load and prepare data
students = pd.read_csv('indian_engineering_student_placement.csv')
targets = pd.read_csv('placement_targets.csv')
df = students.merge(targets, on='Student_ID')

# Handle missing values
df['extracurricular_involvement'].fillna('None', inplace=True)

# Drop unnecessary columns
df.drop(['Student_ID', 'placement_status'], axis=1, inplace=True)

# Separate features and target
X = df.drop('salary_lpa', axis=1)
y = df['salary_lpa']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define column groups based on data types
numerical_cols = [
    'cgpa', 'tenth_percentage', 'twelfth_percentage', 'backlogs',
    'study_hours_per_day', 'attendance_percentage', 'projects_completed',
    'internships_completed', 'coding_skill_rating', 'communication_skill_rating',
    'aptitude_skill_rating', 'hackathons_participated', 'certifications_count',
    'sleep_hours', 'stress_level'
]

# Binary categorical (will use label encoding or one-hot with 2 categories)
binary_cols = ['gender', 'part_time_job', 'internet_access']

# Nominal categorical (no inherent order)
nominal_cols = ['branch']

# Ordinal categorical (has meaningful order)
ordinal_cols = ['family_income_level', 'city_tier', 'extracurricular_involvement']
```

#### Step 2: Create Individual Preprocessors

```python
# Numerical preprocessor
num_preprocessor = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Binary categorical preprocessor (OneHotEncoder will create n-1 columns)
binary_preprocessor = Pipeline(steps=[
    ('ohe', OneHotEncoder(drop='first', sparse_output=False))
])

# Nominal categorical preprocessor
nominal_preprocessor = Pipeline(steps=[
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Ordinal categorical preprocessor
ordinal_preprocessor = Pipeline(steps=[
    ('ord_enc', OrdinalEncoder(
        categories=[
            ['Low', 'Medium', 'High'],                    # family_income_level
            ['Tier 3', 'Tier 2', 'Tier 1'],              # city_tier (ascending quality)
            ['None', 'Low', 'Medium', 'High']             # extracurricular_involvement
        ],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ))
])
```

**Explanation of Preprocessors:**

- **StandardScaler**: Transforms numerical features to have mean=0 and std=1
  - Formula: `z = (x - μ) / σ`
  - Important for Linear Regression to give equal weight to all features

- **OneHotEncoder**: Creates binary columns for each category
  - `drop='first'`: Removes first category to avoid multicollinearity (for binary)
  - `handle_unknown='ignore'`: Creates all-zero vector for unseen categories
  - `sparse_output=False`: Returns dense array instead of sparse matrix

- **OrdinalEncoder**: Maps categories to integers preserving order
  - `categories`: Defines order explicitly (low to high)
  - `handle_unknown='use_encoded_value'`: Handles unseen categories with -1

#### Step 3: Combine All Preprocessors with ColumnTransformer

```python
# Create the main preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_preprocessor, numerical_cols),
        ('binary', binary_preprocessor, binary_cols),
        ('nominal', nominal_preprocessor, nominal_cols),
        ('ordinal', ordinal_preprocessor, ordinal_cols)
    ],
    remainder='drop'  # Drop any columns not specified
)
```

**ColumnTransformer Parameters Explained:**

- **transformers**: List of (name, transformer, columns) tuples
  - `name`: Identifier for this transformation
  - `transformer`: The preprocessing object (Pipeline or transformer)
  - `columns`: List of column names to apply this transformation to

- **remainder**: What to do with columns not in any transformer
  - `'drop'`: Remove them (default)
  - `'passthrough'`: Keep them as-is
  - A transformer object: Apply that transformer

#### Step 4: Create the Complete Pipeline

```python
# Create the full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

print("Pipeline created successfully!")
print(model_pipeline)
```

**Pipeline Structure:**
```
Pipeline(steps=[
    ('preprocessor', ColumnTransformer(...)),
    ('regressor', LinearRegression())
])
```

#### Step 5: Train the Pipeline

```python
# Fit the entire pipeline on training data
model_pipeline.fit(X_train, y_train)

print("\nPipeline trained successfully!")
print(f"Training set size: {len(X_train)}")
```

**What Happens During fit():**
1. Preprocessor fits on X_train (learns scaling parameters, categories, etc.)
2. Preprocessor transforms X_train
3. LinearRegression fits on transformed data

#### Step 6: Make Predictions

```python
# Make predictions (preprocessing is applied automatically!)
y_train_pred = model_pipeline.predict(X_train)
y_test_pred = model_pipeline.predict(X_test)

print("Predictions made!")
```

**What Happens During predict():**
1. Preprocessor transforms input data (using parameters learned during fit)
2. LinearRegression predicts on transformed data
3. No need to manually apply preprocessing!

#### Step 7: Evaluate the Model

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\n" + "="*60)
print("PIPELINE MODEL PERFORMANCE")
print("="*60)
print(f"\nTraining R² Score:   {train_r2:.4f}")
print(f"Test R² Score:       {test_r2:.4f}")
print(f"Test RMSE:           {test_rmse:.4f}")
print(f"Test MAE:            {test_mae:.4f}")
```

---

### Advanced Pipeline Techniques

#### 1. Adding Custom Transformers

```python
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer to create engineered features"""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Create new features
        X_copy['avg_academic_score'] = (
            X_copy['cgpa'] * 10 + 
            X_copy['tenth_percentage'] + 
            X_copy['twelfth_percentage']
        ) / 3
        
        X_copy['overall_skill_score'] = (
            X_copy['coding_skill_rating'] + 
            X_copy['communication_skill_rating'] + 
            X_copy['aptitude_skill_rating']
        ) / 3
        
        X_copy['practical_experience'] = (
            X_copy['projects_completed'] + 
            X_copy['internships_completed'] + 
            X_copy['hackathons_participated']
        )
        
        X_copy['study_sleep_ratio'] = (
            X_copy['study_hours_per_day'] / (X_copy['sleep_hours'] + 1e-6)
        )
        
        return X_copy

# Add to pipeline
enhanced_pipeline = Pipeline(steps=[
    ('feature_engineering', FeatureEngineer()),
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
```

#### 2. Pipeline with Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_regression

# Create pipeline with feature selection
pipeline_with_selection = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_regression, k=15)),
    ('regressor', LinearRegression())
])

# Fit and evaluate
pipeline_with_selection.fit(X_train, y_train)
y_pred_selected = pipeline_with_selection.predict(X_test)

print(f"\nR² with feature selection: {r2_score(y_test, y_pred_selected):.4f}")
```

#### 3. Pipeline with Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Perform cross-validation on the entire pipeline
cv_scores = cross_val_score(
    model_pipeline, 
    X_train, 
    y_train, 
    cv=5, 
    scoring='r2'
)

print("\nCross-Validation Results:")
print(f"CV R² Scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

#### 4. Pipeline with GridSearchCV for Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

# Create pipeline with Ridge regression
ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

# Define parameter grid
param_grid = {
    'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
    'regressor__fit_intercept': [True, False]
}

# Grid search
grid_search = GridSearchCV(
    ridge_pipeline,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

# Fit grid search
grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print(f"Best CV R² Score: {grid_search.best_score_:.4f}")

# Best model
best_model = grid_search.best_estimator_
```

---

### Accessing Pipeline Components

```python
# Access the preprocessor
preprocessor_obj = model_pipeline.named_steps['preprocessor']

# Access the model
lr_model = model_pipeline.named_steps['regressor']

# Get feature names after preprocessing
feature_names = []

# Numerical features (unchanged names)
feature_names.extend(numerical_cols)

# Binary features (one-hot encoded with drop='first')
for col in binary_cols:
    unique_vals = X_train[col].unique()
    feature_names.extend([f"{col}_{val}" for val in unique_vals[1:]])

# Nominal features (one-hot encoded)
nominal_encoder = preprocessor_obj.named_transformers_['nominal']
ohe = nominal_encoder.named_steps['ohe']
nominal_feature_names = ohe.get_feature_names_out(nominal_cols)
feature_names.extend(nominal_feature_names)

# Ordinal features (unchanged names)
feature_names.extend(ordinal_cols)

print("\nTotal features after preprocessing:", len(feature_names))

# Get coefficients
coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lr_model.coef_
})
coefficients['Abs_Coefficient'] = coefficients['Coefficient'].abs()
coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)

print("\nTop 10 Most Important Features:")
print(coefficients.head(10))
```

---

### Saving and Loading Pipelines

```python
import pickle

# Save the entire pipeline
with open('placement_pipeline.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("Pipeline saved successfully!")

# Load the pipeline
with open('placement_pipeline.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)

# Use loaded pipeline
new_predictions = loaded_pipeline.predict(X_test)
print(f"Loaded pipeline R² Score: {r2_score(y_test, new_predictions):.4f}")
```

---

### Making Predictions with Pipeline on New Data

```python
# Create new student data (must have same columns as training data)
new_student = pd.DataFrame({
    'gender': ['Male'],
    'branch': ['CSE'],
    'cgpa': [8.5],
    'tenth_percentage': [85.0],
    'twelfth_percentage': [82.0],
    'backlogs': [0],
    'study_hours_per_day': [5.0],
    'attendance_percentage': [85.0],
    'projects_completed': [7],
    'internships_completed': [2],
    'coding_skill_rating': [4],
    'communication_skill_rating': [4],
    'aptitude_skill_rating': [5],
    'hackathons_participated': [3],
    'certifications_count': [4],
    'sleep_hours': [7.0],
    'stress_level': [5],
    'part_time_job': ['No'],
    'family_income_level': ['Medium'],
    'city_tier': ['Tier 1'],
    'internet_access': ['Yes'],
    'extracurricular_involvement': ['High']
})

# Predict (preprocessing happens automatically!)
predicted_salary = model_pipeline.predict(new_student)

print(f"\nPredicted Salary for New Student: {predicted_salary[0]:.2f} LPA")
```

---

### Pipeline Best Practices

1. **Always Use Pipelines in Production**
   - Ensures consistency between training and inference
   - Reduces chances of errors

2. **Fit Only on Training Data**
   - Never fit pipeline on test data
   - Use `fit()` on train, `transform()` or `predict()` on test

3. **Include All Preprocessing Steps**
   - Put every transformation in the pipeline
   - Missing value imputation, scaling, encoding, feature selection

4. **Use ColumnTransformer for Mixed Data Types**
   - Different transformations for different column types
   - More flexible than single transformer

5. **Name Your Steps Clearly**
   - Use descriptive names for pipeline steps
   - Makes debugging and hyperparameter tuning easier

6. **Save Entire Pipeline, Not Just Model**
   - Saves preprocessing steps too
   - Deployment becomes trivial

7. **Test Pipeline on Small Sample First**
   - Verify transformations are correct
   - Check output shape and types

---

### Common Pitfalls to Avoid

❌ **Don't do this:**
```python
# Fitting scaler on all data (DATA LEAKAGE!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)
```

✅ **Do this instead:**
```python
# Pipeline ensures no data leakage
X_train, X_test = train_test_split(X)
pipeline.fit(X_train, y_train)
pipeline.predict(X_test)
```

❌ **Don't do this:**
```python
# Manually transforming train and test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model.fit(X_train_scaled, y_train)
model.predict(X_test_scaled)
```

✅ **Do this instead:**
```python
# Pipeline handles everything
pipeline.fit(X_train, y_train)
pipeline.predict(X_test)
```

---

### Complete Working Example

```python
# Full pipeline implementation from start to finish
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np

# 1. Load data
students = pd.read_csv('indian_engineering_student_placement.csv')
targets = pd.read_csv('placement_targets.csv')
df = students.merge(targets, on='Student_ID')

# 2. Basic preprocessing
df['extracurricular_involvement'].fillna('None', inplace=True)
df.drop(['Student_ID', 'placement_status'], axis=1, inplace=True)

# 3. Separate features and target
X = df.drop('salary_lpa', axis=1)
y = df['salary_lpa']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Define column groups
numerical_cols = ['cgpa', 'tenth_percentage', 'twelfth_percentage', 'backlogs',
                 'study_hours_per_day', 'attendance_percentage', 'projects_completed',
                 'internships_completed', 'coding_skill_rating', 
                 'communication_skill_rating', 'aptitude_skill_rating', 
                 'hackathons_participated', 'certifications_count', 
                 'sleep_hours', 'stress_level']

binary_cols = ['gender', 'part_time_job', 'internet_access']
nominal_cols = ['branch']
ordinal_cols = ['family_income_level', 'city_tier', 'extracurricular_involvement']

# 6. Create preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('binary', OneHotEncoder(drop='first', sparse_output=False), binary_cols),
    ('nominal', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_cols),
    ('ordinal', OrdinalEncoder(
        categories=[['Low', 'Medium', 'High'],
                   ['Tier 3', 'Tier 2', 'Tier 1'],
                   ['None', 'Low', 'Medium', 'High']],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ), ordinal_cols)
])

# 7. Create pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 8. Train pipeline
model_pipeline.fit(X_train, y_train)

# 9. Make predictions
y_train_pred = model_pipeline.predict(X_train)
y_test_pred = model_pipeline.predict(X_test)

# 10. Evaluate
print("\n" + "="*60)
print("PIPELINE MODEL RESULTS")
print("="*60)
print(f"\nTraining R² Score:   {r2_score(y_train, y_train_pred):.4f}")
print(f"Test R² Score:       {r2_score(y_test, y_test_pred):.4f}")
print(f"Test RMSE:           {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"Test MAE:            {mean_absolute_error(y_test, y_test_pred):.4f}")

# 11. Cross-validation
cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='r2')
print(f"\nMean CV R² Score:    {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# 12. Save pipeline
import pickle
with open('placement_pipeline.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)
print("\nPipeline saved successfully!")
```

---

## 4. Train Linear Regression Model

### Step 1: Initialize and Train the Model

```python
from sklearn.linear_model import LinearRegression

# Initialize Linear Regression
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train_scaled, y_train)

print("Model trained successfully!")
print(f"Number of features used: {X_train_scaled.shape[1]}")
```

### Step 2: Make Predictions

```python
# Predictions on training set
y_train_pred = lr_model.predict(X_train_scaled)

# Predictions on test set
y_test_pred = lr_model.predict(X_test_scaled)

print("Predictions completed!")
```

### Step 3: Evaluate Model Performance

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Training set performance
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Test set performance
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)
print("\nTraining Set:")
print(f"  R² Score:            {train_r2:.4f}")
print(f"  Mean Squared Error:  {train_mse:.4f}")
print(f"  Root MSE:            {train_rmse:.4f}")
print(f"  Mean Absolute Error: {train_mae:.4f}")

print("\nTest Set:")
print(f"  R² Score:            {test_r2:.4f}")
print(f"  Mean Squared Error:  {test_mse:.4f}")
print(f"  Root MSE:            {test_rmse:.4f}")
print(f"  Mean Absolute Error: {test_mae:.4f}")

print("\nInterpretation:")
print(f"  The model explains {test_r2*100:.2f}% of variance in salary")
print(f"  Average prediction error: ±{test_mae:.2f} LPA")
```

### Step 4: Check Model Assumptions

**Linearity Check:**
```python
# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Salary (LPA)')
plt.ylabel('Predicted Salary (LPA)')
plt.title('Actual vs Predicted Salary')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
```

**Residual Analysis:**
```python
# Calculate residuals
residuals = y_test - y_test_pred

# Plot residuals
plt.figure(figsize=(12, 5))

# Residual plot
plt.subplot(1, 2, 1)
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Salary')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Q-Q plot for normality
plt.subplot(1, 2, 2)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')

plt.tight_layout()
plt.savefig('residual_analysis.png')
```

### Step 5: Feature Importance Analysis

```python
# Get feature coefficients
feature_importance = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Coefficient': lr_model.coef_
})

# Sort by absolute coefficient value
feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualize
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(top_features['Feature'], top_features['Coefficient'])
plt.xlabel('Coefficient Value')
plt.title('Top 15 Feature Coefficients')
plt.tight_layout()
plt.savefig('feature_importance.png')
```

### Step 6: Model Equation

```python
# Display model equation
print("\nLinear Regression Equation:")
print(f"Salary (LPA) = {lr_model.intercept_:.4f}")
for feature, coef in zip(X_train_scaled.columns, lr_model.coef_):
    print(f"  + ({coef:.4f}) × {feature}")
```

### Step 7: Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, 
                            cv=5, scoring='r2')

print("\nCross-Validation Results:")
print(f"  CV R² Scores: {cv_scores}")
print(f"  Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

### Step 8: Save the Model

```python
import pickle

# Save the model
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModel and scaler saved successfully!")
```

### Step 9: Make Predictions on New Data

```python
# Example: Predict salary for new student
new_student = pd.DataFrame({
    'cgpa': [8.5],
    'tenth_percentage': [85.0],
    'twelfth_percentage': [82.0],
    'backlogs': [0],
    'study_hours_per_day': [5.0],
    'attendance_percentage': [85.0],
    'projects_completed': [7],
    'internships_completed': [2],
    'coding_skill_rating': [4],
    'communication_skill_rating': [4],
    'aptitude_skill_rating': [5],
    # ... add all other features
})

# Apply same preprocessing and scaling
new_student_scaled = scaler.transform(new_student)
predicted_salary = lr_model.predict(new_student_scaled)

print(f"\nPredicted Salary: {predicted_salary[0]:.2f} LPA")
```

---

## 4. Sklearn Pipeline Approach (Recommended)

### Why Use Pipelines?

**Benefits:**
1. **Cleaner Code:** Encapsulates all preprocessing and modeling steps
2. **Prevents Data Leakage:** Ensures transformations are fit only on training data
3. **Reproducibility:** Easy to apply same transformations to new data
4. **Easy Deployment:** Save entire pipeline as single object
5. **Cross-Validation Friendly:** Works seamlessly with GridSearchCV and cross_val_score

### Step 1: Identify Column Types

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load and merge data
students = pd.read_csv('indian_engineering_student_placement.csv')
targets = pd.read_csv('placement_targets.csv')
df = students.merge(targets, on='Student_ID')

# Handle missing values
df['extracurricular_involvement'].fillna(df['extracurricular_involvement'].mode()[0], inplace=True)

# Drop unnecessary columns
df.drop(['Student_ID', 'placement_status'], axis=1, inplace=True)

# Define column groups
numerical_cols = [
    'cgpa', 'tenth_percentage', 'twelfth_percentage', 'backlogs',
    'study_hours_per_day', 'attendance_percentage', 'projects_completed',
    'internships_completed', 'coding_skill_rating', 'communication_skill_rating',
    'aptitude_skill_rating', 'hackathons_participated', 'certifications_count',
    'sleep_hours', 'stress_level'
]

# Binary categorical (will be label encoded)
binary_cols = ['gender', 'part_time_job', 'internet_access']

# Nominal categorical (no inherent order - needs one-hot encoding)
nominal_cols = ['branch']

# Ordinal categorical (has meaningful order)
ordinal_cols = ['city_tier', 'family_income_level', 'extracurricular_involvement']

print(f"Numerical columns: {len(numerical_cols)}")
print(f"Binary columns: {len(binary_cols)}")
print(f"Nominal columns: {len(nominal_cols)}")
print(f"Ordinal columns: {len(ordinal_cols)}")
```

### Step 2: Create Preprocessing Pipelines for Each Column Type

```python
# Numerical preprocessing pipeline
num_preprocessor = Pipeline(steps=[
    ('scaler', StandardScaler())  # Standardize numerical features
])

# Nominal categorical preprocessing pipeline
nominal_preprocessor = Pipeline(steps=[
    ('ohe', OneHotEncoder(
        drop='first',           # Drop first category to avoid multicollinearity
        handle_unknown='ignore', # Ignore unknown categories in test set
        sparse_output=False      # Return dense array
    ))
])

# Ordinal categorical preprocessing pipeline
ordinal_preprocessor = Pipeline(steps=[
    ('ord_enc', OrdinalEncoder(
        categories=[
            ['Tier 3', 'Tier 2', 'Tier 1'],  # city_tier (ascending quality)
            ['Low', 'Medium', 'High'],        # family_income_level
            ['None', 'Low', 'Medium', 'High'] # extracurricular_involvement
        ],
        handle_unknown='use_encoded_value',
        unknown_value=-1  # Assign -1 to unknown categories
    ))
])

# Binary categorical preprocessing pipeline
# Option 1: Use OrdinalEncoder with 2 categories
binary_preprocessor = Pipeline(steps=[
    ('binary_enc', OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ))
])

# Option 2: Use OneHotEncoder and drop one column (same result)
# binary_preprocessor = Pipeline(steps=[
#     ('ohe', OneHotEncoder(drop='first', sparse_output=False))
# ])
```

### Step 3: Combine All Preprocessors with ColumnTransformer

```python
# Combine all preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_preprocessor, numerical_cols),
        ('nominal', nominal_preprocessor, nominal_cols),
        ('ordinal', ordinal_preprocessor, ordinal_cols),
        ('binary', binary_preprocessor, binary_cols)
    ],
    remainder='drop',  # Drop any columns not specified
    verbose_feature_names_out=False  # Cleaner feature names
)
```

### Step 4: Create Complete Pipeline with Model

```python
# Create full pipeline: preprocessing + model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

print("Pipeline created successfully!")
print("\nPipeline steps:")
print(model_pipeline)
```

### Step 5: Train-Test Split and Fit Pipeline

```python
# Separate features and target
X = df.drop('salary_lpa', axis=1)
y = df['salary_lpa']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit the entire pipeline (preprocessing + model)
# The pipeline will:
# 1. Fit preprocessor on X_train
# 2. Transform X_train
# 3. Fit Linear Regression on transformed X_train
model_pipeline.fit(X_train, y_train)

print("\nPipeline fitted successfully!")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
```

### Step 6: Make Predictions

```python
# Make predictions (pipeline automatically applies preprocessing)
y_train_pred = model_pipeline.predict(X_train)
y_test_pred = model_pipeline.predict(X_test)

print("Predictions completed!")
```

### Step 7: Evaluate Pipeline Performance

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\n" + "="*70)
print("PIPELINE MODEL PERFORMANCE")
print("="*70)
print("\nTraining Set:")
print(f"  R² Score:            {train_r2:.4f}")
print(f"  RMSE:                {train_rmse:.4f}")
print(f"  MAE:                 {train_mae:.4f}")

print("\nTest Set:")
print(f"  R² Score:            {test_r2:.4f}")
print(f"  RMSE:                {test_rmse:.4f}")
print(f"  MAE:                 {test_mae:.4f}")

# Check for overfitting
print(f"\nOverfitting Check:")
print(f"  R² difference: {abs(train_r2 - test_r2):.4f}")
if abs(train_r2 - test_r2) < 0.05:
    print("  ✓ Model generalizes well")
else:
    print("  ⚠ Possible overfitting")
```

### Step 8: Cross-Validation with Pipeline

```python
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(
    model_pipeline, 
    X_train, 
    y_train, 
    cv=5, 
    scoring='r2',
    n_jobs=-1  # Use all CPU cores
)

print("\n" + "="*70)
print("CROSS-VALIDATION RESULTS")
print("="*70)
print(f"CV R² Scores: {cv_scores}")
print(f"Mean CV R²:   {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

### Step 9: Access Transformed Feature Names

```python
# Get feature names after preprocessing
try:
    feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
    print(f"\nTotal features after preprocessing: {len(feature_names)}")
    print("\nFeature names (first 20):")
    print(list(feature_names[:20]))
except AttributeError:
    print("Feature names not available")
```

### Step 10: Extract Coefficients and Feature Importance

```python
# Get the trained model
trained_model = model_pipeline.named_steps['model']

# Get coefficients
coefficients = trained_model.coef_

# Get feature names
feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()

# Create feature importance DataFrame
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("\n" + "="*70)
print("TOP 15 MOST IMPORTANT FEATURES")
print("="*70)
print(feature_importance.head(15).to_string(index=False))
```

### Step 11: Save Complete Pipeline

```python
import pickle
import joblib

# Method 1: Using pickle
with open('salary_prediction_pipeline.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

# Method 2: Using joblib (recommended for sklearn objects)
joblib.dump(model_pipeline, 'salary_prediction_pipeline.joblib')

print("\nPipeline saved successfully!")
print("Files created:")
print("  - salary_prediction_pipeline.pkl")
print("  - salary_prediction_pipeline.joblib")
```

### Step 12: Load and Use Saved Pipeline

```python
# Load the pipeline
loaded_pipeline = joblib.load('salary_prediction_pipeline.joblib')

# Make predictions on new data
# The pipeline handles ALL preprocessing automatically!
new_student = pd.DataFrame({
    'gender': ['Male'],
    'branch': ['CSE'],
    'cgpa': [8.5],
    'tenth_percentage': [85.0],
    'twelfth_percentage': [82.0],
    'backlogs': [0],
    'study_hours_per_day': [5.0],
    'attendance_percentage': [85.0],
    'projects_completed': [7],
    'internships_completed': [2],
    'coding_skill_rating': [4],
    'communication_skill_rating': [4],
    'aptitude_skill_rating': [5],
    'hackathons_participated': [3],
    'certifications_count': [4],
    'sleep_hours': [7.0],
    'stress_level': [6],
    'part_time_job': ['No'],
    'family_income_level': ['Medium'],
    'city_tier': ['Tier 1'],
    'internet_access': ['Yes'],
    'extracurricular_involvement': ['Medium']
})

predicted_salary = loaded_pipeline.predict(new_student)
print(f"\nPredicted Salary for new student: {predicted_salary[0]:.2f} LPA")
```

### Step 13: Pipeline with Feature Engineering

```python
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for feature engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer to create engineered features"""
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Academic performance composite
        X['avg_academic_score'] = (X['cgpa'] * 10 + 
                                   X['tenth_percentage'] + 
                                   X['twelfth_percentage']) / 3
        
        # Overall skill score
        X['overall_skill_score'] = (X['coding_skill_rating'] + 
                                    X['communication_skill_rating'] + 
                                    X['aptitude_skill_rating']) / 3
        
        # Practical experience
        X['practical_experience'] = (X['projects_completed'] + 
                                     X['internships_completed'] + 
                                     X['hackathons_participated'])
        
        # Study-sleep ratio
        X['study_sleep_ratio'] = X['study_hours_per_day'] / (X['sleep_hours'] + 1)
        
        # Binary flags
        X['high_cgpa'] = (X['cgpa'] >= 8.5).astype(int)
        X['no_backlogs'] = (X['backlogs'] == 0).astype(int)
        X['is_cs_it'] = X['branch'].isin(['CSE', 'IT']).astype(int)
        
        return X

# Enhanced pipeline with feature engineering
enhanced_pipeline = Pipeline(steps=[
    ('feature_engineer', FeatureEngineer()),
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Update numerical columns to include engineered features
numerical_cols_enhanced = numerical_cols + [
    'avg_academic_score', 'overall_skill_score', 'practical_experience',
    'study_sleep_ratio', 'high_cgpa', 'no_backlogs', 'is_cs_it'
]

# Update preprocessor
preprocessor_enhanced = ColumnTransformer(
    transformers=[
        ('num', num_preprocessor, numerical_cols_enhanced),
        ('nominal', nominal_preprocessor, nominal_cols),
        ('ordinal', ordinal_preprocessor, ordinal_cols),
        ('binary', binary_preprocessor, binary_cols)
    ],
    remainder='drop'
)

# Update pipeline
enhanced_pipeline = Pipeline(steps=[
    ('feature_engineer', FeatureEngineer()),
    ('preprocessor', preprocessor_enhanced),
    ('model', LinearRegression())
])

# Fit enhanced pipeline
enhanced_pipeline.fit(X_train, y_train)
y_test_pred_enhanced = enhanced_pipeline.predict(X_test)

# Evaluate
enhanced_r2 = r2_score(y_test, y_test_pred_enhanced)
print(f"\nEnhanced Pipeline R² Score: {enhanced_r2:.4f}")
```

### Step 14: Hyperparameter Tuning with Pipeline

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso

# Create pipeline with regularized regression
regularized_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', Ridge())  # Can also use Lasso()
])

# Define parameter grid
param_grid = {
    'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'model__fit_intercept': [True, False],
    'preprocessor__num__scaler': [StandardScaler(), None]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    regularized_pipeline,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# Fit grid search
grid_search.fit(X_train, y_train)

print("\n" + "="*70)
print("GRID SEARCH RESULTS")
print("="*70)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV R² score: {grid_search.best_score_:.4f}")

# Test best model
best_pipeline = grid_search.best_estimator_
y_test_pred_best = best_pipeline.predict(X_test)
best_test_r2 = r2_score(y_test, y_test_pred_best)
print(f"Test R² score with best params: {best_test_r2:.4f}")
```

### Step 15: Pipeline Visualization

```python
# Display pipeline structure
from sklearn import set_config
set_config(display='diagram')

# This will show a visual diagram in Jupyter notebooks
model_pipeline
```

---

## Complete Pipeline Code (All-in-One)

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np

# 1. Load and prepare data
students = pd.read_csv('indian_engineering_student_placement.csv')
targets = pd.read_csv('placement_targets.csv')
df = students.merge(targets, on='Student_ID')

# 2. Basic preprocessing
df['extracurricular_involvement'].fillna(df['extracurricular_involvement'].mode()[0], inplace=True)
df.drop(['Student_ID', 'placement_status'], axis=1, inplace=True)

# 3. Define column groups
numerical_cols = [
    'cgpa', 'tenth_percentage', 'twelfth_percentage', 'backlogs',
    'study_hours_per_day', 'attendance_percentage', 'projects_completed',
    'internships_completed', 'coding_skill_rating', 'communication_skill_rating',
    'aptitude_skill_rating', 'hackathons_participated', 'certifications_count',
    'sleep_hours', 'stress_level'
]
binary_cols = ['gender', 'part_time_job', 'internet_access']
nominal_cols = ['branch']
ordinal_cols = ['city_tier', 'family_income_level', 'extracurricular_involvement']

# 4. Create preprocessing pipelines
num_preprocessor = Pipeline(steps=[
    ('scaler', StandardScaler())
])

nominal_preprocessor = Pipeline(steps=[
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
])

ordinal_preprocessor = Pipeline(steps=[
    ('ord_enc', OrdinalEncoder(
        categories=[
            ['Tier 3', 'Tier 2', 'Tier 1'],
            ['Low', 'Medium', 'High'],
            ['None', 'Low', 'Medium', 'High']
        ],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ))
])

binary_preprocessor = Pipeline(steps=[
    ('binary_enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

# 5. Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_preprocessor, numerical_cols),
        ('nominal', nominal_preprocessor, nominal_cols),
        ('ordinal', ordinal_preprocessor, ordinal_cols),
        ('binary', binary_preprocessor, binary_cols)
    ],
    remainder='drop'
)

# 6. Create full pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# 7. Train-test split
X = df.drop('salary_lpa', axis=1)
y = df['salary_lpa']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Fit pipeline
model_pipeline.fit(X_train, y_train)

# 9. Predictions
y_train_pred = model_pipeline.predict(X_train)
y_test_pred = model_pipeline.predict(X_test)

# 10. Evaluation
print("="*70)
print("MODEL PERFORMANCE")
print("="*70)
print(f"Train R²: {r2_score(y_train, y_train_pred):.4f}")
print(f"Test R²:  {r2_score(y_test, y_test_pred):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"Test MAE:  {mean_absolute_error(y_test, y_test_pred):.4f}")

# 11. Cross-validation
cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='r2')
print(f"\nCV R² (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 12. Save pipeline
import joblib
joblib.dump(model_pipeline, 'salary_prediction_pipeline.joblib')
print("\nPipeline saved successfully!")
```

---

## Pipeline vs Traditional Approach Comparison

| Aspect | Traditional Approach | Pipeline Approach |
|--------|---------------------|-------------------|
| **Code Lines** | 50-100+ lines | 20-30 lines |
| **Data Leakage Risk** | High (manual fit/transform) | Low (automatic) |
| **Reproducibility** | Manual tracking needed | Single object |
| **Deployment** | Multiple files/objects | Single .pkl/.joblib |
| **Cross-validation** | Complex setup | Works seamlessly |
| **New predictions** | Manual preprocessing | Automatic |
| **Hyperparameter tuning** | Difficult | Easy with GridSearchCV |
| **Code readability** | Lower | Higher |

---

## Summary of Two Variable Types

### 1. **Numerical (Continuous) Variables**
These are features with continuous or discrete numerical values:

- **Academic Features:** cgpa, tenth_percentage, twelfth_percentage
- **Study Metrics:** study_hours_per_day, attendance_percentage, sleep_hours
- **Experience Counts:** projects_completed, internships_completed, hackathons_participated, certifications_count, backlogs
- **Skill Ratings:** coding_skill_rating, communication_skill_rating, aptitude_skill_rating
- **Other:** stress_level, salary_lpa (target)

**Linear Regression Handling:** Used directly after scaling

### 2. **Categorical Variables**
These are features with discrete categories:

**Binary Categorical:**
- gender (Male, Female)
- part_time_job (Yes, No)
- internet_access (Yes, No)

**Nominal Categorical (No Order):**
- branch (CSE, ECE, IT, ME, CE)
- family_income_level (Low, Medium, High)
- city_tier (Tier 1, Tier 2, Tier 3)
- extracurricular_involvement (None, Low, Medium, High)
- placement_status (Placed, Not Placed)

**Linear Regression Handling:** 
- Binary: Label Encoding (0, 1)
- Nominal: One-Hot Encoding (creates dummy variables)
- Ordinal (if any): Label Encoding with proper ordering

---

## Key Considerations for Linear Regression

### Assumptions to Check:
1. **Linearity:** Relationship between features and target is linear
2. **Independence:** Observations are independent
3. **Homoscedasticity:** Constant variance of residuals
4. **Normality:** Residuals are normally distributed
5. **No Multicollinearity:** Features are not highly correlated

### Best Practices:
1. **Always scale features** using StandardScaler or MinMaxScaler
2. **Handle categorical variables** properly (One-Hot or Label Encoding)
3. **Check for outliers** and handle appropriately
4. **Remove highly correlated features** (correlation > 0.85)
5. **Use cross-validation** to assess model generalization
6. **Analyze residuals** to verify assumptions
7. **Interpret coefficients** to understand feature importance

### Expected Performance:
- **R² Score:** Aim for > 0.70 (explains 70%+ variance)
- **RMSE:** Should be reasonable compared to salary range (ideally < 3 LPA)
- **Training vs Test Performance:** Should be similar (avoid overfitting)

---

## Complete Pipeline Summary

### Traditional Approach:
```
1. Data Loading & Merging
   ↓
2. Missing Value Treatment
   ↓
3. Outlier Detection & Treatment
   ↓
4. Feature Engineering
   ↓
5. Train-Test Split
   ↓
6. Categorical Encoding
   ↓
7. Feature Selection
   ↓
8. Feature Scaling
   ↓
9. Model Training (Linear Regression)
   ↓
10. Model Evaluation
   ↓
11. Residual Analysis
   ↓
12. Model Deployment/Saving
```

### Sklearn Pipeline Approach (Recommended):
```
1. Data Loading & Merging
   ↓
2. Basic Cleaning (Missing Values, Drop ID)
   ↓
3. Define Column Groups (Numerical, Categorical, etc.)
   ↓
4. Train-Test Split (BEFORE any transformations)
   ↓
5. Create Pipeline:
   │
   ├── Preprocessor (ColumnTransformer)
   │   ├── Numerical Pipeline (StandardScaler)
   │   ├── Nominal Pipeline (OneHotEncoder)
   │   ├── Ordinal Pipeline (OrdinalEncoder)
   │   └── Binary Pipeline (OrdinalEncoder)
   │
   └── Model (LinearRegression)
   ↓
6. Fit Pipeline (Automatically fits preprocessor + model)
   ↓
7. Predict (Automatically applies preprocessing)
   ↓
8. Evaluate & Cross-Validate
   ↓
9. Save Single Pipeline Object
```

**Key Advantage:** Pipeline ensures preprocessing is fit ONLY on training data and consistently applied to test/new data, preventing data leakage!

---

## Final Notes

- **Not Placed Students:** The 697 students with 0 salary might skew results. Consider:
  - Training separate classification model first (Placed vs Not Placed)
  - Then train regression model only on placed students
  - Or use these as valid data points

- **Feature Engineering:** More domain knowledge can create better features
  
- **Alternative Models:** If Linear Regression doesn't perform well, consider:
  - Ridge Regression (handles multicollinearity)
  - Lasso Regression (feature selection + regularization)
  - Polynomial Regression (captures non-linear relationships)

- **Regularization:** If overfitting occurs, use L1 (Lasso) or L2 (Ridge) regularization

---

**Dataset Source:** Indian Engineering Student Placement Dataset  
**Analysis Date:** February 2026  
**Model Type:** Linear Regression (Supervised Learning - Regression)

---

## Quick Reference Cheat Sheet

### Essential Pipeline Code Template

```python
# 1. Basic imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

# 2. Define column groups
numerical_cols = ['col1', 'col2', ...]
categorical_cols = ['cat1', 'cat2', ...]

# 3. Create preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(), categorical_cols)
])

# 4. Create pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# 5. Fit and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### Common Encoders Quick Reference

| Encoder | Use Case | Example |
|---------|----------|---------|
| `LabelEncoder` | Binary categories (0/1) | Male/Female → 0/1 |
| `OrdinalEncoder` | Ordered categories | Low/Med/High → 0/1/2 |
| `OneHotEncoder` | Unordered categories | Red/Blue/Green → 3 columns |
| `StandardScaler` | Numerical (mean=0, std=1) | All numerical features |
| `MinMaxScaler` | Numerical (range 0-1) | Alternative to StandardScaler |

### Sklearn Pipeline Parameters

```python
# OneHotEncoder parameters
OneHotEncoder(
    drop='first',              # Prevent multicollinearity
    handle_unknown='ignore',   # Handle unseen categories
    sparse_output=False        # Return dense array
)

# OrdinalEncoder parameters
OrdinalEncoder(
    categories=[['Low', 'Med', 'High']],  # Define order
    handle_unknown='use_encoded_value',   # Handle unseen
    unknown_value=-1                       # Value for unseen
)

# ColumnTransformer parameters
ColumnTransformer(
    transformers=[...],    # List of transformers
    remainder='drop'       # 'drop', 'passthrough', or transformer
)
```

### Model Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# R² Score (0 to 1, higher is better)
r2 = r2_score(y_test, y_pred)

# RMSE (same unit as target, lower is better)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# MAE (same unit as target, lower is better)
mae = mean_absolute_error(y_test, y_pred)
```

### Common Pipeline Patterns

**Pattern 1: Simple Numerical + Categorical**
```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(), cat_cols)
    ])),
    ('model', LinearRegression())
])
```

**Pattern 2: With Feature Selection**
```python
Pipeline([
    ('preprocessor', ...),
    ('feature_selection', SelectKBest(k=10)),
    ('model', LinearRegression())
])
```

**Pattern 3: With Custom Transformer**
```python
Pipeline([
    ('feature_engineer', CustomTransformer()),
    ('preprocessor', ...),
    ('model', LinearRegression())
])
```

### Troubleshooting Guide

| Problem | Solution |
|---------|----------|
| ValueError: could not convert string to float | Check if categorical encoding is applied |
| ValueError: unknown category | Use `handle_unknown='ignore'` in encoders |
| Data leakage | Ensure train-test split BEFORE pipeline.fit() |
| Column mismatch | Ensure X_test has same columns as X_train |
| Poor performance | Check feature scaling, encoding, and selection |
| MemoryError | Use `sparse_output=True` in OneHotEncoder |

---

## Additional Resources

- **Scikit-learn Pipeline Documentation:** https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- **ColumnTransformer Documentation:** https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
- **Preprocessing Guide:** https://scikit-learn.org/stable/modules/preprocessing.html

---

**End of ML Pipeline Notes**

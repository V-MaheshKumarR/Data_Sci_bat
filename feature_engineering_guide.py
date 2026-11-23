# Feature Engineering: Complete Practical Guide
# Dataset: Titanic (from Kaggle)
# This comprehensive guide covers all major feature engineering techniques

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, 
                                   LabelEncoder, OneHotEncoder)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (SelectKBest, chi2, mutual_info_classif,
                                       RFE, SelectFromModel)
import warnings
warnings.filterwarnings('ignore')

"""
=============================================================================
PART 1: DATA LOADING AND INITIAL EXPLORATION
=============================================================================
"""

# Load Titanic dataset
# Download from: https://www.kaggle.com/c/titanic/data
# For this example, we'll create a sample dataset structure

# Create sample Titanic data (in practice, load from CSV)
np.random.seed(42)
n_samples = 891

data = {
    'PassengerId': range(1, n_samples + 1),
    'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
    'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
    'Name': [f'Person_{i}' for i in range(n_samples)],
    'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
    'Age': np.random.choice(list(range(1, 80)) + [np.nan]*20, n_samples),
    'SibSp': np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.68, 0.23, 0.06, 0.02, 0.005, 0.005]),
    'Parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples, p=[0.76, 0.13, 0.08, 0.01, 0.01, 0.005, 0.005]),
    'Ticket': [f'TICKET{i}' for i in range(n_samples)],
    'Fare': np.random.exponential(32, n_samples),
    'Cabin': np.random.choice(['C85', 'C123', 'E46', np.nan], n_samples, p=[0.1, 0.1, 0.1, 0.7]),
    'Embarked': np.random.choice(['C', 'Q', 'S', np.nan], n_samples, p=[0.19, 0.09, 0.72, 0.002])
}

df = pd.DataFrame(data)

print("=" * 80)
print("TITANIC DATASET - FEATURE ENGINEERING GUIDE")
print("=" * 80)
print("\nDataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe())

"""
=============================================================================
PART 2: HANDLING MISSING VALUES
=============================================================================
"""

print("\n" + "=" * 80)
print("HANDLING MISSING VALUES")
print("=" * 80)

# Strategy 1: Fill with statistical measures
df['Age'].fillna(df['Age'].median(), inplace=True)
print("\n✓ Age: Filled with median")

# Strategy 2: Fill with mode for categorical
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
print("✓ Embarked: Filled with mode")

# Strategy 3: Create missing indicator feature
df['Cabin_Missing'] = df['Cabin'].isnull().astype(int)
print("✓ Cabin_Missing: Created binary indicator for missing cabins")

# Strategy 4: Extract information before dropping
df['Has_Cabin'] = df['Cabin'].notna().astype(int)
print("✓ Has_Cabin: Extracted cabin availability feature")

"""
=============================================================================
PART 3: FEATURE CREATION FROM EXISTING FEATURES
=============================================================================
"""

print("\n" + "=" * 80)
print("CREATING NEW FEATURES")
print("=" * 80)

# 1. FAMILY SIZE FEATURES
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print("\n✓ FamilySize: Combined siblings/spouses and parents/children")

df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
print("✓ IsAlone: Binary indicator for solo travelers")

df['FamilyCategory'] = pd.cut(df['FamilySize'], 
                               bins=[0, 1, 4, 20], 
                               labels=['Alone', 'Small', 'Large'])
print("✓ FamilyCategory: Categorized family sizes")

# 2. AGE GROUP FEATURES
df['AgeGroup'] = pd.cut(df['Age'], 
                        bins=[0, 12, 18, 35, 60, 100],
                        labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
print("✓ AgeGroup: Created age categories")

df['IsChild'] = (df['Age'] < 18).astype(int)
print("✓ IsChild: Binary indicator for children")

# 3. FARE FEATURES
df['FarePerPerson'] = df['Fare'] / df['FamilySize']
print("✓ FarePerPerson: Calculated fare per family member")

df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'])
print("✓ FareBin: Quartile-based fare categories")

# 4. TEXT FEATURE ENGINEERING - Extract Title from Name
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 
                                    'Don', 'Dr', 'Major', 'Rev', 'Sir', 
                                    'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
print("✓ Title: Extracted and cleaned titles from names")

# 5. TICKET FEATURES
df['TicketPrefix'] = df['Ticket'].str.extract('([A-Za-z]+)', expand=False)
df['TicketPrefix'].fillna('None', inplace=True)
df['TicketLength'] = df['Ticket'].str.len()
print("✓ Ticket features: Extracted prefix and length")

# 6. INTERACTION FEATURES
df['Pclass_Sex'] = df['Pclass'].astype(str) + '_' + df['Sex']
print("✓ Pclass_Sex: Interaction between class and gender")

df['Age_Class'] = df['Age'] * df['Pclass']
print("✓ Age_Class: Interaction between age and class")

# 7. POLYNOMIAL FEATURES
df['Age_Squared'] = df['Age'] ** 2
df['Fare_Log'] = np.log1p(df['Fare'])  # log(1 + x) to handle zeros
print("✓ Polynomial features: Age squared and log-transformed fare")

# 8. BINNING CONTINUOUS VARIABLES
df['AgeTimesClass'] = df['Age'] * df['Pclass']
print("✓ AgeTimesClass: Combined age and passenger class")

print("\nNew features created:")
new_features = ['FamilySize', 'IsAlone', 'FamilyCategory', 'AgeGroup', 
                'IsChild', 'FarePerPerson', 'FareBin', 'Title', 
                'TicketPrefix', 'Pclass_Sex', 'Age_Squared', 'Fare_Log']
print(new_features)

"""
=============================================================================
PART 4: ENCODING CATEGORICAL VARIABLES
=============================================================================
"""

print("\n" + "=" * 80)
print("ENCODING CATEGORICAL FEATURES")
print("=" * 80)

# Create a copy for encoding demonstrations
df_encoded = df.copy()

# 1. LABEL ENCODING (for ordinal features)
from sklearn.preprocessing import LabelEncoder

le_sex = LabelEncoder()
df_encoded['Sex_Label'] = le_sex.fit_transform(df_encoded['Sex'])
print(f"\n✓ Label Encoding - Sex: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")

le_embarked = LabelEncoder()
df_encoded['Embarked_Label'] = le_embarked.fit_transform(df_encoded['Embarked'])
print(f"✓ Label Encoding - Embarked: {dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))}")

# 2. ONE-HOT ENCODING (for nominal features)
embarked_dummies = pd.get_dummies(df_encoded['Embarked'], prefix='Embarked', drop_first=True)
df_encoded = pd.concat([df_encoded, embarked_dummies], axis=1)
print(f"\n✓ One-Hot Encoding - Embarked: Created {embarked_dummies.shape[1]} binary columns")

title_dummies = pd.get_dummies(df_encoded['Title'], prefix='Title', drop_first=True)
df_encoded = pd.concat([df_encoded, title_dummies], axis=1)
print(f"✓ One-Hot Encoding - Title: Created {title_dummies.shape[1]} binary columns")

# 3. ORDINAL ENCODING (manual mapping)
pclass_mapping = {1: 'First', 2: 'Second', 3: 'Third'}
df_encoded['Pclass_Name'] = df_encoded['Pclass'].map(pclass_mapping)
print(f"\n✓ Ordinal Mapping - Pclass: {pclass_mapping}")

# 4. FREQUENCY ENCODING
title_freq = df_encoded['Title'].value_counts(normalize=True)
df_encoded['Title_Frequency'] = df_encoded['Title'].map(title_freq)
print(f"✓ Frequency Encoding - Title: Mapped to occurrence frequency")

# 5. TARGET ENCODING (mean encoding)
target_mean = df_encoded.groupby('Title')['Survived'].mean()
df_encoded['Title_Target_Enc'] = df_encoded['Title'].map(target_mean)
print(f"✓ Target Encoding - Title: Mapped to mean survival rate")

"""
=============================================================================
PART 5: FEATURE SCALING AND NORMALIZATION
=============================================================================
"""

print("\n" + "=" * 80)
print("FEATURE SCALING & NORMALIZATION")
print("=" * 80)

# Select numerical features for scaling
numerical_features = ['Age', 'Fare', 'FamilySize', 'FarePerPerson']
df_scaled = df[numerical_features].copy()

# 1. STANDARDIZATION (Z-score normalization)
scaler_standard = StandardScaler()
df_scaled['Age_Standardized'] = scaler_standard.fit_transform(df[['Age']])
print(f"\n✓ StandardScaler - Age:")
print(f"  Mean: {df_scaled['Age_Standardized'].mean():.4f}")
print(f"  Std: {df_scaled['Age_Standardized'].std():.4f}")

# 2. MIN-MAX NORMALIZATION
scaler_minmax = MinMaxScaler()
df_scaled['Fare_MinMax'] = scaler_minmax.fit_transform(df[['Fare']])
print(f"\n✓ MinMaxScaler - Fare:")
print(f"  Min: {df_scaled['Fare_MinMax'].min():.4f}")
print(f"  Max: {df_scaled['Fare_MinMax'].max():.4f}")

# 3. ROBUST SCALING (resistant to outliers)
scaler_robust = RobustScaler()
df_scaled['Fare_Robust'] = scaler_robust.fit_transform(df[['Fare']])
print(f"\n✓ RobustScaler - Fare:")
print(f"  Median: {df_scaled['Fare_Robust'].median():.4f}")
print(f"  IQR: {df_scaled['Fare_Robust'].quantile(0.75) - df_scaled['Fare_Robust'].quantile(0.25):.4f}")

# 4. LOG TRANSFORMATION (for skewed distributions)
df_scaled['Fare_Log'] = np.log1p(df['Fare'])
print(f"\n✓ Log Transform - Fare:")
print(f"  Original skewness: {df['Fare'].skew():.4f}")
print(f"  Log-transformed skewness: {df_scaled['Fare_Log'].skew():.4f}")

"""
=============================================================================
PART 6: FEATURE SELECTION
=============================================================================
"""

print("\n" + "=" * 80)
print("FEATURE SELECTION TECHNIQUES")
print("=" * 80)

# Prepare features for selection
feature_cols = ['Pclass', 'Sex_Label', 'Age', 'SibSp', 'Parch', 'Fare', 
                'FamilySize', 'IsAlone', 'FarePerPerson', 'IsChild', 
                'Has_Cabin', 'Embarked_Label']

X = df_encoded[feature_cols].fillna(0)
y = df_encoded['Survived']

# 1. CORRELATION-BASED SELECTION
correlation_matrix = X.corrwith(y).abs().sort_values(ascending=False)
print("\n✓ Correlation with Target (Top 5):")
print(correlation_matrix.head())

high_corr_features = correlation_matrix[correlation_matrix > 0.1].index.tolist()
print(f"\nFeatures with |correlation| > 0.1: {len(high_corr_features)}")

# 2. UNIVARIATE FEATURE SELECTION (Chi-squared)
# Make all features non-negative for chi2
X_positive = X - X.min() + 1
selector_chi2 = SelectKBest(chi2, k=5)
X_chi2 = selector_chi2.fit_transform(X_positive, y)
chi2_features = X.columns[selector_chi2.get_support()].tolist()
print(f"\n✓ Chi-squared Selection (Top 5):")
print(chi2_features)

# 3. MUTUAL INFORMATION
selector_mi = SelectKBest(mutual_info_classif, k=5)
X_mi = selector_mi.fit_transform(X, y)
mi_features = X.columns[selector_mi.get_support()].tolist()
print(f"\n✓ Mutual Information (Top 5):")
print(mi_features)

# 4. RECURSIVE FEATURE ELIMINATION (RFE)
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rfe = RFE(estimator=rf_model, n_features_to_select=5)
rfe.fit(X, y)
rfe_features = X.columns[rfe.support_].tolist()
print(f"\n✓ RFE (Top 5):")
print(rfe_features)

# 5. FEATURE IMPORTANCE FROM TREE-BASED MODELS
rf_model.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n✓ Random Forest Feature Importance (Top 5):")
print(feature_importance.head())

# 6. L1-BASED FEATURE SELECTION
from sklearn.linear_model import LogisticRegression
selector_l1 = SelectFromModel(
    LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
    threshold='median'
)
selector_l1.fit(X, y)
l1_features = X.columns[selector_l1.get_support()].tolist()
print(f"\n✓ L1 Regularization Selection:")
print(f"Selected {len(l1_features)} features: {l1_features}")

"""
=============================================================================
PART 7: ADVANCED FEATURE ENGINEERING TECHNIQUES
=============================================================================
"""

print("\n" + "=" * 80)
print("ADVANCED FEATURE ENGINEERING")
print("=" * 80)

# 1. AGGREGATION FEATURES
df['MeanFare_PerClass'] = df.groupby('Pclass')['Fare'].transform('mean')
df['MeanAge_PerClass'] = df.groupby('Pclass')['Age'].transform('mean')
print("\n✓ Aggregation Features:")
print("  - Mean Fare per Passenger Class")
print("  - Mean Age per Passenger Class")

# 2. RATIO FEATURES
df['FareToMeanRatio'] = df['Fare'] / (df['MeanFare_PerClass'] + 1)
df['AgeToMeanRatio'] = df['Age'] / (df['MeanAge_PerClass'] + 1)
print("\n✓ Ratio Features:")
print("  - Individual fare to class average ratio")
print("  - Individual age to class average ratio")

# 3. CLUSTERING-BASED FEATURES
from sklearn.cluster import KMeans

cluster_features = ['Age', 'Fare', 'FamilySize']
X_cluster = df[cluster_features].fillna(df[cluster_features].median())
kmeans = KMeans(n_clusters=3, random_state=42)
df['PassengerCluster'] = kmeans.fit_predict(X_cluster)
print("\n✓ Clustering Features:")
print(f"  - Created 3 passenger clusters based on Age, Fare, FamilySize")
print(f"  - Cluster distribution: {df['PassengerCluster'].value_counts().to_dict()}")

# 4. DATETIME FEATURES (simulated)
# Note: Titanic doesn't have dates, but this shows the technique
df['SimulatedDate'] = pd.date_range(start='1912-04-10', periods=len(df), freq='H')
df['DayOfWeek'] = df['SimulatedDate'].dt.dayofweek
df['Hour'] = df['SimulatedDate'].dt.hour
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
print("\n✓ DateTime Features (Simulated):")
print("  - Day of week")
print("  - Hour of day")
print("  - Weekend indicator")

# 5. OUTLIER DETECTION FEATURES
from scipy import stats

df['Age_Zscore'] = np.abs(stats.zscore(df['Age'].fillna(df['Age'].median())))
df['Age_IsOutlier'] = (df['Age_Zscore'] > 3).astype(int)

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df['Fare_IsOutlier'] = ((df['Fare'] < (Q1 - 1.5 * IQR)) | 
                         (df['Fare'] > (Q3 + 1.5 * IQR))).astype(int)
print("\n✓ Outlier Detection Features:")
print(f"  - Age outliers (Z-score > 3): {df['Age_IsOutlier'].sum()}")
print(f"  - Fare outliers (IQR method): {df['Fare_IsOutlier'].sum()}")

"""
=============================================================================
PART 8: FEATURE ENGINEERING SUMMARY & BEST PRACTICES
=============================================================================
"""

print("\n" + "=" * 80)
print("SUMMARY OF ENGINEERED FEATURES")
print("=" * 80)

all_features = df.columns.tolist()
original_features = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 
                     'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
engineered_features = [f for f in all_features if f not in original_features]

print(f"\nOriginal Features: {len(original_features)}")
print(f"Engineered Features: {len(engineered_features)}")
print(f"Total Features: {len(all_features)}")

print("\n" + "=" * 80)
print("BEST PRACTICES & KEY TAKEAWAYS")
print("=" * 80)

best_practices = """
1. DOMAIN KNOWLEDGE
   - Understand the problem domain to create meaningful features
   - Consult domain experts when possible
   
2. MISSING VALUE HANDLING
   - Always analyze missing data patterns before imputing
   - Consider creating missing indicators as features
   - Don't always drop missing values - they might be informative
   
3. FEATURE CREATION
   - Combine existing features to create interactions
   - Extract information from text and categorical data
   - Create aggregations and statistical features
   
4. ENCODING STRATEGIES
   - Use label encoding for ordinal variables
   - Use one-hot encoding for nominal variables (watch for high cardinality)
   - Consider target encoding for high-cardinality features
   
5. SCALING & NORMALIZATION
   - Always scale features for distance-based algorithms
   - Use StandardScaler for normally distributed data
   - Use RobustScaler when you have outliers
   - MinMaxScaler when you need bounded ranges
   
6. FEATURE SELECTION
   - Remove highly correlated features
   - Use multiple selection methods and compare
   - Consider computational cost vs. performance gain
   - Keep feature selection separate for train/test
   
7. VALIDATION
   - Always validate on holdout data
   - Use cross-validation to ensure features generalize
   - Monitor for overfitting
   
8. ITERATION
   - Feature engineering is iterative
   - Start simple, then add complexity
   - Track which features improve model performance
   
9. AVOID DATA LEAKAGE
   - Fit transformers only on training data
   - Never use target variable to create features (except proper target encoding)
   - Be careful with time-series data
   
10. DOCUMENTATION
    - Document all feature transformations
    - Keep track of feature engineering steps for reproducibility
"""

print(best_practices)

print("\n" + "=" * 80)
print("FEATURE ENGINEERING PIPELINE COMPLETE")
print("=" * 80)
print(f"\nFinal dataset shape: {df.shape}")
print(f"Ready for model training!")

# Display sample of final features
print("\nSample of engineered dataset:")
display_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'FamilySize', 'IsAlone', 
                'Title', 'FarePerPerson', 'Has_Cabin', 'PassengerCluster']
print(df[display_cols].head(10))

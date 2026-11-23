# EDA Teaching Notes — Based on `sample.ipynb`

Companion notes for the live-examples notebook. Sections follow the exact order of the notebook cells.

---

## 1. Setting Up the Environment

```python
pip install seaborn
```

| Library | Purpose |
|---|---|
| `pandas` | Tabular data handling (Series, DataFrame) |
| `seaborn` | Statistical plots, built on matplotlib |
| `matplotlib.pyplot` | Base plotting engine / figure control |

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

**Teaching points**

- `%matplotlib inline` is a *magic command* — it renders plots inside the notebook instead of a pop-up window.
- Install with `!pip install seaborn` (bang) inside a notebook cell; `%pip install seaborn` is the modern, kernel-safe form.
- Import aliases (`pd`, `sns`, `plt`) are community conventions — always follow them.

---

## 2. Creating Data Manually — `pandas.Series`

```python
import random
marks = [random.randint(45, 100) for i in range(1, 200)]
```

**Concept: Series**

- A `Series` is a **1-dimensional labelled array** — think of it as a single column of a table.
- It has two parts: the **index** (labels) and the **values**.

```python
ser = pd.Series(data=marks)
```

**Teaching points**

- `random.randint(45, 100)` is **inclusive** on both ends.
- `range(1, 200)` produces 199 items — not 200. Great place to discuss off-by-one errors.
- Default index is `RangeIndex` starting from 0.

### 2.1 `describe()` — the five-number summary

```python
ser.describe()
```

| Statistic | Meaning |
|---|---|
| `count` | Number of non-null values |
| `mean` | Arithmetic average |
| `std` | Standard deviation (spread around the mean) |
| `min` | Smallest value |
| `25%` (Q1) | 25% of data lies below this |
| `50%` (Q2) | Median |
| `75%` (Q3) | 75% of data lies below this |
| `max` | Largest value |

**Expected result here:** since the data is *uniformly* random between 45–100, mean ≈ median ≈ 72.5. This is a nice demonstration of a **symmetric distribution**.

### 2.2 Quick plot

```python
ser.plot()
```

- Default `kind='line'` → index on X-axis, value on Y-axis.
- For random data this looks like noise — a good teaching moment: **a line plot is meaningless without a meaningful ordering (e.g., time).**
- Better choice here: `ser.plot(kind='hist')` or `sns.histplot(ser)`.

---

## 3. Loading Real Data

```python
subcribers_df = pd.read_csv(r"D:\Data Science_bat\Pandas_topic\subscribers.csv")
```

**Teaching points**

- The `r""` prefix creates a **raw string** so Windows backslashes are not read as escape characters (`\n`, `\t`).
- Alternatives: forward slashes `"D:/data/file.csv"` or double backslashes.
- Prefer **relative paths** in shared projects so notebooks run on any machine.
- Other readers: `read_excel`, `read_json`, `read_sql`, `read_parquet`.

### 3.1 First look at a DataFrame

```python
subcribers_df          # displays head + tail
subcribers_df.head()   # first 5 rows
subcribers_df.tail()   # last 5 rows
subcribers_df.shape    # (rows, columns)
subcribers_df.columns  # column names
```

### 3.2 `info()` — structure & data types

```python
subcribers_df.info()
```

Tells you:

1. Number of rows (`RangeIndex`)
2. Column names
3. **Non-null count** → instantly reveals missing data
4. **Dtype** of each column
5. Memory usage

**Key idea:** dates loaded from CSV arrive as `object` (string). They must be converted before any date arithmetic.

---

## 4. Data Cleaning — Fixing Data Types

```python
subcribers_df["subscription_date"] = subcribers_df["subscription_date"].astype("datetime64[ns]")
subcribers_df["plan_change_date"]  = subcribers_df["plan_change_date"].astype("datetime64[ns]")
subcribers_df["last_active_date"]  = subcribers_df["last_active_date"].astype("datetime64[ns]")
```

**Why convert?** Only after conversion can you:

- Subtract dates → `Timedelta` (e.g., subscription tenure)
- Use `.dt` accessor: `.dt.year`, `.dt.month`, `.dt.day_name()`
- Resample / group by time period
- Sort chronologically instead of alphabetically

**More robust alternative** (handles bad/blank values gracefully):

```python
date_cols = ["subscription_date", "plan_change_date", "last_active_date"]
for col in date_cols:
    subcribers_df[col] = pd.to_datetime(subcribers_df[col], errors="coerce")
```

- `errors="coerce"` turns unparseable values into `NaT` (Not a Time) instead of raising an error.
- You can also parse at load time: `pd.read_csv(path, parse_dates=date_cols)`.

Always re-run `.info()` after conversion to **verify** the change.

---

## 5. Summary Statistics for Mixed Data

```python
subcribers_df.describe(include="all")
```

| Argument | Behaviour |
|---|---|
| default | Numeric columns only |
| `include="all"` | All columns; irrelevant stats become `NaN` |
| `include="object"` | Text/categorical only |
| `include=["int64","float64"]` | Specific dtypes |

**Extra stats shown for categorical columns**

- `unique` — number of distinct values
- `top` — most frequent value (mode)
- `freq` — how many times `top` appears

---

## 6. Univariate Analysis of a Categorical Column

```python
subcribers_df['subscription_plan'].value_counts().plot(kind="barh")
```

**Chain explained**

1. `['subscription_plan']` → select one column (a Series)
2. `.value_counts()` → frequency of each category, sorted descending
3. `.plot(kind="barh")` → horizontal bar chart

**Teaching points**

- `value_counts(normalize=True)` gives proportions/percentages.
- `value_counts(dropna=False)` includes `NaN` in the counts — very useful for auditing missing data.
- Use **horizontal** bars when category names are long.
- Bar charts are for *categorical* data; histograms are for *continuous* data.

---

## 7. Splitting Columns by Data Type

```python
insurance_df = pd.read_csv(r"D:\Data Science_bat\Machine Learning\Raw_datasets\insurance.csv")

numerical_df   = insurance_df.select_dtypes(include=["int64", "float64"])
categorical_df = insurance_df.select_dtypes(include=["object"])
```

**Why split?** EDA techniques differ by type:

| Type | Summary | Plots |
|---|---|---|
| Numerical | `describe()`, mean, median, std | histogram, boxplot, KDE, scatter |
| Categorical | `value_counts()`, mode | countplot, bar chart, pie |

**Safer, dtype-agnostic version**

```python
numerical_df   = insurance_df.select_dtypes(include="number")
categorical_df = insurance_df.select_dtypes(include=["object", "category"])
```

For the insurance dataset:

- Numerical → `age`, `bmi`, `children`, `charges`
- Categorical → `sex`, `smoker`, `region`

> **Caution:** `children` is stored as an integer but is really **discrete/ordinal**. Statistics like its "mean" are less meaningful. Discuss *storage type vs. semantic type*.

---

## 8. Built-in Datasets (`tips`)

```python
tips_df = sns.load_dataset("tips")
tips_df.info()
```

- Seaborn ships small practice datasets: `tips`, `titanic`, `iris`, `penguins`, `flights`, `diamonds`.
- Note that in `tips`, the text columns are already dtype **`category`**, not `object`.

```python
numerical_df   = tips_df.select_dtypes(include=["int64", "float64"])
categorical_df = tips_df.select_dtypes(include=["category"])
```

**Important teaching moment:** these two lines **overwrite** the earlier insurance-based variables. This is a classic notebook pitfall — later cells that assume `categorical_df` holds insurance data will silently give wrong results.

> **Rule for students:** use descriptive, dataset-prefixed names — `ins_num_df`, `tips_cat_df` — and remember that notebook state depends on *execution order*, not cell order.

**Why `category` dtype matters**

- Uses far less memory than `object`
- Can carry an explicit order (`Ordered Categorical`)
- Some libraries treat it specially

---

## 9. Handling Missing Values

```python
insurance_df.isnull().sum()
```

**Chain explained**

- `isnull()` → DataFrame of `True`/`False`
- `.sum()` → column-wise count (True counts as 1)

Useful variants:

```python
insurance_df.isnull().sum().sum()                 # total missing cells
insurance_df.isnull().mean() * 100                # % missing per column
sns.heatmap(insurance_df.isnull(), cbar=False)    # visual missing-value map
```

### 9.1 Imputation

```python
avg_age = insurance_df['age'].mean()
insurance_df.fillna(avg_age)
```

**Two bugs to discuss with students:**

1. **Not saved.** `fillna()` returns a *new* DataFrame. Without assignment (or `inplace=True`), the original is unchanged.
2. **Wrong scope.** This fills *every* column with the mean **age** — filling `charges` or `bmi` with an age value is nonsense.

**Correct approach — impute per column:**

```python
insurance_df['age']    = insurance_df['age'].fillna(insurance_df['age'].median())
insurance_df['bmi']    = insurance_df['bmi'].fillna(insurance_df['bmi'].median())
insurance_df['region'] = insurance_df['region'].fillna(insurance_df['region'].mode()[0])
```

**Choosing a strategy**

| Situation | Strategy |
|---|---|
| Numeric, roughly symmetric | Mean |
| Numeric, skewed or has outliers | **Median** (safer default) |
| Categorical | Mode / a new `"Unknown"` category |
| Time series | Forward fill `ffill()` / interpolation |
| Very high % missing | Consider dropping the column |
| Few missing rows | `dropna()` |

---

## 10. Mean vs. Median — Why Outliers Matter

```python
heights = [5, 6, 4.5, 4.7, 5, 7, 120]
sum(heights) / len(heights)     # ≈ 21.7  → nonsense!
```

```python
h_sr = pd.Series(heights)
h_sr.mean()     # ≈ 21.74  — dragged up by 120
h_sr.median()   # 5.0      — unaffected
```

**Core lesson**

- The **mean is sensitive** to extreme values; the **median is robust**.
- `120` is clearly a **data-entry error** (unit mix-up: cm vs. ft).
- Rule of thumb: if `mean ≫ median`, the data is **right-skewed** / has high outliers.
- This is why we prefer the **median** for imputation and for reporting skewed values like income or medical charges.

---

## 11. Visualising Outliers

```python
sns.boxplot(insurance_df['charges'], orient='h')
sns.boxenplot(insurance_df['charges'], orient='h')
```

### Reading a boxplot

```text
    |-----[   |   ]-----|    o   o    o
   min    Q1  Q2   Q3  max          outliers
 (whisker)        (whisker)
```

- **Box** = IQR, spans Q1 → Q3, containing the middle 50% of data
- **Line inside box** = median
- **Whiskers** = extend to the furthest point within 1.5 × IQR
- **Dots beyond whiskers** = candidate outliers

### `boxplot` vs `boxenplot`

| Plot | Best for | Detail |
|---|---|---|
| `boxplot` | General use, small/medium data | Shows quartiles only |
| `boxenplot` (letter-value plot) | Large datasets | Shows many more quantile levels → better view of tail shape |

**Modern seaborn syntax** (clearer and avoids deprecation warnings):

```python
sns.boxplot(data=insurance_df, x='charges')
```

**Interpretation for `charges`:** the distribution is strongly **right-skewed** — most people have low medical costs, a small group (likely smokers / chronic conditions) has very high costs.

---

## 12. Detecting Outliers with the IQR Rule

```python
charges_info = insurance_df['charges'].describe()

Q1 = charges_info['25%']
Q3 = charges_info['75%']

IQR = Q3 - Q1
print(IQR)

lower_boundry = Q1 - (1.5 * IQR)
upper_boundry = Q3 + (1.5 * IQR)
```

**The formulas**

$$IQR = Q3 - Q1$$
$$\text{Lower fence} = Q1 - 1.5 \times IQR$$
$$\text{Upper fence} = Q3 + 1.5 \times IQR$$

**Teaching points**

- `describe()` returns a Series, so quartiles can be pulled out by label: `charges_info['25%']`.
- Equivalent direct method: `insurance_df['charges'].quantile(0.25)`.
- Why 1.5? It is a convention (Tukey) that flags roughly the most extreme ~0.7% of a normal distribution. Use **3.0** for "extreme" outliers only.
- For `charges`, the lower fence comes out **negative** — impossible for a cost. So *no* low outliers exist; all outliers are on the high side.

### 12.1 Filtering out the outliers

```python
outlier_removed_df = insurance_df[
    (insurance_df['charges'] < upper_boundry) &
    (insurance_df['charges'] > lower_boundry)
]
```

**Boolean masking explained**

1. Each comparison produces a `True`/`False` Series.
2. `&` combines them element-wise (**must** use `&`, not `and`).
3. **Parentheses are mandatory** — `&` binds tighter than `<`.
4. `df[mask]` keeps only the `True` rows.

Cleaner equivalent:

```python
mask = insurance_df['charges'].between(lower_boundry, upper_boundry)
outlier_removed_df = insurance_df[mask].copy()
```

> **Should you always remove outliers?** No. In this dataset, high charges are *genuine* high-cost patients, not errors — removing them may destroy the very signal you want to model. Remove outliers when they are **errors**; investigate them when they are **real**.

---

## 13. Duplicates

```python
outlier_removed_df.duplicated().sum()
outlier_removed_df.drop_duplicates(inplace=True)
outlier_removed_df.shape
```

**Teaching points**

- `duplicated()` marks a row `True` if an *identical* row appeared earlier.
- `keep='first'` (default) / `'last'` / `False` (mark all copies).
- `subset=['col1','col2']` checks duplicates on selected columns only (e.g., duplicate customer IDs).

> **`SettingWithCopyWarning`.** `outlier_removed_df` was created by slicing `insurance_df`, so it may be a *view*. Calling `inplace=True` on it triggers a warning and may not work reliably.
>
> **Fix:** add `.copy()` when slicing, or reassign:
>
> ```python
> outlier_removed_df = outlier_removed_df.drop_duplicates()
> ```

- `.shape` afterwards confirms how many rows were dropped — **always verify a cleaning step**.

---

## 14. Looping Over Columns — Automating EDA

### 14.1 Frequency of every categorical column

```python
for col in categorical_df:
    print(categorical_df[col].value_counts())
```

- Iterating a DataFrame directly yields **column names**.
- More explicit and readable: `for col in categorical_df.columns:`
- Add a header for readability:

```python
for col in categorical_df.columns:
    print(f"--- {col} ---")
    print(categorical_df[col].value_counts())
    print()
```

### 14.2 Count plots for all categorical columns

```python
for col in categorical_df:
    sns.countplot(categorical_df[col])
    plt.show()
```

- `sns.countplot` = bar chart of `value_counts()`.
- `plt.show()` **inside** the loop is essential — without it, all plots draw on one figure.
- Preferred syntax with a title:

```python
for col in categorical_df.columns:
    sns.countplot(data=categorical_df, x=col)
    plt.title(f"Distribution of {col}")
    plt.show()
```

### 14.3 Boxen plots for all numerical columns

```python
for col in numerical_df:
    sns.boxenplot(numerical_df[col], orient="h")
    plt.show()
```

Purpose: scan **every** numeric feature for outliers and skew in one pass.

Add histograms for shape:

```python
for col in numerical_df.columns:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    sns.histplot(numerical_df[col], kde=True, ax=ax[0])
    sns.boxplot(x=numerical_df[col], ax=ax[1])
    fig.suptitle(col)
    plt.show()
```

---

## 15. The EDA Workflow — Summary Checklist

| # | Step | Key commands |
|---|---|---|
| 1 | Load data | `pd.read_csv()` |
| 2 | Inspect shape & structure | `.head()`, `.shape`, `.info()` |
| 3 | Fix data types | `.astype()`, `pd.to_datetime()` |
| 4 | Summary statistics | `.describe(include="all")` |
| 5 | Split by type | `.select_dtypes()` |
| 6 | Check missing values | `.isnull().sum()` |
| 7 | Impute / drop | `.fillna()`, `.dropna()` |
| 8 | Check duplicates | `.duplicated().sum()`, `.drop_duplicates()` |
| 9 | Univariate — categorical | `.value_counts()`, `sns.countplot()` |
| 10 | Univariate — numerical | `sns.histplot()`, `sns.boxplot()` |
| 11 | Outlier detection | IQR rule + boolean masking |
| 12 | Bivariate analysis | `sns.scatterplot()`, `sns.barplot()`, `.corr()` |

---

## 16. Common Mistakes Highlighted by This Notebook

1. **Variable shadowing** — `numerical_df` / `categorical_df` redefined for a different dataset.
2. **Forgetting to assign** — `df.fillna(x)` without `df = df.fillna(x)`.
3. **Blanket imputation** — filling all columns with one column's mean.
4. **`SettingWithCopyWarning`** — using `inplace=True` on a slice.
5. **Missing `.copy()`** when creating a filtered DataFrame.
6. **Using `and`/`or`** instead of `&`/`|` in boolean masks.
7. **Removing outliers blindly** without asking whether they are errors or signal.
8. **`range(1, 200)`** producing 199 items when 200 were intended.

---

## 17. Suggested Exercises for Students

1. Compute the average subscription tenure in days from `subscription_date` and `last_active_date`.
2. Compare mean vs. median `charges` for smokers and non-smokers — which differs more, and why?
3. Write a reusable function `iqr_outliers(df, col)` returning the fences and the outlier rows.
4. Build a correlation heatmap of `insurance_df` numeric columns using `sns.heatmap(df.corr(numeric_only=True), annot=True)`.
5. Create a `sns.pairplot(insurance_df, hue='smoker')` and describe three insights.
6. Rewrite the missing-value handling so each column uses an appropriate strategy.

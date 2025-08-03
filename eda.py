import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('datasets/laliga_player_stats.csv')

# Show first few rows
print('Head:')
print(df.head())

# Show info
print('\nInfo:')
print(df.info())

# Describe statistics
print('\nDescribe:')
print(df.describe(include='all'))

# Check for missing values
print('\nMissing values:')
print(df.isnull().sum())



# Check for duplicates
print('\nDuplicate rows:', df.duplicated().sum())

# Value counts for categorical columns
for col in df.select_dtypes(include='object').columns:
    print(f'\nValue counts for {col}:')
    print(df[col].value_counts().head())

# Correlation heatmap for numeric columns
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

plt.savefig('correlation_heatmap.png')

# Histograms for numeric columns
df.hist(figsize=(12, 8), bins=20)
plt.suptitle('Histograms of Numeric Features')
plt.tight_layout()
plt.show()
plt.savefig('histograms_numeric_features.png')




# 1. Top Performers: Top 5 by goals, assists, or minutes played
print('\nTop 5 performers:')
if 'Goals scored' in df.columns:
    print('By goals:')
    print(df.sort_values('Goals scored', ascending=False)[['Name', 'Team', 'Goals scored']].head(5))
if 'Assists' in df.columns:
    print('By assists:')
    print(df.sort_values('Assists', ascending=False)[['Name', 'Team', 'Assists']].head(5))
if 'Minutes played' in df.columns:
    print('By minutes played:')
    print(df.sort_values('Minutes played', ascending=False)[['Name', 'Team', 'Minutes played']].head(5))

# 2. Missing Data: Columns with missing values
missing = df.isnull().sum()
missing_cols = missing[missing > 0]
print('\nColumns with missing values:')
print(missing_cols)
if not missing_cols.empty:
    print('Columns with missing data may affect analysis. Consider imputing or dropping.')
else:
    print('No missing data detected.')

# 3. Correlation: Strongly correlated numeric features
correlation = df.corr(numeric_only=True)
strong_corr = correlation[(correlation.abs() > 0.7) & (correlation.abs() < 1.0)]
print('\nStrong correlations (|corr| > 0.7):')
print(strong_corr.dropna(how='all').dropna(axis=1, how='all'))

# 4. Distribution: Skewness and outliers for numeric columns
print('\nDistribution skewness:')
print(df.skew(numeric_only=True))
for col in df.select_dtypes(include='number').columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
    print(f'Outliers in {col}:', len(outliers))


# Split the data into train and test sets (80/20 split)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f'\nTrain set shape: {train_df.shape}')
print(f'Test set shape: {test_df.shape}')

import pandas as pd
import seaborn as sns

# Load Titanic dataset
df = sns.load_dataset('titanic')

print("Original Dataset:")
print(df.head())

# 1. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 2. Remove duplicate rows
df = df.drop_duplicates()

# 3. Fill missing numerical values with median
df['age'] = df['age'].fillna(df['age'].median())
df['fare'] = df['fare'].fillna(df['fare'].median())

# 4. Fill missing categorical values with mode
df['embarked'] = df['embarked'].fillna(
    df['embarked'].mode()[0]
)

# 5. Drop column with excessive missing values
df = df.drop(columns=['deck'])

# 6. Rename columns
df = df.rename(columns={
    'sex': 'gender',
    'survived': 'survival'
})

# 7. Convert data type
df['survival'] = df['survival'].astype(int)

# 8. Check missing values after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 9. Display cleaned dataset
print("\nCleaned Dataset:")
print(df.head())

print("\nDataset Shape:", df.shape)

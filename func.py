import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

# Constants

SPENDING_COLS = [
    'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts'
]
 
PURCHASE_COLS = [
    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases'
]

# 1. Data Loading & Cleaning

def load_data(filepath: str) -> pd.DataFrame:

    df = pd.read_excel(filepath, sheet_name='marketing_campaign')
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {df.columns.tolist()}")
    return df

def drop_missing_income(df: pd.DataFrame) -> pd.DataFrame:

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print('Missing values:')
    print(missing)

    df = df.dropna(subset=['Income'])
    print(f'\nMissing after: {df.isnull().sum().sum()}')
    print(f'Remaining rows: {len(df)}')
    return df

def remove_outliers(df: pd.DataFrame,
                    max_age: int = 100,
                    max_income: int = 600_000) -> pd.DataFrame:

    df = df[(df['Age'] <= max_age) & (df['Income'] <= max_income)]
    return df

# 2. Feature Engineering

def create_features(df: pd.DataFrame, reference_year: int = 2026) -> pd.DataFrame:
    """
    New columns created:
    - TotalSpending  : sum of all spending category columns
    - Age            : reference_year minus Year_Birth
    - Children       : Kidhome + Teenhome
    - TotalPurchases : sum of web, catalog and store purchases
    - FamilySize     : 1 (self) + partner indicator + children count
    """
    
    df = df.copy()

    df['TotalSpending'] = df[SPENDING_COLS].sum(axis=1)
    df['Age'] = reference_year - df['Year_Birth']
    df['Children'] = df['Kidhome'] + df['Teenhome']
    df['TotalPurchases'] = df[PURCHASE_COLS].sum(axis=1)
    df['FamilySize'] = (
        1
        + df['Marital_Status'].isin(['Married', 'Together']).astype(int)
        + df['Kidhome']
        + df['Teenhome']
    )

    print(df[['TotalSpending', 'Age', 'Children',
              'TotalPurchases', 'FamilySize']].describe().round(1))
    return df

def group_education(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mapping:
    - 'High' : Graduation, Master, PhD
    - 'Low'  : Basic, 2n Cycle
    """

    mapping = {
        'Basic'     : 'Low',
        '2n Cycle'  : 'Low',
        'Graduation': 'High',
        'Master'    : 'High',
        'PhD'       : 'High',
    }
    df = df.copy()
    df['Education_grouped'] = df['Education'].map(mapping)
    print(df['Education_grouped'].value_counts())
    return df

def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:

    # to avoid ValueError when passed to statsmodels OLS
    df_encoded = pd.get_dummies(df, columns=['Education_grouped'], drop_first=True)
    bool_cols = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
 
    edu_cols = [c for c in df_encoded.columns if c.startswith('Education_grouped_')]
    return df_encoded, edu_cols

# 3. Visualisation

def plot_spending_distribution(df: pd.DataFrame) -> None:

    plt.figure(figsize=(8, 5))
    plt.hist(df['TotalSpending'], bins=50, color='steelblue', edgecolor='white')
    plt.title('Total Spending Distribution')
    plt.xlabel('Total Spending ($)')
    plt.ylabel('Number of customers')
    plt.tight_layout()
    plt.show()

    print(f'Mean    : ${df["TotalSpending"].mean():.0f}')
    print(f'Median  : ${df["TotalSpending"].median():.0f}')
    print(f'Skewness: {df["TotalSpending"].skew():.3f}')

def plot_income_vs_spending(df: pd.DataFrame) -> None:

    plt.figure(figsize=(8, 5))
    plt.scatter(df['Income'], df['TotalSpending'],
                alpha=0.3, s=15, color='steelblue')
    plt.xlabel('Income ($)')
    plt.ylabel('TotalSpending ($)')
    plt.title('Income vs TotalSpending')
    plt.tight_layout()
    plt.show()

    corr = df['Income'].corr(df['TotalSpending'])
    print(f'Correlation: {corr:.3f}')

# 4. Modelling
 
def run_ols(y: pd.Series, X: pd.DataFrame) -> RegressionResultsWrapper:
    """
    Fit an OLS regression model using statsmodels.
    """
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    return model
 
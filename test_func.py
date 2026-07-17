import pandas as pd
from func import (
    drop_missing_income,
    remove_outliers,
    create_features,
    group_education,
    encode_categoricals,
    run_ols,
    load_data,
)

## DataFrames for testing (fake only)

def make_sample_df() -> pd.DataFrame:

    return pd.DataFrame({
        'Year_Birth'      : [1980, 1975, 1990, 1800],   
        'Education'       : ['Graduation', 'PhD', 'Basic', 'Master'],
        'Marital_Status'  : ['Married', 'Single', 'Together', 'Divorced'],
        'Income'          : [50000.0, None, 80000.0, 700000.0],
        'Kidhome'         : [1, 0, 2, 0],
        'Teenhome'        : [0, 1, 0, 0],
        'Recency'         : [10, 30, 5, 60],
        'MntWines'        : [200, 50, 10, 300],
        'MntFruits'       : [30, 10, 5, 20],
        'MntMeatProducts' : [100, 40, 8, 150],
        'MntFishProducts' : [20, 5, 2, 30],
        'MntSweetProducts': [15, 8, 3, 25],
        'NumWebPurchases' : [3, 1, 2, 5],
        'NumCatalogPurchases': [2, 0, 1, 4],
        'NumStorePurchases'  : [4, 2, 3, 6],
    })

def test_drop_missing_income() -> None:

    df = make_sample_df()
    result = drop_missing_income(df)

    assert result['Income'].isnull().sum() == 0, "Missing Income rows not removed"
    assert len(result) == 3, f"Expected 3 rows after drop, got {len(result)}"
    print("PASS — test_drop_missing_income")

    # drop_missing_income should remove exactly the rows where Income is NaN.

def test_create_features_total_spending() -> None:

    df = make_sample_df().dropna(subset=['Income'])
    result = create_features(df)

    expected = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] \
             + df['MntFishProducts'] + df['MntSweetProducts']

    assert (result['TotalSpending'].values == expected.values).all(), \
        "TotalSpending does not match sum of Mnt* columns"
    print("PASS — test_create_features_total_spending")

    # TotalSpending should equal the sum of the five Mnt* columns.
    
def test_create_features_family_size() -> None:

    df = make_sample_df().dropna(subset=['Income'])
    result = create_features(df)

    # Row 0: Married(1) + Kidhome(1) + Teenhome(0) + self(1) = 3
    assert result['FamilySize'].iloc[0] == 3, \
        f"Expected FamilySize=3, got {result['FamilySize'].iloc[0]}"
    # Row 2: Together(1) + Kidhome(2) + Teenhome(0) + self(1) = 4
    assert result['FamilySize'].iloc[1] == 4, \
        f"Expected FamilySize=4, got {result['FamilySize'].iloc[1]}"
    print("PASS — test_create_features_family_size")


def test_group_education() -> None:

    df = make_sample_df().dropna(subset=['Income'])
    result = group_education(df)

    valid_groups = {'High', 'Low'}
    unique = set(result['Education_grouped'].dropna().unique())
    assert unique.issubset(valid_groups), \
        f"Unexpected groups: {unique - valid_groups}"
    print("PASS — test_group_education")

    # group_education should map all 5 categories to High or Low.

def test_remove_outliers() -> None:

    df = make_sample_df().dropna(subset=['Income'])
    df = create_features(df)           # adds Age column
    result = remove_outliers(df)

    assert result['Age'].max() <= 100, "Age outlier not removed"
    assert result['Income'].max() <= 600_000, "Income outlier not removed"
    print("PASS — test_remove_outliers")

    # rows with Age > 100 or Income > 600,000 should be removed.

def test_encode_categoricals_no_bool() -> None:

    df = make_sample_df().dropna(subset=['Income'])
    df = group_education(df)
    df_enc, edu_cols = encode_categoricals(df)

    for col in edu_cols:
        assert df_enc[col].dtype != bool, \
            f"Column {col} is bool — should be int"
    print("PASS — test_encode_categoricals_no_bool")

    # encoded dummy columns should be integer dtype, not bool.

## Integration Test — full pipeline end-to-end

def test_full_pipeline() -> None:

    df = make_sample_df()

    # clean
    df = drop_missing_income(df)
    df = create_features(df)
    df = remove_outliers(df)

    df = group_education(df)
    df_enc, edu_cols = encode_categoricals(df)

    # model
    y = df_enc['TotalSpending']
    X = df_enc[['Income', 'FamilySize']]
    result = run_ols(y, X)

    # Assertions
    assert 0 <= result.rsquared <= 1, \
        f"R² out of range: {result.rsquared}"
    assert "Income" in result.params, \
        "Income should be in model parameters"
    assert len(result.params) == 3, \
        "Expected 3 params: const, Income, FamilySize"

    print(f"PASS — test_full_pipeline  (R²={result.rsquared:.3f})")

## Run all tests

if __name__ == '__main__':
    print("=" * 50)
    print("Running unit tests...")
    print("=" * 50)
    test_drop_missing_income()
    test_create_features_total_spending()
    test_create_features_family_size()
    test_group_education()
    test_remove_outliers()
    test_encode_categoricals_no_bool()
 
    print()
    print("=" * 50)
    print("Running integration test...")
    print("=" * 50)
    test_full_pipeline()
 
    print()
    print("All tests passed")
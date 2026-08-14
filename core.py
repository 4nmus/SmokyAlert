"""Train and analyse the company risk-score regression model."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).with_name("data.csv")
TARGET_COLUMN = "risk_score"
EXCLUDED_COLUMNS = ("company_id", "risk_class")
RANDOM_STATE = 6


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path)
    required_columns = {TARGET_COLUMN, *EXCLUDED_COLUMNS}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if data.empty:
        raise ValueError("Dataset contains no rows")

    model_columns = data.drop(columns=[TARGET_COLUMN, *EXCLUDED_COLUMNS]).columns
    non_numeric = data[model_columns].select_dtypes(exclude="number").columns
    if len(non_numeric):
        raise ValueError(
            "Model features must be numeric: " + ", ".join(non_numeric)
        )

    return data


def print_data_analysis(data: pd.DataFrame) -> None:
    missing_values = int(data.isna().sum().sum())

    print("Data analysis")
    print(f"  Rows: {len(data)}")
    print(f"  Companies: {data['company_id'].nunique()}")
    print(f"  Missing values: {missing_values}")
    print(f"  Duplicate rows: {data.duplicated().sum()}")
    print(
        "  Risk score: "
        f"mean={data[TARGET_COLUMN].mean():.2f}, "
        f"std={data[TARGET_COLUMN].std():.2f}, "
        f"range={data[TARGET_COLUMN].min():.2f}-"
        f"{data[TARGET_COLUMN].max():.2f}"
    )


def split_data(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    features = data.drop(columns=[TARGET_COLUMN, *EXCLUDED_COLUMNS])
    target = data[TARGET_COLUMN]
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    train_indices, test_indices = next(
        splitter.split(features, target, groups=data["company_id"])
    )

    return (
        features.iloc[train_indices],
        features.iloc[test_indices],
        target.iloc[train_indices],
        target.iloc[test_indices],
    )


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )


def print_model_analysis(
    model: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    predictions = model.predict(x_test)
    residuals = y_test.to_numpy() - predictions
    regressor = model.named_steps["regressor"]
    feature_effects = pd.Series(
        regressor.coef_, index=x_train.columns, name="standardized_effect"
    ).sort_values(key=abs, ascending=False)

    print("\nModel analysis")
    print(f"  Training rows: {len(x_train)}")
    print(f"  Test rows: {len(x_test)}")
    print(f"  R-squared: {r2_score(y_test, predictions):.3f}")
    print(f"  MAE: {mean_absolute_error(y_test, predictions):.3f}")
    print(f"  RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.3f}")
    print(f"  Mean residual: {residuals.mean():.3f}")

    print("\nStrongest standardized feature effects")
    for feature, effect in feature_effects.head(10).items():
        direction = "increases" if effect >= 0 else "decreases"
        print(f"  {feature}: {effect:+.3f} ({direction} predicted risk)")


def main() -> None:
    data = load_data()
    print_data_analysis(data)

    x_train, x_test, y_train, y_test = split_data(data)
    model = build_model()
    model.fit(x_train, y_train)
    print_model_analysis(model, x_train, x_test, y_test)


if __name__ == "__main__":
    main()

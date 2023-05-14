"""Script to read the file and transform the columns accordingly"""

import pandas as pd

from pathlib import Path
from typing import List, Tuple


def read_csv_data(
    filepath_name: str, drop_null: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    In this function we read the csv file and we also define variables as
    either categorical or numeric. We also label encode all variables to be
    in the range of the different categories available for each entry and
    return a separate dataframe containing the encoded variables.
    """

    numerical_columns = ["user_age"]
    drop_columns = ["local_user_id", "subregion_id"]
    date_columns = ["transaction_date"]

    filepath = Path(filepath_name)
    if filepath.is_file():
        labelled_df = pd.DataFrame()
        data = pd.read_csv(
            filepath,
            usecols=lambda x: x not in drop_columns,
            parse_dates=date_columns,
        )
        if drop_null:
            data.dropna(
                subset=[
                    "user_id",
                    "region",
                    "product_id",
                ],
                inplace=True,
            )

        data = add_season_from_date(data)

        categorical_columns = [
            col
            for col in data.columns
            if col not in [*numerical_columns, *date_columns]
        ]
        data[categorical_columns] = data[categorical_columns].astype(
            "category"
        )

        for cat in categorical_columns:
            labelled_df[cat] = data[cat].cat.codes
        labelled_df[numerical_columns] = data[numerical_columns]
        labelled_df[date_columns] = data[date_columns]

        return data, labelled_df
    raise ValueError(f"File {filepath_name} does not exist")


def add_season_from_date(data: pd.DataFrame) -> pd.DataFrame:
    """Add a season column to the data"""
    data["season"] = data["transaction_date"].dt.month % 12 // 3
    return data


def get_monthly_timeseries(
    data: pd.DataFrame,
) -> Tuple[dict, dict]:
    """
    Get monthly timeseries by product_id and region
    """
    try:
        data["product_id"] = (
            data["product_id"].cat.remove_categories(6.0).dropna()
        )
    except ValueError:
        pass

    product_timeseries = get_timeseries(data, "product_id")
    region_timeseries = get_timeseries(data, "region")
    return product_timeseries, region_timeseries


def get_daily_timeseries(
    data: pd.DataFrame,
) -> Tuple[dict, dict]:
    """
    Get daily timeseries by product_id and region
    """
    try:
        data["product_id"] = (
            data["product_id"].cat.remove_categories(6.0).dropna()
        )
    except ValueError:
        pass

    product_timeseries = get_timeseries(data, "product_id", "D")
    region_timeseries = get_timeseries(data, "region", "D")
    return product_timeseries, region_timeseries


def get_timeseries(data: pd.DataFrame, column: str, period: str = "M") -> dict:
    """
    Get the product count time series
    """
    grouped = (
        data.groupby([data["transaction_date"].dt.to_period(period), column])
        .size()
        .reset_index()
    )
    grouped.columns = [*grouped.columns[:-1], "count"]

    time_series = {}

    for _, row in grouped.iterrows():
        date = row["transaction_date"].to_timestamp()
        if date not in time_series:
            time_series[date] = {}
        if row[column] not in time_series[date]:
            time_series[date][row[column]] = row["count"]

    return time_series

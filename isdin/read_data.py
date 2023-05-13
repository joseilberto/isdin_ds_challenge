"""Script to read the file and transform the columns accordingly"""

import pandas as pd

from pathlib import Path
from typing import List, Tuple


def read_csv_data(filepath_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

        categorical_columns = [
            col
            for col in data.columns
            if col not in numerical_columns or col not in date_columns
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

import sys

from src.exception.exception import CustomException
from src.logger.logger import logger

logger.info("starting preprocessing...")
def drop_columns(df):
    columns_to_drop = [
        "customer_id",
        "favorite_color",
        "lucky_number"
    ]
    return df.drop(columns=columns_to_drop)


def remove_duplicates(df):
    return df.drop_duplicates()


def handle_missing_values(df):
    return df


def clean_data(df):
    df = drop_columns(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    logger.info("done with preprocessing, no errors!!!")

    return df

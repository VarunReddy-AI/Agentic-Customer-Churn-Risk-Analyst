import sys
import pandas as pd

from src.logger.logger import logger
from src.exception.exception import CustomException


def load_data(file_path):

    try:

        logger.info(f"Loading data from {file_path}")

        df = pd.read_csv(file_path)
        # comment

        logger.info("Data loaded successfully.")

        return df

    except Exception as e:

        logger.error("Failed to load data.")

        raise CustomException(
            e,
            sys
        )
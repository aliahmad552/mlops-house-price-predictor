import os
import sys
import yaml
import pandas as pd
from dataclasses import dataclass

from src.utils.logger import logger
from src.utils.exceptions import CustomException
from src.utils.file_ops import read_sql_data


# Read YAML configuration
def read_config():
    try:
        config_path = os.path.join(os.getcwd(),"src", "config", "config.yaml")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise CustomException(f"Failed to load config.yaml: {e}", sys)


@dataclass
class DataIngestionConfig:
    mysql_table: str
    raw_path: str


class DataIngestion:
    def __init__(self):
        cfg = read_config()

        self.config = DataIngestionConfig(
            mysql_table=cfg["data"]["mysql_table"],
            raw_path=cfg["data"]["raw_path"]
        )

        logger.info(f"DataIngestion initialized: table={self.config.mysql_table}, "
                    f"raw_path={self.config.raw_path}")

    def fetch_data(self) -> pd.DataFrame:
        try:
            logger.info(f"Reading data from MySQL table: {self.config.mysql_table}")

            df = read_sql_data("house_prices")  # Your custom MySQL function
            logger.info(f"MySQL data fetched successfully. Shape: {df.shape}")

            if df.empty:
                raise CustomException("Fetched MySQL table is empty!", sys)

            return df

        except Exception as e:
            raise CustomException(f"MySQL fetch failed: {e}", sys)

    def save_raw_data(self, df: pd.DataFrame):
        try:
            os.makedirs(os.path.dirname(self.config.raw_path), exist_ok=True)
            df.to_csv(self.config.raw_path, index=False)
            logger.info(f"Raw dataset saved at: {self.config.raw_path}")
        except Exception as e:
            raise CustomException(f"Failed to save raw CSV: {e}", sys)

    def run(self):
        try:
            logger.info("===== DATA INGESTION STARTED =====")

            df = self.fetch_data()
            self.save_raw_data(df)

            logger.info("===== DATA INGESTION COMPLETED SUCCESSFULLY =====")
            return self.config.raw_path

        except Exception as e:
            raise CustomException(f"Data Ingestion Failed: {e}", sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run()

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
        config_path = os.path.join(os.getcwd(), "src", "config", "config.yaml")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise CustomException(f"Failed to load config.yaml: {e}", sys)


@dataclass
class DataIngestionConfig:
    mysql_table: str
    raw_path: str
    sale_path: str
    rent_path: str


class DataIngestion:
    def __init__(self):
        cfg = read_config()

        self.config = DataIngestionConfig(
            mysql_table=cfg["data"]["mysql_table"],
            raw_path=cfg["data"]["raw_path"],
            sale_path=cfg["data"].get("sale_path", "artifacts/raw_data/sale.csv"),
            rent_path=cfg["data"].get("rent_path", "artifacts/raw_data/rent.csv"),
        )

        logger.info(f"DataIngestion initialized: table={self.config.mysql_table}, "
                    f"raw_path={self.config.raw_path}, "
                    f"sale_path={self.config.sale_path}, "
                    f"rent_path={self.config.rent_path}")

    def fetch_data(self) -> pd.DataFrame:
        try:
            logger.info(f"Reading data from MySQL table: {self.config.mysql_table}")

            df = read_sql_data(self.config.mysql_table)
            logger.info(f"MySQL data fetched successfully. Shape: {df.shape}")

            if df.empty:
                raise CustomException("Fetched MySQL table is empty!", sys)

            return df

        except Exception as e:
            raise CustomException(f"MySQL fetch failed: {e}", sys)

    def save_raw_data(self, df: pd.DataFrame):
        try:
            # Save full dataset
            os.makedirs(os.path.dirname(self.config.raw_path), exist_ok=True)
            df.to_csv(self.config.raw_path, index=False)
            logger.info(f"Raw dataset saved at: {self.config.raw_path}")

            # Split Sale and Rent
            df["purpose"] = df["purpose"].str.strip().str.lower()
            df_sale = df[df["purpose"] == "for sale"]
            df_rent = df[df["purpose"] == "for rent"]

            os.makedirs(os.path.dirname(self.config.sale_path), exist_ok=True)
            df_sale.to_csv(self.config.sale_path, index=False)
            logger.info(f"Sale dataset saved at: {self.config.sale_path}")

            os.makedirs(os.path.dirname(self.config.rent_path), exist_ok=True)
            df_rent.to_csv(self.config.rent_path, index=False)
            logger.info(f"Rent dataset saved at: {self.config.rent_path}")

        except Exception as e:
            raise CustomException(f"Failed to save raw CSVs: {e}", sys)

    def run(self):
        try:
            logger.info("===== DATA INGESTION STARTED =====")

            df = self.fetch_data()
            self.save_raw_data(df)

            logger.info("===== DATA INGESTION COMPLETED SUCCESSFULLY =====")
            return self.config.raw_path, self.config.sale_path, self.config.rent_path

        except Exception as e:
            raise CustomException(f"Data Ingestion Failed: {e}", sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run()

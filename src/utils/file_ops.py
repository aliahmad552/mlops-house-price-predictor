import os
import sys
import yaml
import pickle
import pandas as pd
import pymysql
from typing import Any
from dotenv import load_dotenv

from src.utils.exceptions import CustomException
from src.utils.logger import logger

# Load environment variables from .env
load_dotenv()


def read_yaml(path: str):
    """Reads and returns YAML content as a dictionary."""
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise CustomException(f"Failed to read YAML file: {e}", sys)


# Fetch credentials from .env
MYSQL_HOST = os.getenv("host")
MYSQL_USER = os.getenv("user")
MYSQL_PASSWORD = os.getenv("password")
MYSQL_DB = os.getenv("db")


def validate_mysql_credentials():
    """Validates that all MySQL environment variables exist."""
    missing = []

    if not MYSQL_HOST:
        missing.append("host")
    if not MYSQL_USER:
        missing.append("user")
    if MYSQL_PASSWORD is None:
        missing.append("password")
    if not MYSQL_DB:
        missing.append("db")

    if missing:
        raise CustomException(
            f"Missing MySQL credentials in .env file: {', '.join(missing)}",
            sys
        )


def read_sql_data(table_name: str):
    """
    Reads a SQL table from MySQL and returns a pandas DataFrame.
    Table name must come from config.yaml.
    """

    logger.info(f"Attempting to connect to MySQL...")

    validate_mysql_credentials()

    try:
        connection = pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB
        )

        logger.info(f"MySQL connection successful: {MYSQL_HOST} / {MYSQL_DB}")

        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, connection)

        logger.info(f"Fetched {df.shape[0]} rows and {df.shape[1]} columns from '{table_name}'")

        return df

    except Exception as e:
        raise CustomException(f"Error reading SQL table '{table_name}': {e}", sys)


def save_object(obj: Any, path: str):
    """Saves any Python object using pickle."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved at: {path}")
    except Exception as e:
        raise CustomException(f"Failed to save object: {e}", sys)


def load_object(path: str):
    """Loads a pickled Python object."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(f"Failed to load object: {e}", sys)

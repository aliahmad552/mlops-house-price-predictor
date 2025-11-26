# create_src_scaffold.py
# Run this script from your project root to create the `src/` package scaffold
# It will create directories and placeholder files for your modular MLOps project.

import os
from pathlib import Path

STRUCTURE = {
    "src": [
        "__init__.py",
        {
            "config": ["__init__.py", "config.yaml"],
        },
        {
            "utils": [
                "__init__.py",
                "logger.py",
                "exceptions.py",
                "file_ops.py",
            ],
        },
        {
            "data": [
                "__init__.py",
                "data_ingestion.py",
                "data_preprocessing.py",
            ],
        },
        {
            "models": [
                "__init__.py",
                "train_sale_model.py",
                "train_rent_model.py",
                "predictor.py",
            ],
        },
        {
            "pipelines": ["__init__.py", "dvc_pipeline.py"],
        },
        {
            "api": ["__init__.py", "schemas.py"],
        },
    ]
}

PROJECT_ROOT = Path.cwd()

TEMPLATE_HEADER = """# Auto-generated placeholder
# Replace or extend this file with your project-specific logic.
"""

TEMPLATES = {
    "__init__.py": "# Package initializer\n",

    "logger.py": TEMPLATE_HEADER + "import logging, sys\n\nlogging.basicConfig(\n    level=logging.INFO,\n    format=\"%(asctime)s — %(levelname)s — %(message)s\",\n    handlers=[logging.StreamHandler(sys.stdout)]\n)\nlogger = logging.getLogger(\"house_price_mlop\")\n",

    "exceptions.py": TEMPLATE_HEADER + "class CustomException(Exception):\n    \"\"\"Simple custom exception for the project\n    Add additional metadata if needed.\n    \"\"\"\n    def __init__(self, message, errors=None):\n        super().__init__(message)\n        self.errors = errors\n",

    "file_ops.py": TEMPLATE_HEADER + "import yaml, os, pickle\nfrom typing import Any\n\ndef read_yaml(path: str):\n    with open(path, 'r') as f:\n        return yaml.safe_load(f)\n\ndef save_object(obj: Any, path: str):\n    os.makedirs(os.path.dirname(path), exist_ok=True)\n    with open(path, 'wb') as f:\n        pickle.dump(obj, f)\n\ndef load_object(path: str):\n    with open(path, 'rb') as f:\n        return pickle.load(f)\n",

    "data_ingestion.py": TEMPLATE_HEADER + "import pandas as pd\nfrom sqlalchemy import create_engine\nfrom sklearn.model_selection import train_test_split\nfrom src.utils.logger import logger\nfrom src.utils.exceptions import CustomException\n\nclass DataIngestion:\n    def __init__(self, config):\n        self.config = config\n\n    def run(self):\n        \"\"\"Placeholder: implement reading from MySQL and writing train/test CSVs.\n        Use SQLAlchemy + pymysql.\n        \"\"\"\n        raise NotImplementedError\n",

    "data_preprocessing.py": TEMPLATE_HEADER + "import pandas as pd\nfrom src.utils.logger import logger\nfrom src.utils.exceptions import CustomException\n\nclass DataPreprocessor:\n    def __init__(self, config):\n        self.config = config\n\n    def run(self, input_path, output_sale_path, output_rent_path):\n        \"\"\"Placeholder: load CSV, clean, split by purpose (Sale vs Rent), save processed CSVs.\n        \"\"\"\n        raise NotImplementedError\n",

    "train_sale_model.py": TEMPLATE_HEADER + "# Placeholder: training script for Sale model\nfrom src.utils.logger import logger\n\nif __name__ == '__main__':\n    logger.info('Train sale model - implement training logic')\n",

    "train_rent_model.py": TEMPLATE_HEADER + "# Placeholder: training script for Rent model\nfrom src.utils.logger import logger\n\nif __name__ == '__main__':\n    logger.info('Train rent model - implement training logic')\n",

    "predictor.py": TEMPLATE_HEADER + "# Placeholder: shared prediction utilities (load model, preprocess input)\nfrom src.utils.logger import logger\n\nclass Predictor:\n    def __init__(self, model):\n        self.model = model\n\n    def predict(self, X):\n        return self.model.predict(X)\n",

    "dvc_pipeline.py": TEMPLATE_HEADER + "# Placeholder: helper utilities to run or orchestrate DVC stages programmatically if needed.\n\nif __name__ == '__main__':\n    print('Run dvc repro from repo root: dvc repro')\n",

    "schemas.py": TEMPLATE_HEADER + "from pydantic import BaseModel\n\nclass HouseFeatures(BaseModel):\n    Area_in_Marla: float\n    baths: int\n    bedrooms: int\n    city: str\n    location: str\n    property_type: str\n    purpose: str\n",

    "config.yaml": "# Add your configuration here (paths, DB creds via env vars, features)\n# Example:\n# data_source:\n#   mysql:\n#     host: ${MYSQL_HOST}\n#     port: ${MYSQL_PORT}\n#     user: ${MYSQL_USER}\n#     password: ${MYSQL_PASSWORD}\n#     database: ${MYSQL_DATABASE}\n\n"
}


def create_path(path: Path):
    os.makedirs(path, exist_ok=True)


def write_file(path: Path, content: str):
    if path.exists():
        return
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def build_structure(base: Path, structure: dict):
    for key, items in structure.items():
        base_dir = base / key
        create_path(base_dir)
        for item in items:
            if isinstance(item, str):
                write_file(base_dir / item, TEMPLATES.get(item, TEMPLATE_HEADER))
            elif isinstance(item, dict):
                for subdir, files in item.items():
                    sub_path = base_dir / subdir
                    create_path(sub_path)
                    for f in files:
                        write_file(sub_path / f, TEMPLATES.get(f, TEMPLATE_HEADER))


if __name__ == '__main__':
    print(f"Creating scaffold in: {PROJECT_ROOT / 'src'}")
    build_structure(PROJECT_ROOT, STRUCTURE)
    print('Scaffold created. Next: open created files and implement logic for each placeholder.')

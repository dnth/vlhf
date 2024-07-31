from __future__ import annotations

from vl_research.sdk.dataset_api import DatasetSession
from loguru import logger

class VisualLayer:
    def __init__(self, user_id: str, env: str) -> None:
        self.user_id = user_id
        self.env = env

    def create_dataset(self, dataset_name: str, dataset_tar_path: str):
        try:
            logger.info(f"Creating dataset {dataset_name}") 
            session = DatasetSession(self.user_id, self.env)
            session.create_dataset_archive(dataset_name, dataset_tar_path)
        except Exception as e:
            logger.error(f"Error creating dataset {dataset_name}: {e}")
            raise

    def list_datasets(self):
        raise NotImplementedError
    
    def to_hf(self, hf: 'HuggingFace', dataset_name: str = None):
        raise NotImplementedError
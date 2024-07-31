from __future__ import annotations

from vl_research.sdk.dataset_api import DatasetSession


class VisualLayer:
    def __init__(self, user_id: str, env: str) -> None:
        self.user_id = user_id
        self.env = env

    def create_dataset(self, dataset_name: str, dataset_tar_path: str):
        session = DatasetSession(self.user_id, self.env)
        session.create_dataset_archive(dataset_name, dataset_tar_path)

    def list_datasets(self):
        raise NotImplementedError
    
    def to_hf(self, hf: 'HuggingFace', dataset_name: str = None):
        raise NotImplementedError
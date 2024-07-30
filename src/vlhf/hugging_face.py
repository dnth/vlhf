import os
import shutil
import uuid

from datasets import Image, load_dataset
from huggingface_hub import HfApi
from loguru import logger
from tqdm.auto import tqdm

from .visual_layer import VisualLayer


class HuggingFace:
    def __init__(self, token: str = None) -> None:
        self.api = HfApi(token=token)

    def download_dataset(
        self, dataset_id: str, save_path: str = None, **dataset_kwargs
    ):
        self.save_path = save_path or dataset_id
        logger.info(f"Downloading dataset {dataset_id} and saving to {self.save_path}")
        self.dataset = load_dataset(dataset_id, split="all", **dataset_kwargs)

        self.dataset = self.dataset.cast_column("image", Image(decode=False))
        self.dataset = self.dataset.map(self.__add_image_filename, batched=True)
        self.dataset = self.dataset.cast_column("image", Image(decode=True))

        os.makedirs(self.save_path, exist_ok=True)

        for row in tqdm(self.dataset, desc="Saving images"):
            image = row["image"]
            image_filename = row["image_filename"]

            if image_filename is None:
                image_filename = f"{uuid.uuid4()}.jpg"

            full_path = os.path.join(self.save_path, image_filename)
            image.save(full_path)

    def list_datasets(
        self, filter: str = "task_categories:image-classification", search: str = None
    ):
        datasets = self.api.list_datasets(filter=filter, search=search)

        return [d for d in datasets]

    def to_vl(self, vl: VisualLayer, dataset_name: str = None):
        if dataset_name is None:
            dataset_name = self.save_path
            
        self.__make_dataset_tar()
        vl.create_dataset(dataset_name, f"{self.save_path}.tar")

    def __add_image_filename(self, examples):
        image_paths = [img["path"] for img in examples["image"]]
        examples["image_filename"] = image_paths
        return examples

    def __make_dataset_tar(self):
        shutil.make_archive(self.save_path, "tar", self.save_path)

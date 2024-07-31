from __future__ import annotations

import os
import shutil
import uuid

import pandas as pd
from datasets import Image, load_dataset
from huggingface_hub import HfApi
from loguru import logger
from tqdm.auto import tqdm


class HuggingFace:
    def __init__(self, token: str = None) -> None:
        self.api = HfApi(token=token)

    def download_dataset(
        self,
        dataset_id: str,
        save_path: str | None = None,
        image_key: str | None = "image",
        **dataset_kwargs,
    ) -> None:
        def add_image_filename(examples):
            image_paths = [img["path"] for img in examples[image_key]]
            examples["image_filename"] = image_paths
            return examples

        self.save_path = save_path or dataset_id
        logger.info(f"Downloading dataset {dataset_id} and saving to {self.save_path}")
        self.dataset = load_dataset(dataset_id, split="all", **dataset_kwargs)

        self.dataset = self.dataset.cast_column(image_key, Image(decode=False))
        self.dataset = self.dataset.map(add_image_filename, batched=True)
        self.dataset = self.dataset.cast_column(image_key, Image(decode=True))

        os.makedirs(self.save_path, exist_ok=True)

        for row in tqdm(self.dataset, desc="Saving images"):
            image = row[image_key]
            image_filename = row["image_filename"]

            if image_filename is None:
                image_filename = f"{uuid.uuid4()}.jpg"

            full_path = os.path.join(self.save_path, image_filename)
            image.save(full_path)

    def list_datasets(
        self,
        filter: str = "task_categories:image-classification",
        search: str | None = None,
    ) -> pd.DataFrame:
        datasets = self.api.list_datasets(filter=filter, search=search)
        dataset_info = [d for d in datasets]

        df = pd.DataFrame(
            [
                {
                    "id": info.id,
                    "author": info.author,
                    "sha": info.sha,
                    "created_at": info.created_at,
                    "last_modified": info.last_modified,
                    "private": info.private,
                    "gated": info.gated,
                    "disabled": info.disabled,
                    "downloads": info.downloads,
                    "likes": info.likes,
                    "paperswithcode_id": info.paperswithcode_id,
                    "tags": ", ".join(info.tags) if info.tags else None,
                }
                for info in dataset_info
            ]
        )

        return df

    def to_vl(self, vl_session: "VisualLayer", dataset_name: str | None = None) -> None:
        if dataset_name is None:
            dataset_name = self.save_path
            dataset_name = dataset_name.split("/")[-1]

        shutil.make_archive(self.save_path, "tar", self.save_path)
        vl_session.create_dataset(dataset_name, f"{self.save_path}.tar")

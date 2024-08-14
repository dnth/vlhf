from __future__ import annotations

import os
import shutil
import uuid
from typing import TYPE_CHECKING

import pandas as pd
from datasets import Image  # type: ignore
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from huggingface_hub import HfApi  # type: ignore
from loguru import logger
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from vlhf.visual_layer import VisualLayer


def is_one_indexed(dataset: Dataset) -> bool:
    all_categories = []
    for item in dataset:
        all_categories.extend(item["objects"]["category"])

    min_category = min(all_categories)
    max_category = max(all_categories)
    unique_categories = set(all_categories)

    if 0 in unique_categories:
        logger.info("Dataset appears to be 0-indexed")
        return False
    elif min_category == 1 and max_category == len(unique_categories):
        logger.info("Dataset appears to be 1-indexed")
        return True
    else:
        logger.info("Indexing is unclear, further investigation needed")
        return False


def convert_to_vl_object_annotations(dataset: Dataset) -> pd.DataFrame:
    data = []

    for row in dataset:
        image_id = row["image_filename"]
        filename = f"{image_id}"

        if "objects" in row and "bbox" in row["objects"]:
            for bbox, category_name in zip(
                row["objects"]["bbox"], row["objects"]["category_name"]
            ):
                x, y, w, h = bbox

                data.append(
                    {
                        "filename": filename,
                        "col_x": x,
                        "row_y": y,
                        "width": w,
                        "height": h,
                        "label": category_name,
                    }
                )

    return pd.DataFrame(data)


class HuggingFace:
    def __init__(self, token: str) -> None:
        self.api = HfApi(token=token)
        self.token: str = token
        self.dataset: (
            DatasetDict | Dataset | IterableDatasetDict | IterableDataset | None
        ) = None
        self.save_path: str | None = None
        self.image_key: str | None = None
        self.label_key: str | None = None
        self.bbox_key: str | None = None
        self.bbox_label_names: list[str] | None = None

        logger.info("Hugging Face session created")

    def download_dataset(
        self,
        dataset_id: str,
        save_path: str | None = None,
        image_key: str | None = "image",
        label_key: str | None = None,
        bbox_key: str | None = None,
        bbox_label_names: list[str] | None = None,
        num_images: int | None = None,
        **dataset_kwargs,
    ) -> None:
        self.image_key = image_key
        self.label_key = label_key
        self.bbox_key = bbox_key
        self.bbox_label_names = bbox_label_names
        self.save_path = save_path or dataset_id

        def add_image_filename(examples):
            image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")

            image_paths = [img["path"] for img in examples[image_key]]
            examples["image_filename"] = []

            for path in image_paths:
                if path is None or not path.lower().endswith(image_extensions):
                    image_filename = f"{uuid.uuid4()}.jpg"
                    examples["image_filename"].append(image_filename)
                else:
                    examples["image_filename"].append(path)
            return examples

        def add_image_label_name(examples):
            labels = examples[label_key]
            label_names = [
                self.dataset.features[label_key].int2str(label) for label in labels
            ]
            examples["label_name"] = label_names
            return examples

        def add_object_label_name(example, label_list):
            category_indices = example["objects"]["category"]
            category_names = [label_list[idx] for idx in category_indices]
            example["objects"]["category_name"] = category_names
            return example

        logger.info(
            f"Downloading dataset {dataset_id} and saving to local path {self.save_path}"
        )
        self.dataset = load_dataset(dataset_id, split="all", **dataset_kwargs)

        # select only a subset of the dataset if num_images is provided
        if num_images is not None:
            if num_images < 0:
                logger.error("num_images must be a positive integer")
                raise
            elif num_images > len(self.dataset):
                logger.error(
                    f"num_images exceeds the number of images in the dataset - {len(self.dataset)}"
                )
                raise
            else:
                logger.info(f"Selecting first {num_images} images")
                self.dataset = self.dataset.select(range(num_images))

        self.dataset = self.dataset.cast_column(image_key, Image(decode=False))
        logger.info("Adding image filename to dataset")
        self.dataset = self.dataset.map(add_image_filename, batched=True)
        self.dataset = self.dataset.cast_column(image_key, Image(decode=True))

        if self.bbox_label_names:
            logger.info("Adding bbox label name to dataset")

            # check if bbox label is 1-indexed
            if is_one_indexed(self.dataset):
                self.bbox_label_names = ["Unknown"] + self.bbox_label_names

            self.dataset = self.dataset.map(
                lambda x: add_object_label_name(x, self.bbox_label_names)
            )

        # if label_key is provided, add label_name to the dataset feature
        if self.label_key:
            self.dataset = self.dataset.map(add_image_label_name, batched=True)

        self.save_path = os.path.join("saved_images", self.save_path)
        os.makedirs(self.save_path, exist_ok=True)

        for row in tqdm(self.dataset, desc="Saving images"):
            image = row[image_key]

            if image.mode != "RGB":
                image = image.convert("RGB")

            full_path = os.path.join(self.save_path, row["image_filename"])
            image.save(full_path, format="JPEG")

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

    def to_vl(
        self,
        vl_session: "VisualLayer",
        dataset_name: str | None = None,
    ) -> None:
        logger.info("Preparing upload to Visual Layer")

        # if label key is provided, create a parquet file with image_filename and label_name
        if self.label_key:
            if self.dataset is not None:
                self.dataset.select_columns(
                    ["image_filename", "label_name"]
                ).rename_columns(
                    {"image_filename": "filename", "label_name": "label"}
                ).to_pandas().to_parquet(f"{self.save_path}/image_annotations.parquet")

        # if bbox key is provided, create a parquet file with annotations
        if self.bbox_key and self.dataset is not None:
            df = convert_to_vl_object_annotations(self.dataset)
            df.to_parquet(f"{self.save_path}/object_annotations.parquet")

        # if no dataset_name is provided, use the name of the dataset_id
        if dataset_name is None and self.save_path is not None:
            dataset_name = self.save_path.split("/")[-1]
            shutil.make_archive(self.save_path, "tar", self.save_path)
            vl_session.create_dataset(dataset_name, f"{self.save_path}.tar")

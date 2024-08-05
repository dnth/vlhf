from __future__ import annotations

import polars as pl
from datasets import Dataset, Features, Value
from loguru import logger
from vl_research.sdk.dataset_api import DatasetSession


class VisualLayer:
    def __init__(self, user_id: str, env: str) -> None:
        self.user_id = user_id
        self.env = env

        self.session = DatasetSession(self.user_id, self.env)

    def create_dataset(self, dataset_name: str, dataset_tar_path: str):
        try:
            logger.info(f"Creating dataset: {dataset_name}")
            self.session.create_dataset_archive(dataset_name, dataset_tar_path)
            logger.info(f"Dataset {dataset_name} successfully created in Visual Layer!")
        except Exception as e:
            logger.error(f"Error creating dataset {dataset_name}: {e}")
            raise

    def list_datasets(self):
        raise NotImplementedError

    def to_hf(self, hf: "HuggingFace", hf_repo_id: str, vl_dataset_df: pl.DataFrame):
        """
        Pushes a dataset to the Hugging Face repository.
        """
        dataset = Dataset.from_polars(vl_dataset_df)
        features = Features({"image_uri": Value("string"), "label": Value("string")})
        dataset = dataset.cast(features)

        logger.info(f"Pushing dataset to HF repository: {hf_repo_id}")
        dataset.push_to_hub(hf_repo_id, token=hf.token)

    def get_dataset(self, dataset_id: str, pg_uri: str) -> pl.DataFrame:
        logger.info("Reading labels from database")
        labels = pl.read_database_uri(
            f"select * from labels where dataset_id = '{dataset_id}'", pg_uri
        )

        image_labels = labels.filter(
            (pl.col("type") == "IMAGE") & (pl.col("source") != "VL")
        )
        image_labels = image_labels.select("image_id", label="category_display_name")

        logger.info("Reading images from database")
        images = pl.read_database_uri(
            f"select * from images where dataset_id = '{dataset_id}'", pg_uri
        )

        images = images.with_columns(pl.col("metadata").str.json_decode())
        images = images.select(
            id="id",
            image_id=pl.col("original_uri").str.extract("([^/\.]+)\..+$"),
            image_uri="image_uri",
        )

        images = images.join(
            image_labels, left_on="id", right_on="image_id", how="left"
        )

        images = images.select("image_uri", "label")
        return images

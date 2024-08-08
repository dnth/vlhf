from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from datasets import Dataset, Features, Value  # type: ignore
from loguru import logger
from vl_research.sdk.dataset_api import DatasetSession  # type: ignore

if TYPE_CHECKING:
    from vlhf.hugging_face import HuggingFace


class VisualLayer:
    def __init__(self, user_id: str, env: str) -> None:
        self.user_id = user_id
        self.env = env

        self.session = DatasetSession(self.user_id, self.env)
        logger.info("Visual Layer session created")

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

    def to_hf(
        self, hf_session: "HuggingFace", hf_repo_id: str, vl_dataset_df: pl.DataFrame
    ) -> None:
        """
        Pushes a dataset to the Hugging Face repository.
        """

        # pandas has better support in hf datasets currently
        # so we convert the polars dataframe to pandas
        # TODO: revisit this when polars support is better
        pd_dataset = vl_dataset_df.to_pandas()
        dataset = Dataset.from_pandas(pd_dataset)
        features = Features(
            {
                "image_uri": Value("string"),
                "label": Value("string"),
                "issues": [
                    {
                        "confidence": Value("float64"),
                        "description": Value("string"),
                        "duplicate_group_id": Value("string"),
                        "issue_type": Value("string"),
                    }
                ],
            }
        )
        dataset = dataset.cast(features)
        hf_session.dataset = dataset

        logger.info(f"Pushing dataset to HF repository: {hf_repo_id}")
        dataset.push_to_hub(hf_repo_id, token=hf_session.token)

    def get_dataset(self, dataset_id: str, pg_uri: str) -> pl.DataFrame:
        logger.info(f"Fetching dataset: {dataset_id}")
        image_labels = self._get_image_labels(dataset_id, pg_uri)
        object_labels = self._get_object_labels(dataset_id, pg_uri)
        images = self._get_images(dataset_id, pg_uri)
        image_issues = self._get_image_issues(dataset_id, pg_uri)

        vl_dataset = images.join(
            image_labels, left_on="id", right_on="image_id", how="left"
        )

        vl_dataset = vl_dataset.join(
            object_labels, left_on="id", right_on="image_id", how="left"
        )

        vl_dataset = vl_dataset.join(
            image_issues, left_on="id", right_on="image_id", how="left"
        ).select("image_uri", "label", "issues", "object_labels")

        return vl_dataset

    def _get_image_labels(self, dataset_id: str, pg_uri: str) -> pl.DataFrame:
        try:
            logger.info(f"Reading labels from database for dataset: {dataset_id}")
            # TODO: handle case where labels contains OBJECT and IMAGE types
            query = f"SELECT * FROM labels WHERE dataset_id = '{dataset_id}'"
            labels = pl.read_database_uri(query, pg_uri)
            filtered_labels = labels.filter(
                (pl.col("type") == "IMAGE") & (pl.col("source") != "VL")
            ).select("image_id", label="category_display_name")
            logger.info(
                f"Retrieved {len(filtered_labels)} labels for dataset {dataset_id}"
            )
            return filtered_labels
        except Exception as e:
            logger.error(f"Error retrieving labels for dataset {dataset_id}: {str(e)}")
            raise

    def _get_object_labels(self, dataset_id: str, pg_uri: str) -> pl.DataFrame:
        try:
            logger.info(
                f"Reading object labels from database for dataset: {dataset_id}"
            )
            query = f"""
            SELECT 
                image_id, 
                id AS bbox_id,
                category_display_name AS object_label, 
                bounding_box AS bbox
            FROM labels 
            WHERE 
                dataset_id = '{dataset_id}' 
                AND type = 'OBJECT' 
                AND source != 'VL'
            """
            objects = pl.read_database_uri(query, pg_uri)
            objects = objects.select(
                "image_id",
                object_labels=pl.struct(
                    label="object_label", bbox="bbox", bbox_id="bbox_id"
                ),
            )
            objects = objects.group_by("image_id").all()

            logger.info(f"Retrieved {len(objects)} labels for dataset {dataset_id}")
            return objects
        except Exception as e:
            logger.error(f"Error retrieving labels for dataset {dataset_id}: {str(e)}")
            raise

    def _get_images(self, dataset_id: str, pg_uri: str) -> pl.DataFrame:
        try:
            logger.info(f"Reading images from database for dataset: {dataset_id}")
            query = f"SELECT * FROM images WHERE dataset_id = '{dataset_id}'"
            images = pl.read_database_uri(query, pg_uri)
            processed_images = images.with_columns(
                pl.col("metadata").str.json_decode()
            ).select(
                id="id",
                image_id=pl.col("original_uri").str.extract("([^/\.]+)\..+$"),
                image_uri="image_uri",
            )
            logger.info(
                f"Retrieved and processed {len(processed_images)} images for dataset {dataset_id}"
            )
            return processed_images
        except Exception as e:
            logger.error(f"Error retrieving images for dataset {dataset_id}: {str(e)}")
            raise

    def _get_image_issues(self, dataset_id: str, pg_uri: str) -> pl.DataFrame:
        try:
            logger.info(f"Fetching issues for dataset: {dataset_id}")

            issues = pl.read_database_uri(
                f"SELECT * FROM image_issues WHERE dataset_id = '{dataset_id}'", pg_uri
            )

            logger.info(f"Retrieved {len(issues)} issues for dataset {dataset_id}")

            issues_types = pl.read_database_uri("SELECT * FROM issue_type", pg_uri)

            issues = issues.join(issues_types, left_on="type_id", right_on="id")

            issues = issues.filter(pl.col("cause").is_null())

            issues = issues.select(
                "image_id",
                issues=pl.struct(
                    "confidence",
                    "description",
                    duplicate_group_id="issue_subject_id",
                    issue_type="name",
                ),
            )

            issues = issues.group_by("image_id").all()

            return issues

        except Exception as e:
            logger.error(f"Error retrieving issues for dataset {dataset_id}: {str(e)}")
            raise

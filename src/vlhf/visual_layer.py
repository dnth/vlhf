from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from datasets import Dataset, Features, Value, Sequence  # type: ignore
from loguru import logger
from vl_research.sdk.dataset_api import DatasetSession  # type: ignore

if TYPE_CHECKING:
    from vlhf.hugging_face import HuggingFace


class VisualLayer:
    def __init__(self, user_id: str, env: str, pg_uri: str) -> None:
        self.user_id = user_id
        self.env = env
        self.pg_uri = pg_uri
        self.dataset_id: str | None = None
        self.dataset: pl.DataFrame | None = None
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

    def to_hf(self, hf_session: "HuggingFace", hf_repo_id: str) -> None:
        """
        Pushes a dataset to the Hugging Face repository.
        """

        # pandas has better support in hf datasets currently
        # so we convert the polars dataframe to pandas
        # TODO: revisit this when polars support is better

        if self.dataset is not None:
            self.dataset = self.dataset.with_columns(
                [
                    pl.col("image_issues").fill_null([]),
                    pl.col("object_labels").fill_null([]),
                ]
            )

            pd_dataset = self.dataset.to_pandas()

        dataset = Dataset.from_pandas(pd_dataset)
        features = Features(
            {
                "image_uri": Value("string"),
                "image_label": Value("string"),
                "image_issues": [
                    {
                        "confidence": Value("float64"),
                        "description": Value("string"),
                        "duplicate_group_id": Value("string"),
                        "issue_type": Value("string"),
                    }
                ],
                "object_labels": [
                    {
                        "label": Value("string"),
                        "bbox": Sequence(Value("float64")),
                        "bbox_id": Value("string"),
                    }
                ],
            }
        )
        dataset = dataset.cast(features)
        hf_session.dataset = dataset

        logger.info(f"Pushing dataset to HF repository: {hf_repo_id}")
        dataset.push_to_hub(hf_repo_id, token=hf_session.token)

    def get_dataset(self, dataset_id: str) -> pl.DataFrame:
        self.dataset_id = dataset_id

        logger.info(f"Fetching dataset: {dataset_id}")

        images = self._get_images()
        image_labels = self._get_image_labels()
        object_labels = self._get_object_labels()
        image_issues = self._get_image_issues()
        object_issues = self._get_object_issues()

        vl_dataset = images.join(
            image_labels, left_on="id", right_on="image_id", how="left"
        )

        vl_dataset = vl_dataset.join(
            object_labels, left_on="id", right_on="image_id", how="left"
        )

        vl_dataset = vl_dataset.join(
            object_issues, left_on="id", right_on="image_id", how="left"
        )

        vl_dataset = vl_dataset.join(
            image_issues, left_on="id", right_on="image_id", how="left"
        ).select(
            "image_uri", "image_label", "image_issues", "object_labels", "object_issues"
        )

        self.dataset = vl_dataset
        return vl_dataset

    def _get_image_labels(self) -> pl.DataFrame:
        try:
            query = f"""
            SELECT 
                image_id, 
                category_display_name AS image_label
            FROM labels 
            WHERE 
                dataset_id = '{self.dataset_id}' 
                AND type = 'IMAGE' 
            """
            # TODO: add source != 'VL' to the query
            image_labels = pl.read_database_uri(query, self.pg_uri)
            logger.info(f"Retrieved {len(image_labels)} image labels")
            return image_labels
        except Exception as e:
            logger.error(f"Error retrieving image labels: {str(e)}")
            raise

    def _get_object_labels(self) -> pl.DataFrame:
        try:
            query = f"""
            SELECT 
                image_id, 
                id AS bbox_id,
                category_display_name AS object_label, 
                bounding_box AS bbox
            FROM labels 
            WHERE 
                dataset_id = '{self.dataset_id}' 
                AND type = 'OBJECT' 
            """
            # TODO: add source != 'VL' to the query
            objects = pl.read_database_uri(query, self.pg_uri)
            objects = objects.select(
                "image_id",
                object_labels=pl.struct(
                    label="object_label", bbox="bbox", bbox_id="bbox_id"
                ),
            )
            objects = objects.group_by("image_id").all()

            logger.info(f"Retrieved {len(objects)} object labels")
            return objects
        except Exception as e:
            logger.error(f"Error retrieving object labels: {str(e)}")
            raise

    def _get_images(self) -> pl.DataFrame:
        try:
            query = f"""
            SELECT 
                id,
                REGEXP_REPLACE(original_uri, '.+/([^/\.]+)\..+$', '\\1') AS image_id,
                image_uri
            FROM 
                images
            WHERE 
                dataset_id = '{self.dataset_id}'
            """

            images = pl.read_database_uri(query, self.pg_uri)

            logger.info(f"Retrieved and processed {len(images)} images")
            return images
        except Exception as e:
            logger.error(f"Error retrieving images: {str(e)}")
            raise

    def _get_image_issues(self) -> pl.DataFrame:
        try:
            issues = pl.read_database_uri(
                f"SELECT * FROM image_issues WHERE dataset_id = '{self.dataset_id}' AND cause IS NULL",
                self.pg_uri,
            )

            issues_types = pl.read_database_uri("SELECT * FROM issue_type", self.pg_uri)

            issues = issues.join(issues_types, left_on="type_id", right_on="id")
            issues = issues.select(
                "image_id",
                image_issues=pl.struct(
                    "confidence",
                    "description",
                    duplicate_group_id="issue_subject_id",
                    issue_type="name",
                ),
            )
            issues = issues.group_by("image_id").all()
            logger.info(f"Retrieved {len(issues)} images with image-level issues")
            return issues

        except Exception as e:
            logger.error(f"Error retrieving image issues: {str(e)}")
            raise

    def _get_object_issues(self) -> pl.DataFrame:
        try:
            issues = pl.read_database_uri(
                f"SELECT * FROM image_issues WHERE dataset_id = '{self.dataset_id}' AND cause IS NOT NULL",
                self.pg_uri,
            )

            issues_types = pl.read_database_uri("SELECT * FROM issue_type", self.pg_uri)

            issues = issues.join(issues_types, left_on="type_id", right_on="id")
            issues = issues.select(
                "image_id",
                object_issues=pl.struct(
                    "confidence",
                    "description",
                    duplicate_group_id="issue_subject_id",
                    issue_type="name",
                    bbox_id="cause",
                ),
            )
            issues = issues.group_by("image_id").all()
            logger.info(f"Retrieved {len(issues)} images with object-level issues")
            return issues
        except Exception as e:
            logger.error(f"Error retrieving object issues: {str(e)}")
            raise

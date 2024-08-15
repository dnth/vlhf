import pytest
from huggingface_hub import HfApi
from vlhf.hugging_face import HuggingFace, is_one_indexed
import pandas as pd
from datasets import Dataset
import os


@pytest.fixture
def hf_instance():
    return HuggingFace(token="dummy_token")


def test_huggingface_init(hf_instance):
    assert isinstance(hf_instance.api, HfApi)
    assert hf_instance.token == "dummy_token"
    assert hf_instance.dataset is None
    assert hf_instance.save_path is None


def test_object_label_is_one_indexed():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(
        current_dir, "sample_data_zero_indexed_object_annotations.parquet"
    )

    mock_data = pd.read_parquet(parquet_path)
    dataset = Dataset.from_dict(mock_data)
    assert is_one_indexed(dataset) is False

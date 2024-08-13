import pytest
from src.vlhf.hugging_face import HuggingFace
from unittest.mock import patch


class MockHfApi:
    def __init__(self, token):
        self.token = token


@pytest.fixture
def hugging_face_instance():
    with patch("src.vlhf.hugging_face.HfApi", new=MockHfApi):
        return HuggingFace(token="test_token")


def test_hugging_face_initialization(hugging_face_instance):
    assert hugging_face_instance.token == "test_token"
    assert (
        hugging_face_instance.api.token == "test_token"
    )  # Check if api is initialized correctly
    assert hugging_face_instance.dataset is None
    assert hugging_face_instance.save_path is None
    assert hugging_face_instance.image_key is None
    assert hugging_face_instance.label_key is None
    assert hugging_face_instance.bbox_key is None
    assert hugging_face_instance.bbox_label_names is None

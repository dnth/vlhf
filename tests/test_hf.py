import pytest
from vlhf.hugging_face import HuggingFace
from huggingface_hub import HfApi


@pytest.fixture
def token():
    return "some_random_token"


def test_huggingface_initialization(token):
    # Act
    hf = HuggingFace(token)

    # Assert
    assert isinstance(hf.api, HfApi)
    assert hf.token == token
    assert hf.dataset is None
    assert hf.save_path is None
    assert hf.image_key is None
    assert hf.label_key is None
    assert hf.bbox_key is None
    assert hf.bbox_label_names is None

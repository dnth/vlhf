import pytest
from huggingface_hub import HfApi
from vlhf.hugging_face import HuggingFace


@pytest.fixture
def hf_instance():
    return HuggingFace(token="dummy_token")


def test_huggingface_init(hf_instance):
    assert isinstance(hf_instance.api, HfApi)
    assert hf_instance.token == "dummy_token"
    assert hf_instance.dataset is None
    assert hf_instance.save_path is None

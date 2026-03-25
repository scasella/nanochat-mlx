import importlib
import os

import pytest
from huggingface_hub import hf_hub_download

from tests.helpers import run_module


def test_convert_help_works():
    result = run_module("scripts.convert_from_hf", ["--help"])

    assert result.returncode == 0
    assert "Download & convert HuggingFace nanochat model to MLX" in result.stdout


def test_require_torch_gives_actionable_message(monkeypatch):
    module = importlib.import_module("scripts.convert_from_hf")
    real_import_module = importlib.import_module

    def fake_import(name):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import_module(name)

    monkeypatch.setattr(module.importlib, "import_module", fake_import)

    with pytest.raises(module.SetupError, match="uv sync --extra convert"):
        module.require_torch()


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("NANOCHAT_LIVE_TESTS") != "1", reason="live HuggingFace test disabled")
def test_live_huggingface_metadata_resolution():
    module = importlib.import_module("scripts.convert_from_hf")
    model_file, meta_file, step, has_tokenizer = module.resolve_files("nanochat-students/base-d20")

    assert model_file.endswith(".pt")
    assert meta_file.endswith(".json")
    assert step > 0
    assert has_tokenizer is True

    meta_path = hf_hub_download("nanochat-students/base-d20", meta_file)
    with open(meta_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    assert '"model_config"' in text

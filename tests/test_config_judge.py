"""Judge configuration is supplied by the explicitly approved runtime client."""

from pathlib import Path

import pytest

from rlxar.config import load_config
from rlxar.pipeline import run_pipeline


def test_pipeline_requires_explicit_judge_client_before_loading_dataset(tmp_path: Path) -> None:
    config = type(
        "Config",
        (),
        {
            "dataset_path": tmp_path / "missing.jsonl",
            "output_dir": tmp_path / "runs",
            "writer_model_id": "writer",
            "generations": 2,
        },
    )()

    with pytest.raises(ValueError, match="judge_client is required"):
        run_pipeline(config)  # type: ignore[arg-type]


def test_config_rejects_removed_judge_fields(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        'dataset_path = "examples.jsonl"\n'
        'writer_model_id = "writer"\n'
        'judge_model = "provider/model"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"Unknown configuration key\(s\): judge_model"):
        load_config(path)

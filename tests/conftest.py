"""Build the generated media once per test session."""

from pathlib import Path

import pytest

from tests.datasets import DATASETS, generate


@pytest.fixture(scope="session")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("clips")
    generate(root, list(DATASETS))
    return root

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from trajectory_risk.runtime.checkpoints import checkpoints_available  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def pytest_collection_modifyitems(config, items):
    if checkpoints_available():
        return
    skip = pytest.mark.skip(reason="model checkpoints not present in checkpoints/")
    for item in items:
        if "needs_weights" in item.keywords:
            item.add_marker(skip)

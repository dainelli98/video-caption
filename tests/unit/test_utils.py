# -*- coding: utf-8 -*-
"""Example test file (consider removing it)."""
import importlib.metadata as im

from vid_cap import __version__
from vid_cap.utils import say_hello_to


def test_hello_world(mike):
    """Dummy test function."""
    assert say_hello_to(mike) == "hello Mike"


def test_version_compatibility():
    """Test that versions in __init__ and in pyproject.toml are the same."""
    assert im.version("vid_cap") == __version__

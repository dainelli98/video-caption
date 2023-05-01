# -*- coding: utf-8 -*-
# ruff: noqa: D401
"""Entry point."""
import click

from . import __version__
from .scripts import dummy


def _main() -> None:
    """Run main function for entrypoint."""

    @click.group(chain=True)
    @click.version_option(__version__)
    def entry_point() -> None:
        """Package entry point."""

    entry_point.add_command(dummy.main)

    entry_point()


if __name__ == "__main__":
    _main()

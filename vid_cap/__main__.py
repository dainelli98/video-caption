# -*- coding: utf-8 -*-
# ruff: noqa: D401
"""Entry point."""
import click

from . import __version__, scripts


def _main() -> None:
    """Main function for entrypoint."""

    @click.group(chain=True)
    @click.version_option(__version__)
    def entry_point() -> None:
        """Package entry points."""
        for command in (scripts.train, scripts.prepare_dataset, scripts.test):
            entry_point.add_command(command)

    entry_point.main(standalone_mode=False)


if __name__ == "__main__":
    _main()

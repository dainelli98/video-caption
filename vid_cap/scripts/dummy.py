# -*- coding: utf-8 -*-
"""Dummy script."""
import click

from ..utils import say_hello_to


@click.command("dummy")
@click.option("--name", help="dummy name", default="Richard Hendricks")
def main(name: str) -> None:
    """Run dummy script.

    \f

    :param name: Someone's name.
    """
    print(say_hello_to(name))  # noqa: T201


if __name__ == "__main__":
    main.main(standalone_mode=False)

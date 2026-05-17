"""reelgrep CLI entry point."""

from __future__ import annotations

import click

from reelgrep import __version__


@click.group()
@click.version_option(__version__, prog_name="reelgrep")
def main() -> None:
    """Local video search and media analysis."""


if __name__ == "__main__":
    main()

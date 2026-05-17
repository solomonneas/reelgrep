"""reelgrep CLI entry point."""

from __future__ import annotations

import click

from reelgrep import __version__
from reelgrep.commands.contact_sheet import contact_sheet
from reelgrep.commands.export_clip import export_clip
from reelgrep.commands.find_person import find_person
from reelgrep.commands.info import info
from reelgrep.commands.ingest import ingest
from reelgrep.commands.jellyfin import jellyfin
from reelgrep.commands.ls import ls
from reelgrep.commands.make_gif import make_gif
from reelgrep.commands.search_subtitles import search_subtitles
from reelgrep.commands.transcribe import transcribe_cmd as transcribe
from reelgrep.config import set_db_override


@click.group()
@click.version_option(__version__, prog_name="reelgrep")
@click.option("--db", "db_override", type=click.Path(dir_okay=False), default=None,
              help="Override the index database path for this invocation.")
@click.option("--quiet/--verbose", default=None, help="Quiet suppresses non-essential output.")
def main(db_override: str | None, quiet: bool | None) -> None:
    """Local video search and media analysis."""
    if db_override is not None:
        set_db_override(db_override)


for cmd in (ingest, export_clip, make_gif, search_subtitles, contact_sheet,
            find_person, transcribe, info, ls, jellyfin):
    main.add_command(cmd)


if __name__ == "__main__":
    main()

"""Jellyfin backend CLI: resolve item names to local paths for piping."""

from __future__ import annotations

import click

from reelgrep.backends import BackendError
from reelgrep.backends.jellyfin import JellyfinBackend


@click.group("jellyfin")
def jellyfin() -> None:
    """Jellyfin backend commands."""


@jellyfin.command("resolve")
@click.argument("query")
@click.option("--base-url", default=None, help="Override JELLYFIN_URL.")
@click.option("--api-key", default=None, help="Override JELLYFIN_API_KEY.")
@click.option("--no-verify-ssl", is_flag=True, default=False)
@click.option("--timeout", type=float, default=30.0, show_default=True)
def resolve(query: str, base_url: str | None, api_key: str | None,
            no_verify_ssl: bool, timeout: float) -> None:
    """Resolve a Jellyfin item name or 32-hex ItemId to its local file path."""
    backend = JellyfinBackend(
        base_url=base_url,
        api_key=api_key,
        verify_ssl=not no_verify_ssl,
        timeout=timeout,
    )
    try:
        path = backend.resolve(query)
    except BackendError as exc:
        click.echo(f"error: {exc}", err=True)
        raise click.exceptions.Exit(2) from exc
    click.echo(str(path))

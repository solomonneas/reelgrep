"""CLI command: serve the local browser UI for the reelgrep index."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import click

from reelgrep.config import ensure_dirs, get_settings

__all__ = ["serve"]


def _static_dir() -> Path:
    """Return the on-disk path to the bundled web/static directory."""
    return Path(str(files("reelgrep").joinpath("web/static")))


@click.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Loopback only by default. Set to 0.0.0.0 at your own risk.")
@click.option("--port", type=int, default=8765, show_default=True,
              help="TCP port to bind.")
@click.option("--reload", is_flag=True, default=False,
              help="Enable uvicorn autoreload (dev only).")
@click.option("--open-browser/--no-open-browser", "open_browser", default=True,
              show_default=True,
              help="Open the browser UI in your default browser on startup.")
def serve(host: str, port: int, reload: bool, open_browser: bool) -> None:
    """Serve the reelgrep browser UI on a local loopback port."""
    try:
        import uvicorn
        from starlette.routing import Mount
        from starlette.staticfiles import StaticFiles
    except ImportError as exc:
        raise click.ClickException(
            "reelgrep serve requires the [web] extra: pip install reelgrep[web]"
        ) from exc

    from reelgrep.web.app import create_app

    settings = ensure_dirs(get_settings())

    static_dir = _static_dir()
    if not (static_dir / "index.html").exists():
        raise click.ClickException(
            f"frontend assets missing at {static_dir} - reinstall reelgrep"
        )

    app = create_app(db_path=settings.db_path)
    # Mount static assets and SPA fallback onto the API app. The /api/* and
    # /file routes are already registered inside create_app; appending the
    # static mounts last means they only catch unmatched paths.
    app.routes.append(
        Mount("/static", app=StaticFiles(directory=str(static_dir)), name="static")
    )
    app.routes.append(
        Mount("/", app=StaticFiles(directory=str(static_dir), html=True), name="ui")
    )

    url = f"http://{host}:{port}/"
    click.echo(f"reelgrep web UI at {url}")
    click.echo(f"index: {settings.db_path}")
    click.echo("Ctrl+C to stop")

    if open_browser:
        import threading
        import time
        import webbrowser

        def _open() -> None:
            time.sleep(0.8)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host=host, port=port, reload=reload, log_level="info")

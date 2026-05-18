"""`reelgrep faces` CLI group: list, show, label, find, cluster, purge."""
from __future__ import annotations

import click

from reelgrep.faces import Faces, FacesError, cluster_faces


@click.group("faces")
def faces_group() -> None:
    """Browse, label, and query face clusters across the library."""


@faces_group.command("list")
@click.option("--labeled-only", is_flag=True)
@click.option("--limit", type=int, default=None)
@click.pass_context
def faces_list(ctx: click.Context, labeled_only: bool, limit: int | None) -> None:
    """List clusters ranked by size."""
    try:
        faces = Faces()
    except FacesError as exc:
        click.echo(f"error: {exc}", err=True)
        ctx.exit(1)
        return
    rows = faces.list_clusters(labeled_only=labeled_only, limit=limit)
    if not rows:
        click.echo("no clusters yet (run `reelgrep faces cluster`)")
        return
    click.echo(f"{'cluster':<10} {'size':<6} {'label'}")
    for c in rows:
        click.echo(f"#{c.id:<9} {c.size:<6} {c.label or '-'}")


@faces_group.command("show")
@click.argument("cluster_id", type=int)
@click.option("--limit", type=int, default=None)
@click.pass_context
def faces_show(ctx: click.Context, cluster_id: int, limit: int | None) -> None:
    """Show all member detections for a cluster."""
    try:
        faces = Faces()
        cluster = faces.get_cluster(cluster_id)
        members = faces.cluster_members(cluster_id, limit=limit)
    except FacesError as exc:
        click.echo(f"error: {exc}", err=True)
        ctx.exit(1)
        return
    click.echo(f"cluster #{cluster.id} ({cluster.label or '-'}) size={cluster.size}")
    for m in members:
        ts = m.timestamp_ms / 1000.0
        click.echo(f"  {ts:>9.3f}s  {m.video_path}  bbox={m.bbox}")


@faces_group.command("label")
@click.argument("cluster_id", type=int)
@click.argument("label")
@click.pass_context
def faces_label(ctx: click.Context, cluster_id: int, label: str) -> None:
    """Set, update, or clear (pass `-`) a cluster's label."""
    try:
        faces = Faces()
        faces.label_cluster(cluster_id, None if label == "-" else label)
    except FacesError as exc:
        # Collision is exit 2 (consistent with the spec); other failures exit 1.
        code = 2 if "already on cluster" in str(exc) else 1
        click.echo(f"error: {exc}", err=True)
        ctx.exit(code)
        return
    if label == "-":
        click.echo(f"cluster #{cluster_id}: label cleared")
    else:
        click.echo(f"cluster #{cluster_id}: label = {label!r}")


@faces_group.command("find")
@click.argument("query")
@click.option(
    "--out",
    type=click.Path(dir_okay=True, file_okay=False),
    default=None,
    help="Reserved hook; v0.5.0 prints text only.",
)
@click.pass_context
def faces_find(ctx: click.Context, query: str, out: str | None) -> None:
    """Resolve a label or cluster id to all member detections."""
    try:
        faces = Faces()
    except FacesError as exc:
        click.echo(f"error: {exc}", err=True)
        ctx.exit(1)
        return

    # Resolution rule: `label:<x>` forces label; numeric matches cluster id if found;
    # otherwise treat as label.
    if query.startswith("label:"):
        members = faces.find_by_label(query[len("label:"):])
    elif query.isdigit():
        try:
            faces.get_cluster(int(query))
            members = faces.cluster_members(int(query))
        except FacesError:
            members = faces.find_by_label(query)
    else:
        members = faces.find_by_label(query)

    if not members:
        click.echo("no matches")
        return
    click.echo(f"matches: {len(members)}")
    for m in members:
        ts = m.timestamp_ms / 1000.0
        click.echo(f"  {ts:>9.3f}s  {m.video_path}  bbox={m.bbox}")
    if out:
        click.echo("note: --out is a planned export hook; v0.5.0 prints text only.")


@faces_group.command("cluster")
@click.option("--min-size", type=int, default=5)
@click.pass_context
def faces_cluster_cmd(ctx: click.Context, min_size: int) -> None:
    """(Re)compute clusters over the current detection pool."""
    try:
        report = cluster_faces(min_cluster_size=min_size)
    except FacesError as exc:
        click.echo(f"error: {exc}", err=True)
        ctx.exit(1)
        return
    click.echo(f"clusters: {report.clusters_found}")
    click.echo(f"clustered: {report.detections_clustered}")
    click.echo(f"noise:    {report.noise_detections}")
    click.echo(f"labels carried over: {report.labels_carried_over}")
    if report.labels_orphaned:
        click.echo(f"labels orphaned: {', '.join(report.labels_orphaned)}")


@faces_group.command("purge")
@click.argument("target", required=False)
@click.option("--all", "all_", is_flag=True, default=False,
              help="Purge every face detection + cluster in the index.")
@click.confirmation_option(prompt="This permanently deletes face data. Continue?")
@click.pass_context
def faces_purge(ctx: click.Context, target: str | None, all_: bool) -> None:
    """Purge face data for VIDEO_PATH, or pass --all to wipe everything."""
    if not all_ and not target:
        raise click.UsageError("either pass a VIDEO path or --all to wipe everything")
    try:
        faces = Faces()
        if all_:
            faces.purge_all()
            click.echo("deleted all face data")
            return
        n = faces.purge_video(target)
        click.echo(f"deleted {n} detections for {target}")
    except FacesError as exc:
        click.echo(f"error: {exc}", err=True)
        ctx.exit(1)

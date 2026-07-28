"""Model store subcommands: list, search, show, pull, delete."""

import asyncio
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


from olmlx.cli.config_cmd import ensure_config
from olmlx.cli.serve import _load_registry_for_audit

logger = logging.getLogger(__name__)


def _create_store():
    """Create a ModelStore instance for CLI use.

    Raises on failure — callers are responsible for catching and exiting.
    """
    from olmlx.engine.registry import ModelRegistry
    from olmlx.models.store import ModelStore

    ensure_config()
    registry = ModelRegistry()
    registry.load()
    return ModelStore(registry)


def _resolve_and_download(model: str, *, download: bool = True):
    """Resolve *model* to an hf_path and return ``(store, local_dir)``.

    The single guarded resolve+download path shared by the prepare-family
    subcommands (#630). Each used to inline this block with no error
    handling, so a typo'd / offline / gated model name dumped a raw
    ``huggingface_hub`` traceback; here a resolve/download failure prints a
    clean message to stderr and exits 1 instead — matching ``cmd_models_pull``.

    ``download=False`` (used by ``flash info``) returns the local path via
    ``local_path`` without fetching. ``ModelsConfigError`` is re-raised so
    ``cli_main``'s dedicated handler can report it (naming the file) rather
    than collapsing it into a generic exit.
    """
    from olmlx.engine.registry import ModelsConfigError

    try:
        store = _create_store()
        resolved = store.registry.resolve(model)
        hf_path = resolved.hf_path if resolved is not None else model
        if download:
            local_dir = store.ensure_downloaded(hf_path)
        else:
            local_dir = store.local_path(hf_path)
    except ModelsConfigError:
        raise
    except Exception as exc:
        print(f"Error resolving {model!r}: {exc}", file=sys.stderr)
        sys.exit(1)
    return store, local_dir


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.1f} GB"
    elif size_bytes >= 1e6:
        return f"{size_bytes / 1e6:.1f} MB"
    elif size_bytes >= 1e3:
        return f"{size_bytes / 1e3:.1f} KB"
    return f"{size_bytes} B"


def cmd_models_list(_args):
    """List locally downloaded models and configured synthetic panels."""
    try:
        store = _create_store()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        models = store.list_local()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Synthetic panel "models" have no weights, so they aren't in the store —
    # pull them from the registry so they show up here and are selectable.
    registry = _load_registry_for_audit()
    panels = registry.list_panels() if registry is not None else {}

    if not models and not panels:
        print("No models downloaded.")
        return

    if models:
        print(f"{'NAME':<30} {'SIZE':<12} {'PARAMS':<10} {'QUANT':<10} {'HF PATH'}")
        print("-" * 90)
        for m in sorted(models, key=lambda x: x.name or ""):
            name = (m.name or "")[:30]
            params = (m.parameter_size or "")[:10]
            quant = (m.quantization_level or "")[:10]
            print(
                f"{name:<30} {_format_size(m.size):<12} "
                f"{params:<10} {quant:<10} {m.hf_path}"
            )

    if panels:
        if models:
            print()
        print("Panels (synthetic — selectable by name):")
        for name, panel in sorted(panels.items()):
            routes = ",".join(panel.routes)
            print(
                f"  {name:<30} judge={panel.judge}  "
                f"routes={routes}  stop={panel.stop_condition}"
            )


def cmd_models_search(args):
    """Search for models by name."""
    try:
        store = _create_store()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    results = store.registry.search(args.query)
    if not results:
        print(f"No models matching '{args.query}'.")
        return
    print(f"{'NAME':<30} {'HF PATH'}")
    print("-" * 60)
    for name, hf_path in results:
        print(f"{name:<30} {hf_path}")


def cmd_models_show(args):
    """Show details for a specific model."""
    try:
        store = _create_store()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        manifest = store.show(args.model_name)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if manifest is None:
        print(
            f"Model '{args.model_name}' not found locally.\n"
            f"Try: olmlx models search {args.model_name}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Name:           {manifest.name}")
    print(f"HF Path:        {manifest.hf_path}")
    print(f"Size:           {_format_size(manifest.size)}")
    print(f"Format:         {manifest.format}")
    print(f"Family:         {manifest.family}")
    print(f"Parameters:     {manifest.parameter_size}")
    print(f"Quantization:   {manifest.quantization_level}")
    print(f"Modified:       {manifest.modified_at}")
    print(f"Digest:         {manifest.digest}")


def cmd_models_pull(args):
    """Pull/download a model."""
    try:
        store = _create_store()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    async def _pull():
        async for status in store.pull(args.model_name):
            if msg := status.get("status", ""):
                print(msg, flush=True)
        # ``is True`` (not just truthiness): argparse store_true yields a real
        # bool, so this stays correct while ignoring the truthy MagicMock attrs
        # bare-mock CLI tests pass for unrelated pull cases.
        if getattr(args, "with_draft", False) is True:
            await _pull_draft()

    async def _pull_draft():
        """Co-download the curated speculative draft for the target (#514)."""
        from olmlx.engine.registry import lookup_known_draft
        from olmlx.models.store import _strip_ollama_tag

        resolved = store.registry.resolve(args.model_name)
        hf_path = resolved.hf_path if resolved is not None else None
        # The curated map is keyed by the canonical (untagged) hf_path, and a
        # full-path pull like "org/model:q4" resolves with its tag — strip it
        # so the lookup matches, mirroring store.pull().
        if hf_path:
            hf_path = _strip_ollama_tag(hf_path)
        draft = lookup_known_draft(hf_path) if hf_path else None
        if draft is None:
            print(
                f"--with-draft: no known speculative draft for "
                f"{hf_path or args.model_name}; skipping.",
                flush=True,
            )
            return
        print(f"pulling speculative draft {draft.draft_repo}", flush=True)
        async for status in store.pull(draft.draft_repo):
            if msg := status.get("status", ""):
                print(msg, flush=True)
        store.register_speculative_draft(args.model_name, hf_path, draft)
        print(
            f"wired speculative draft {draft.draft_repo} "
            f"(strategy={draft.strategy}) into config",
            flush=True,
        )

    try:
        asyncio.run(_pull())
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        msg = f"Error: {e}"
        if "not found" in str(e).lower():
            msg += f"\nTry: olmlx models search {args.model_name}"
        print(msg, file=sys.stderr, flush=True)
        sys.exit(1)


def cmd_models_delete(args):
    """Delete a locally downloaded model."""
    try:
        store = _create_store()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if not args.yes:
        try:
            confirm = input(f"Delete model '{args.model_name}'? [y/N] ")
        except EOFError:
            print("Aborted.")
            return
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return
    try:
        deleted = store.delete(args.model_name)
    except Exception as e:
        print(f"Error deleting model '{args.model_name}': {e}", file=sys.stderr)
        sys.exit(1)
    if deleted:
        print(f"Model '{args.model_name}' deleted.")
    else:
        print(f"Model '{args.model_name}' not found locally.", file=sys.stderr)
        sys.exit(1)

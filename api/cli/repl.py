"""Interactive CLI for running search queries.

Usage:
    python -m api.cli.repl
"""

import argparse
import sys

from embedding import EmbeddingProviderFactory
from shared.config import load_config

from ..formatters import ResponseFormatter
from ..use_cases import SearchUseCase
from ..validators import RequestValidator, ValidationError


def print_help() -> None:
    print(
        "\nCommands:\n"
        "  :help                 Show this help\n"
        "  :quit / :q / exit      Quit\n"
        "  :show                 Show current settings\n"
        "  :view <type|none>     Set view filter (text/code/image/caption/table/figure)\n"
        "  :lang <name|none>     Set language filter (python/javascript/etc.)\n"
        "  :topk <int>           Set top-k results\n"
        "  :context <on|off>     Toggle parent context\n"
        "  :json <on|off>        Toggle JSON output\n"
        "\nEnter any other text to run a search.\n"
    )


def parse_toggle(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "y", "on")


def show_settings(view, language, top_k, show_context, as_json) -> None:
    print("Current settings:")
    print(f"  view:        {view or '<none>'}")
    print(f"  language:    {language or '<none>'}")
    print(f"  top_k:       {top_k}")
    print(f"  context:     {'on' if show_context else 'off'}")
    print(f"  json:        {'on' if as_json else 'off'}")


def run_repl(args: argparse.Namespace) -> int:
    config = load_config()
    embeddings_client = EmbeddingProviderFactory.create(config)
    use_case = SearchUseCase(embeddings_client, config)

    view = args.view
    language = args.language
    top_k = args.top_k
    show_context = not args.no_context
    as_json = args.json

    print("OCR Vector DB Search REPL")
    print("Type :help for commands.")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        cmd = line.split()
        head = cmd[0].lower()

        if head in (":quit", ":q", "exit"):
            break
        if head == ":help":
            print_help()
            continue
        if head == ":show":
            show_settings(view, language, top_k, show_context, as_json)
            continue
        if head == ":view":
            if len(cmd) < 2:
                print("[error] usage: :view <type|none>")
                continue
            value = cmd[1].lower()
            view = None if value == "none" else value
            print(f"[ok] view set to {view or '<none>'}")
            continue
        if head == ":lang":
            if len(cmd) < 2:
                print("[error] usage: :lang <name|none>")
                continue
            value = cmd[1]
            language = None if value.lower() == "none" else value
            print(f"[ok] language set to {language or '<none>'}")
            continue
        if head == ":topk":
            if len(cmd) < 2 or not cmd[1].isdigit():
                print("[error] usage: :topk <int>")
                continue
            top_k = int(cmd[1])
            print(f"[ok] top_k set to {top_k}")
            continue
        if head == ":context":
            if len(cmd) < 2:
                print("[error] usage: :context <on|off>")
                continue
            show_context = parse_toggle(cmd[1])
            print(f"[ok] context {'on' if show_context else 'off'}")
            continue
        if head == ":json":
            if len(cmd) < 2:
                print("[error] usage: :json <on|off>")
                continue
            as_json = parse_toggle(cmd[1])
            print(f"[ok] json {'on' if as_json else 'off'}")
            continue

        query = line
        try:
            RequestValidator.validate_query(query)
            if view:
                RequestValidator.validate_view(view)
            RequestValidator.validate_top_k(top_k)
        except ValidationError as exc:
            print(ResponseFormatter.format_error(exc))
            continue

        results = use_case.execute(
            query=query,
            view=view,
            language=language,
            top_k=top_k,
            expand_context=show_context,
        )
        if as_json:
            output = ResponseFormatter.format_search_results_json(
                results,
                show_context=show_context,
            )
        else:
            output = ResponseFormatter.format_search_results_text(
                results,
                show_context=show_context,
            )
        print(output)

    return 0


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive search REPL")
    parser.add_argument(
        "--view",
        choices=["text", "code", "image", "caption", "table", "figure"],
        help="Default view filter",
    )
    parser.add_argument(
        "--language",
        help="Default language filter (python/javascript/etc.)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Default number of results",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Disable parent context expansion by default",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON by default",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    sys.exit(run_repl(parser.parse_args()))


__all__ = ["run_repl", "create_parser"]

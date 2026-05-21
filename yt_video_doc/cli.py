from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from yt_video_doc.formatters import render_payload
from yt_video_doc.service import fetch_document, metadata_document, read_document, search_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yt-video",
        description="Read YouTube videos as timestamped transcript documents.",
    )
    parser.add_argument("--cache-dir", default=None, help="Override the transcript cache directory.")
    parser.add_argument("--lang", action="append", dest="languages", help="Preferred transcript language.")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache and fetch the transcript again.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="Fetch and cache a transcript.")
    fetch.add_argument("url")
    fetch.add_argument("--format", choices=["text", "json"], default="text")

    read = subparsers.add_parser("read", help="Read a transcript or time slice.")
    read.add_argument("url")
    read.add_argument("--from", dest="start", default=None)
    read.add_argument("--to", dest="end", default=None)
    read.add_argument("--max-chars", type=int, default=None)
    read.add_argument("--format", choices=["markdown", "text", "json"], default="markdown")

    search = subparsers.add_parser("search", help="Search inside a transcript.")
    search.add_argument("url")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=8)
    search.add_argument("--context-seconds", type=float, default=20)
    search.add_argument("--format", choices=["text", "json"], default="text")

    metadata = subparsers.add_parser("metadata", help="Show video transcript metadata.")
    metadata.add_argument("url")
    metadata.add_argument("--format", choices=["text", "json"], default="text")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    languages = args.languages

    try:
        payload: Any
        output_format = getattr(args, "format", "text")
        if args.command == "fetch":
            payload = fetch_document(args.url, languages=languages, refresh=args.refresh, cache_dir=args.cache_dir)
            if output_format == "text":
                payload = _format_fetch_text(payload)
        elif args.command == "read":
            payload = read_document(
                args.url,
                start=args.start,
                end=args.end,
                max_chars=args.max_chars,
                output_format=args.format,
                languages=languages,
                refresh=args.refresh,
                cache_dir=args.cache_dir,
            )
            output_format = args.format
            if output_format == "json":
                payload = payload["content"]
            else:
                payload = payload["content"]
        elif args.command == "search":
            payload = search_document(
                args.url,
                args.query,
                limit=args.limit,
                context_seconds=args.context_seconds,
                languages=languages,
                refresh=args.refresh,
                cache_dir=args.cache_dir,
            )
            if output_format == "text":
                payload = _format_search_text(payload)
        elif args.command == "metadata":
            payload = metadata_document(args.url, languages=languages, refresh=args.refresh, cache_dir=args.cache_dir)
            if output_format == "text":
                payload = _format_metadata_text(payload)
        else:
            parser.error(f"Unknown command: {args.command}")

        print(render_payload(payload, output_format))
        return 0
    except Exception as exc:
        print(f"yt-video: {exc}", file=sys.stderr)
        return 1


def _format_fetch_text(payload: dict[str, Any]) -> str:
    cache = payload.get("cache", {})
    metadata = payload.get("metadata", {})
    return "\n".join(
        [
            f"Fetched {payload.get('video_id')}",
            f"Title: {metadata.get('title') or ''}".rstrip(),
            f"Source: {metadata.get('source') or ''}".rstrip(),
            f"Segments: {payload.get('segment_count')}",
            f"JSON: {cache.get('json')}",
            f"Markdown: {cache.get('markdown')}",
        ]
    )


def _format_search_text(payload: dict[str, Any]) -> str:
    results = payload.get("results", [])
    if not results:
        return f"No results for: {payload.get('query')}"
    lines = [f"Results for: {payload.get('query')}", ""]
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result['text']}")
        lines.append(result["context"])
        lines.append("")
    return "\n".join(lines).strip()


def _format_metadata_text(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata", {})
    return json.dumps(metadata, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())

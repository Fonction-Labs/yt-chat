from __future__ import annotations

import argparse
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from yt_video_doc.service import fetch_document, metadata_document, read_document, search_document

mcp = FastMCP(
    "yt-video-doc",
    json_response=True,
    instructions=(
        "Use youtube_video_doc to treat a YouTube video as a timestamped text document. "
        "Fetch once, then read the whole transcript, read a time slice, search terms, or inspect metadata."
    ),
)


@mcp.tool()
def youtube_video_doc(
    url: str,
    mode: Literal["fetch", "read", "search", "metadata"] = "read",
    query: str | None = None,
    start: str | None = None,
    end: str | None = None,
    format: Literal["markdown", "text", "json"] = "markdown",
    max_chars: int | None = 12000,
    languages: list[str] | None = None,
    refresh: bool = False,
    limit: int = 8,
    context_seconds: float = 20,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Read, slice, search, or fetch a YouTube transcript document."""
    if mode == "fetch":
        return fetch_document(url, languages=languages, refresh=refresh, cache_dir=cache_dir)
    if mode == "read":
        return read_document(
            url,
            start=start,
            end=end,
            max_chars=max_chars,
            output_format=format,
            languages=languages,
            refresh=refresh,
            cache_dir=cache_dir,
        )
    if mode == "search":
        if not query:
            raise ValueError("query is required when mode is search")
        return search_document(
            url,
            query=query,
            limit=limit,
            context_seconds=context_seconds,
            languages=languages,
            refresh=refresh,
            cache_dir=cache_dir,
        )
    if mode == "metadata":
        return metadata_document(url, languages=languages, refresh=refresh, cache_dir=cache_dir)
    raise ValueError(f"Unsupported mode: {mode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the yt-video-doc MCP server.")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    args = parser.parse_args(argv)
    mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Streaming readers for local files, ZIP files, and recursively nested ZIP files."""

from __future__ import annotations

import csv
import io
import re
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import pandas as pd


TABULAR_EXTENSIONS = (".csv", ".tab", ".tsv", ".txt")


@dataclass(frozen=True)
class VirtualMember:
    source_path: str
    chain: tuple[str, ...]
    size_bytes: int

    @property
    def virtual_path(self) -> str:
        if not self.chain:
            return self.source_path
        return self.source_path + "::" + "::".join(self.chain)


_NESTED_BYTES: dict[tuple[str, tuple[str, ...]], bytes] = {}


def _nested_bytes(source: Path, chain: tuple[str, ...]) -> bytes:
    key = (str(source.resolve()), chain)
    if key in _NESTED_BYTES:
        return _NESTED_BYTES[key]
    if len(chain) == 1:
        with zipfile.ZipFile(source) as archive:
            data = archive.read(chain[0])
    else:
        parent = _nested_bytes(source, chain[:-1])
        with zipfile.ZipFile(io.BytesIO(parent)) as archive:
            data = archive.read(chain[-1])
    _NESTED_BYTES[key] = data
    return data


@lru_cache(maxsize=64)
def list_virtual_members(source_path: str, max_depth: int = 4) -> tuple[VirtualMember, ...]:
    source = Path(source_path)
    if not source.exists():
        return tuple()
    if source.suffix.lower() != ".zip":
        return (VirtualMember(str(source), tuple(), source.stat().st_size),)

    output: list[VirtualMember] = []

    def walk(archive: zipfile.ZipFile, prefix: tuple[str, ...], depth: int) -> None:
        for info in archive.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/") or "/._" in info.filename:
                continue
            chain = prefix + (info.filename,)
            if info.filename.lower().endswith(".zip") and depth < max_depth:
                if len(chain) == 1:
                    data = archive.read(info)
                    _NESTED_BYTES[(str(source.resolve()), chain)] = data
                else:
                    data = archive.read(info)
                    _NESTED_BYTES[(str(source.resolve()), chain)] = data
                try:
                    with zipfile.ZipFile(io.BytesIO(data)) as nested:
                        walk(nested, chain, depth + 1)
                except zipfile.BadZipFile:
                    output.append(VirtualMember(str(source), chain, info.file_size))
            else:
                output.append(VirtualMember(str(source), chain, info.file_size))

    with zipfile.ZipFile(source) as archive:
        walk(archive, tuple(), 0)
    return tuple(output)


@contextmanager
def open_virtual_member(member: VirtualMember):
    source = Path(member.source_path)
    if not member.chain:
        with source.open("rb") as handle:
            yield handle
        return
    if len(member.chain) == 1:
        with zipfile.ZipFile(source) as archive:
            with archive.open(member.chain[0]) as handle:
                yield handle
        return
    parent_bytes = _nested_bytes(source, member.chain[:-1])
    with zipfile.ZipFile(io.BytesIO(parent_bytes)) as archive:
        with archive.open(member.chain[-1]) as handle:
            yield handle


def select_members(source_path: Path, include_regex: str | None, extensions=TABULAR_EXTENSIONS) -> list[VirtualMember]:
    members = list(list_virtual_members(str(source_path)))
    pattern = re.compile(include_regex, re.IGNORECASE) if include_regex else None
    return [
        member for member in members
        if member.virtual_path.lower().endswith(tuple(ext.lower() for ext in extensions))
        and (pattern is None or pattern.search(member.virtual_path))
    ]


def read_header(member: VirtualMember, reader: dict) -> list[str]:
    options = _reader_options(reader)
    with open_virtual_member(member) as handle:
        frame = pd.read_csv(handle, nrows=0, **options)
    return [str(column).strip() for column in frame.columns]


def iter_csv_chunks(
    member: VirtualMember,
    reader: dict,
    usecols: list[str] | None = None,
    chunksize: int = 200_000,
) -> Iterator[pd.DataFrame]:
    options = _reader_options(reader)
    if usecols:
        options["usecols"] = lambda value: str(value).strip() in set(usecols)
    with open_virtual_member(member) as handle:
        for chunk in pd.read_csv(handle, chunksize=chunksize, **options):
            chunk.columns = [str(column).strip() for column in chunk.columns]
            yield chunk


def read_small_text_member(source_path: Path, member_regex: str) -> str | None:
    pattern = re.compile(member_regex, re.IGNORECASE)
    for member in list_virtual_members(str(source_path)):
        if pattern.search(member.virtual_path):
            with open_virtual_member(member) as handle:
                return handle.read().decode("utf-8-sig", errors="replace")
    return None


def _reader_options(reader: dict) -> dict:
    options = {
        "sep": reader.get("sep", ","),
        "skiprows": reader.get("skiprows", 0),
        "encoding": reader.get("encoding", "utf-8-sig"),
        "encoding_errors": "replace",
        "on_bad_lines": "skip",
        "low_memory": False,
    }
    if reader.get("quotechar"):
        options["quotechar"] = reader["quotechar"]
    if reader.get("quoting") == "none":
        # Some exported logger files contain one unmatched quote near EOF.  Treat
        # quotes literally for those explicitly declared sources and keep the
        # audit deterministic; malformed rows are skipped, never imputed.
        options["quoting"] = csv.QUOTE_NONE
    return options

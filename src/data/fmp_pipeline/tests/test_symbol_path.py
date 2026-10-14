"""Tests for the Windows-safe symbol file names of the raw archive (plan WP-5)."""

from __future__ import annotations

import os
import random
import string
import sys
from pathlib import Path
from typing import Optional

import pytest

from src.data.fmp_pipeline.archive.symbol_path import (
    MAX_NAME_LEN,
    decode_symbol,
    encode_symbol,
    endpoint_dir,
    find_case_collisions,
)

RESERVED = ["CON", "PRN", "AUX", "NUL", "COM1", "COM9", "COM0", "LPT1", "LPT9", "CONIN$", "CONOUT$"]
WINDOWS_ILLEGAL = set('<>:"/\\|?*') | {chr(i) for i in range(32)}


@pytest.mark.parametrize("symbol,expected", [
    ("AAPL", "AAPL"),
    ("BRK-B", "BRK-B"),
    ("BRK.B", "BRK.B"),
    ("^GSPC", "^GSPC"),
    ("EURUSD=X", "EURUSD=X"),
    ("BTC/USD", "BTC%2FUSD"),
    ("A:B", "A%3AB"),
    ('A"B', "A%22B"),
    ("A*B?C", "A%2AB%3FC"),
    ("A<B>C|D\\E", "A%3CB%3EC%7CD%5CE"),
    ("100%", "100%25"),
    ("A~B", "A%7EB"),
    ("ABC.", "ABC%2E"),
    ("ABC..", "ABC%2E%2E"),
    ("ABC ", "ABC%20"),
    (" ABC", "%20ABC"),
    ("A B", "A B"),
    (".", "%2E"),
    ("..", "%2E%2E"),
    (".ABC", "%2EABC"),
    ("_ABC", "%5FABC"),
    ("A_B", "A_B"),
    ("A\tB", "A%09B"),
    ("ÄB", "%C3%84B"),
    (None, "_"),
])
def test_encode_known_values(symbol: Optional[str], expected: str):
    assert encode_symbol(symbol) == expected
    assert decode_symbol(expected) == symbol


@pytest.mark.parametrize("name", RESERVED)
def test_reserved_device_names_are_prefixed(name: str):
    assert encode_symbol(name) == "_" + name
    assert encode_symbol(name + ".A") == "_" + name + ".A"
    assert decode_symbol("_" + name) == name
    assert decode_symbol("_" + name + ".A") == name + ".A"


def test_reserved_lookalikes_are_not_prefixed():
    for symbol in ["CONN", "COM10", "LPT", "NULL", "AUXO", "XCON", "COM"]:
        assert encode_symbol(symbol) == symbol


def test_lower_case_gets_a_hash_suffix_and_does_not_collide():
    upper, lower, mixed = encode_symbol("ABC"), encode_symbol("abc"), encode_symbol("Abc")
    assert upper == "ABC"
    assert lower.startswith("abc~") and mixed.startswith("Abc~")
    assert len({upper.casefold(), lower.casefold(), mixed.casefold()}) == 3
    assert decode_symbol(lower) == "abc" and decode_symbol(mixed) == "Abc"


def test_lower_case_reserved_name_is_prefixed_and_suffixed():
    name = encode_symbol("con")
    assert name.startswith("_con~")
    assert decode_symbol(name) == "con"


def test_encoded_names_are_legal_windows_names():
    rng = random.Random(7)
    alphabet = string.printable + "ÄéΩ€.~_ "
    for _ in range(3000):
        symbol = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 8)))
        name = encode_symbol(symbol)
        assert not (set(name) & WINDOWS_ILLEGAL)
        assert name.isascii() and not name.endswith((" ", ".")) and not name.startswith((" ", "."))
        assert name.split(".")[0].upper() not in RESERVED
        assert decode_symbol(name) == symbol


def test_property_no_case_insensitive_collisions():
    rng = random.Random(11)
    symbols = {"".join(rng.choice("aAbB.-_ ") for _ in range(rng.randint(1, 4))) for _ in range(5000)}
    assert find_case_collisions(symbols) == {}


def test_find_case_collisions_reports_groups(monkeypatch: pytest.MonkeyPatch):
    import src.data.fmp_pipeline.archive.symbol_path as sp

    monkeypatch.setattr(sp, "_case_suffix", lambda _: "~00000000")
    collisions = find_case_collisions(["abc", "Abc", "XYZ", "abc"])
    assert collisions == {"abc~00000000": ["Abc", "abc"]}


@pytest.mark.parametrize("name", ["%2", "%zz", "A%2fB", "%41BC", "_ABC", "abc", "abc~00000000", "ABC~1234abcd", ""])
def test_decode_rejects_non_canonical_names(name: str):
    with pytest.raises(ValueError):
        decode_symbol(name)


def test_empty_and_too_long_symbols_are_rejected():
    with pytest.raises(ValueError):
        encode_symbol("")
    with pytest.raises(ValueError):
        encode_symbol("/" * (MAX_NAME_LEN // 3 + 1))


@pytest.mark.parametrize("endpoint", ["splits", "historical-price-eod/non-split-adjusted", "historical-chart/1hour",
                                      "income-statement", "v3.alt/x_y"])
def test_endpoint_dir_accepts_fmp_paths(endpoint: str):
    assert endpoint_dir(endpoint) == endpoint


@pytest.mark.parametrize("endpoint", ["", "/splits", "splits/", "../x", "a/../b", "a/./b", "a\\b", "a:b", "a b",
                                      ".hidden", "C:/x"])
def test_endpoint_dir_rejects_unsafe_paths(endpoint: str):
    with pytest.raises(ValueError):
        endpoint_dir(endpoint)


@pytest.mark.skipif(sys.platform != "win32", reason="checks real NTFS behaviour")
def test_encoded_names_round_trip_through_ntfs(tmp_path: Path):
    symbols = ["CON", "nul", "COM1.A", "ABC.", "abc", "ABC", "A:B", " X ", "_Y", "Ä", "BRK.B", "^GSPC"]
    for symbol in symbols:
        (tmp_path / (encode_symbol(symbol) + ".jsonl.gz")).write_text(symbol, encoding="utf-8")
    names = os.listdir(tmp_path)
    assert len(names) == len(symbols)
    assert sorted(decode_symbol(n.removesuffix(".jsonl.gz")) or "" for n in names) == sorted(symbols)
    for symbol in symbols:
        assert (tmp_path / (encode_symbol(symbol) + ".jsonl.gz")).read_text(encoding="utf-8") == symbol

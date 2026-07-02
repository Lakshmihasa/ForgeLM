"""Tests for schema parsing + serialization (pure Python, no torch)."""

import json

import pytest

from text2sql.data.schema import (
    SchemaStore,
    load_schemas,
    serialize_schema,
)
from text2sql.data.schema import _flatten_primary_keys  # internal helper


# A tiny Spider-style tables.json with two databases:
#  - concert_singer: column_types INCLUDES the leading "*" (offset 0)
#  - small: column_types EXCLUDES "*" (offset 1) — exercises the alignment guard
TABLES = [
    {
        "db_id": "concert_singer",
        "table_names_original": ["stadium", "concert"],
        "table_names": ["big stadium", "the concert"],  # NL names differ on purpose
        "column_names_original": [
            [-1, "*"], [0, "Stadium_ID"], [0, "Name"], [1, "concert_ID"], [1, "Stadium_ID"]
        ],
        "column_names": [[-1, "*"], [0, "id"], [0, "name"], [1, "cid"], [1, "sid"]],
        "column_types": ["text", "number", "text", "number", "number"],
        "primary_keys": [1, 3],
        "foreign_keys": [[4, 1]],  # concert.Stadium_ID -> stadium.Stadium_ID
    },
    {
        "db_id": "small",
        "table_names_original": ["t"],
        "table_names": ["t"],
        "column_names_original": [[-1, "*"], [0, "id"], [0, "name"]],
        "column_names": [[-1, "*"], [0, "id"], [0, "name"]],
        "column_types": ["number", "text"],  # excludes "*" (length n-1)
        "primary_keys": [1],
        "foreign_keys": [],
    },
]


@pytest.fixture
def tables_json(tmp_path):
    p = tmp_path / "tables.json"
    p.write_text(json.dumps(TABLES))
    return str(p)


def test_uses_original_names_not_nl(tables_json):
    s = load_schemas(tables_json)["concert_singer"]
    assert [t.name for t in s.tables] == ["stadium", "concert"]  # not "big stadium"


def test_columns_types_and_primary_keys(tables_json):
    s = load_schemas(tables_json)["concert_singer"]
    stadium = s.table("stadium")
    cols = {c.name: c for c in stadium.columns}
    assert cols["Stadium_ID"].type == "number"
    assert cols["Stadium_ID"].is_primary_key
    assert cols["Name"].type == "text"
    assert not cols["Name"].is_primary_key


def test_foreign_key_direction(tables_json):
    s = load_schemas(tables_json)["concert_singer"]
    assert len(s.foreign_keys) == 1
    fk = s.foreign_keys[0]
    assert (fk.child_table, fk.child_column) == ("concert", "Stadium_ID")
    assert (fk.parent_table, fk.parent_column) == ("stadium", "Stadium_ID")


def test_column_types_offset_when_star_excluded(tables_json):
    # 'small' has column_types of length n-1; types must still align.
    s = load_schemas(tables_json)["small"]
    cols = {c.name: c for c in s.table("t").columns}
    assert cols["id"].type == "number" and cols["id"].is_primary_key
    assert cols["name"].type == "text"


def test_serialize_contents(tables_json):
    out = SchemaStore(tables_json).serialize("concert_singer")
    assert "Table: stadium" in out
    assert "Stadium_ID (number, primary key)" in out
    assert "Foreign keys:" in out
    assert "concert.Stadium_ID -> stadium.Stadium_ID" in out


def test_serialize_is_deterministic(tables_json):
    store = SchemaStore(tables_json)
    assert store.serialize("concert_singer") == store.serialize("concert_singer")


def test_serialize_toggles(tables_json):
    s = load_schemas(tables_json)["concert_singer"]
    no_fk = serialize_schema(s, include_foreign_keys=False)
    no_types = serialize_schema(s, include_types=False)
    no_pk = serialize_schema(s, include_primary_keys=False)
    assert "Foreign keys:" not in no_fk
    assert "(number)" not in no_types and "(text)" not in no_types
    assert "primary key" not in no_pk


def test_store_membership_and_ids(tables_json):
    store = SchemaStore(tables_json)
    assert "concert_singer" in store and "ghost" not in store
    assert store.db_ids() == ["concert_singer", "small"]  # sorted


def test_flatten_primary_keys_handles_composite():
    assert _flatten_primary_keys([1, [2, 3]]) == {1, 2, 3}
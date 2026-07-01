"""Schema parsing and serialization for text-to-SQL.

This module is the single source of truth for how a database schema becomes
text in a prompt. Every model — base zero-shot, base few-shot, fine-tuned —
sees schemas produced here, so the output must be:

  * deterministic  : same db_id -> byte-identical string, every run
  * SQL-accurate   : uses the *original* table/column names (the ones SQL
                     references), never the natural-language aliases
  * configurable   : the include_* / sample_values toggles drive the
                     schema-representation ablation without code changes

Parses the Spider `tables.json` format. Indices in `primary_keys` and
`foreign_keys` refer to positions in `column_names_original`, whose index 0
is always the wildcard column `[-1, "*"]` (skipped when building tables).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Column",
    "ForeignKey",
    "Table",
    "DatabaseSchema",
    "SchemaStore",
    "load_schemas",
    "serialize_schema",
    "fetch_sample_values",
]


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Column:
    """A single column. `name` is the original identifier used in SQL."""

    name: str
    type: str
    is_primary_key: bool = False


@dataclass(frozen=True)
class ForeignKey:
    """A foreign-key relation: child_table.child_column -> parent_table.parent_column."""

    child_table: str
    child_column: str
    parent_table: str
    parent_column: str


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]


@dataclass(frozen=True)
class DatabaseSchema:
    """One database's structure. Table/column order is preserved from the file
    so serialization is deterministic."""

    db_id: str
    tables: tuple[Table, ...]
    foreign_keys: tuple[ForeignKey, ...] = field(default_factory=tuple)

    def table(self, name: str) -> Table | None:
        for t in self.tables:
            if t.name == name:
                return t
        return None


# --------------------------------------------------------------------------- #
# Parsing (Spider tables.json -> DatabaseSchema)
# --------------------------------------------------------------------------- #
def _flatten_primary_keys(primary_keys: list) -> set[int]:
    """Spider uses flat ints; some variants use lists for composite keys."""
    out: set[int] = set()
    for pk in primary_keys:
        if isinstance(pk, (list, tuple)):
            out.update(int(i) for i in pk)
        else:
            out.add(int(pk))
    return out


def _parse_one(entry: dict) -> DatabaseSchema:
    db_id: str = entry["db_id"]
    table_names: list[str] = entry["table_names_original"]
    columns_raw: list[list] = entry["column_names_original"]  # [[table_idx, name], ...]
    column_types: list[str] = entry.get("column_types", [])

    # column_types may or may not include the leading "*" column; align robustly.
    type_offset = len(columns_raw) - len(column_types)

    def type_for(col_idx: int) -> str:
        t_idx = col_idx - type_offset
        if 0 <= t_idx < len(column_types):
            return column_types[t_idx]
        return "text"

    pk_indices = _flatten_primary_keys(entry.get("primary_keys", []))

    # Build columns grouped by their table, skipping the wildcard "*" at index 0.
    grouped: list[list[Column]] = [[] for _ in table_names]
    for col_idx, (table_idx, col_name) in enumerate(columns_raw):
        if table_idx < 0:  # the "*" wildcard column
            continue
        grouped[table_idx].append(
            Column(
                name=col_name,
                type=type_for(col_idx),
                is_primary_key=col_idx in pk_indices,
            )
        )

    tables = tuple(
        Table(name=table_names[i], columns=tuple(cols))
        for i, cols in enumerate(grouped)
    )

    # Foreign keys: each pair is [child_col_idx, parent_col_idx] into columns_raw.
    fks: list[ForeignKey] = []
    for pair in entry.get("foreign_keys", []):
        child_idx, parent_idx = int(pair[0]), int(pair[1])
        if not (_valid(child_idx, columns_raw) and _valid(parent_idx, columns_raw)):
            continue
        c_tbl, c_col = columns_raw[child_idx]
        p_tbl, p_col = columns_raw[parent_idx]
        if c_tbl < 0 or p_tbl < 0:
            continue
        fks.append(
            ForeignKey(
                child_table=table_names[c_tbl],
                child_column=c_col,
                parent_table=table_names[p_tbl],
                parent_column=p_col,
            )
        )

    return DatabaseSchema(db_id=db_id, tables=tables, foreign_keys=tuple(fks))


def _valid(idx: int, columns_raw: list) -> bool:
    return 0 <= idx < len(columns_raw)


def load_schemas(tables_json_path: str | Path) -> dict[str, DatabaseSchema]:
    """Load Spider `tables.json` into {db_id: DatabaseSchema}."""
    path = Path(tables_json_path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):  # tolerate a single-entry file
        data = [data]
    return {e["db_id"]: _parse_one(e) for e in data}


# --------------------------------------------------------------------------- #
# Serialization (DatabaseSchema -> prompt text)
# --------------------------------------------------------------------------- #
def serialize_schema(
    schema: DatabaseSchema,
    *,
    include_types: bool = True,
    include_primary_keys: bool = True,
    include_foreign_keys: bool = True,
    sample_values: dict[tuple[str, str], list] | None = None,
) -> str:
    """Serialize a schema into the prompt body (no `### Database schema` header —
    that is added by prompt.py).

    The keyword flags are the ablation surface. Freeze one configuration for a
    given experiment; changing it changes the contract every model sees.

    `sample_values` optionally maps (table_name, column_name) -> example values,
    typically produced by `fetch_sample_values`.

    Example output:
        Table: stadium
          Stadium_ID (number, primary key)
          Location (text)
          Name (text)
        Table: concert
          concert_ID (number, primary key)
          Stadium_ID (number)
        Foreign keys:
          concert.Stadium_ID -> stadium.Stadium_ID
    """
    lines: list[str] = []

    for table in schema.tables:
        lines.append(f"Table: {table.name}")
        for col in table.columns:
            lines.append(
                "  "
                + _render_column(
                    table.name,
                    col,
                    include_types=include_types,
                    include_primary_keys=include_primary_keys,
                    sample_values=sample_values,
                )
            )

    if include_foreign_keys and schema.foreign_keys:
        lines.append("Foreign keys:")
        for fk in schema.foreign_keys:
            lines.append(
                f"  {fk.child_table}.{fk.child_column} "
                f"-> {fk.parent_table}.{fk.parent_column}"
            )

    return "\n".join(lines)


def _render_column(
    table_name: str,
    col: Column,
    *,
    include_types: bool,
    include_primary_keys: bool,
    sample_values: dict[tuple[str, str], list] | None,
) -> str:
    attrs: list[str] = []
    if include_types:
        attrs.append(col.type)
    if include_primary_keys and col.is_primary_key:
        attrs.append("primary key")

    text = col.name
    if attrs:
        text += f" ({', '.join(attrs)})"

    if sample_values:
        vals = sample_values.get((table_name, col.name))
        if vals:
            rendered = ", ".join(_fmt_value(v) for v in vals)
            text += f": {rendered}"

    return text


def _fmt_value(v: object) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


# --------------------------------------------------------------------------- #
# Optional: sample values from the actual sqlite DB (for the schema ablation)
# --------------------------------------------------------------------------- #
def fetch_sample_values(
    db_path: str | Path,
    schema: DatabaseSchema,
    k: int = 3,
) -> dict[tuple[str, str], list]:
    """Pull up to `k` distinct non-null example values per column from the sqlite
    database. Best-effort: any column/table that errors is skipped, so a broken
    table never breaks serialization.

    NOTE: sample values expose real data to the model. Keep this OFF for your
    primary run and turn it on only as a deliberate ablation, so results stay
    comparable and you don't accidentally leak eval-set contents.
    """
    out: dict[tuple[str, str], list] = {}
    conn = sqlite3.connect(str(db_path))
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        cur = conn.cursor()
        for table in schema.tables:
            for col in table.columns:
                try:
                    cur.execute(
                        f'SELECT DISTINCT "{col.name}" FROM "{table.name}" '
                        f'WHERE "{col.name}" IS NOT NULL LIMIT {int(k)}'
                    )
                    values = [row[0] for row in cur.fetchall()]
                    if values:
                        out[(table.name, col.name)] = values
                except sqlite3.Error:
                    continue
    finally:
        conn.close()
    return out


# --------------------------------------------------------------------------- #
# Ergonomic store
# --------------------------------------------------------------------------- #
class SchemaStore:
    """Loads all schemas once and serializes by db_id. Pass this around instead
    of re-reading tables.json."""

    def __init__(self, tables_json_path: str | Path):
        self._schemas = load_schemas(tables_json_path)

    def __contains__(self, db_id: str) -> bool:
        return db_id in self._schemas

    def __getitem__(self, db_id: str) -> DatabaseSchema:
        return self._schemas[db_id]

    def db_ids(self) -> list[str]:
        return sorted(self._schemas)

    def serialize(self, db_id: str, **kwargs) -> str:
        """Serialize one database. Extra kwargs pass through to serialize_schema."""
        return serialize_schema(self._schemas[db_id], **kwargs)


# --------------------------------------------------------------------------- #
# Manual sanity check: python -m text2sql.data.schema path/to/tables.json [db_id]
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python schema.py <tables.json> [db_id]")
        raise SystemExit(1)

    store = SchemaStore(sys.argv[1])
    target = sys.argv[2] if len(sys.argv) > 2 else store.db_ids()[0]
    print(f"# db_id: {target}\n")
    print(store.serialize(target))
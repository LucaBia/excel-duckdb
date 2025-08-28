# src/db.py
from __future__ import annotations
import duckdb
from pathlib import Path
import pandas as pd


def connect_db(db_path: str) -> duckdb.DuckDBPyConnection:
    """Abre o crea un archivo .duckdb y devuelve la conexión."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(db_path)

def quote_ident(name: str) -> str:
    """Quoted identifier para columnas/tablas seguras."""
    return '"' + name.replace('"', '""') + '"'

def infer_duckdb_type(dtype) -> str:
    """Mapeo simple de dtype pandas → tipo DuckDB."""
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE"
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    # Nota: si luego necesitas DATE, castea explícito en ingest
    return "VARCHAR"

def ensure_table(conn: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    """Crea tabla si no existe con esquema inferido del DataFrame (sin filas)."""
    cols = []
    for c in df.columns:
        typ = infer_duckdb_type(df[c].dtype)
        cols.append(f"{quote_ident(c)} {typ}")
    cols_sql = ", ".join(cols)
    conn.execute(f"CREATE TABLE IF NOT EXISTS {quote_ident(table)} ({cols_sql});")

def add_unique_index(conn: duckdb.DuckDBPyConnection, table: str, keys: list[str]) -> None:
    """Crea índice único con nombre estable (si hay claves)."""
    if not keys:
        return
    idx_name = f"ux_{table}_" + "_".join(keys)
    keys_quoted = ", ".join(quote_ident(k) for k in keys)
    conn.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS {quote_ident(idx_name)} "
        f"ON {quote_ident(table)}({keys_quoted});"
    )
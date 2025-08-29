# src/db.py
from __future__ import annotations
import duckdb
from pathlib import Path
import pandas as pd
import json
from datetime import datetime


def connect_db(db_path: str) -> duckdb.DuckDBPyConnection:
    """Abre o crea un archivo .duckdb, prepara tablas de metadatos y devuelve la conexión."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db_path)

    # Tablas de metadatos (sin constraints para compatibilidad amplia)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_views (
          id BIGINT,
          table_name TEXT NOT NULL,
          view_name TEXT NOT NULL,
          spec_json TEXT NOT NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_charts (
          id BIGINT,
          table_name TEXT NOT NULL,
          chart_name TEXT NOT NULL,
          spec_json TEXT NOT NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_dashboards (
          id BIGINT,
          dashboard_name TEXT NOT NULL,
          description TEXT,
          charts_json TEXT NOT NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    return conn


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
    """Intenta crear índice único con nombre estable (si hay claves). Ignora si no está soportado."""
    if not keys:
        return
    try:
        idx_name = f"ux_{table}_" + "_".join(keys)
        keys_quoted = ", ".join(quote_ident(k) for k in keys)
        conn.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {quote_ident(idx_name)} "
            f"ON {quote_ident(table)}({keys_quoted});"
        )
    except Exception:
        # Algunas versiones no soportan UNIQUE INDEX; continuar sin constraint
        pass


def get_table_columns(conn: duckdb.DuckDBPyConnection, table: str) -> list[dict]:
    """Devuelve columnas y tipos usando PRAGMA table_info."""
    info = conn.execute(f"PRAGMA table_info({quote_ident(table)});").df()
    return [
        {"name": r["name"], "type": r["type"], "pk": bool(r.get("pk", 0))}
        for _, r in info.iterrows()
    ]


def save_view(conn: duckdb.DuckDBPyConnection, table: str, name: str, spec: dict) -> None:
    new_id = next_id(conn, "saved_views")
    conn.execute(
        "INSERT INTO saved_views(id, table_name, view_name, spec_json, created_at) VALUES (?, ?, ?, ?, ?);",
        [new_id, table, name, json.dumps(spec), datetime.now()],
    )


def list_views(conn: duckdb.DuckDBPyConnection, table: str):
    return conn.execute(
        "SELECT id, view_name, spec_json, created_at FROM saved_views WHERE table_name = ? ORDER BY created_at DESC;",
        [table],
    ).df()


def load_view_spec(conn: duckdb.DuckDBPyConnection, view_id: int) -> dict:
    row = conn.execute("SELECT spec_json FROM saved_views WHERE id = ?;", [view_id]).fetchone()
    if not row:
        return {}
    return json.loads(row[0])


def save_chart(conn: duckdb.DuckDBPyConnection, table: str, name: str, spec: dict) -> None:
    new_id = next_id(conn, "saved_charts")
    conn.execute(
        "INSERT INTO saved_charts(id, table_name, chart_name, spec_json, created_at) VALUES (?, ?, ?, ?, ?);",
        [new_id, table, name, json.dumps(spec), datetime.now()],
    )


def list_charts(conn: duckdb.DuckDBPyConnection, table: str):
    return conn.execute(
        "SELECT id, chart_name, spec_json, created_at FROM saved_charts WHERE table_name = ? ORDER BY created_at DESC;",
        [table],
    ).df()


def save_dashboard(conn: duckdb.DuckDBPyConnection, name: str, charts: list[int], description: str = "") -> None:
    new_id = next_id(conn, "saved_dashboards")
    conn.execute(
        "INSERT INTO saved_dashboards(id, dashboard_name, description, charts_json, created_at) VALUES (?, ?, ?, ?, ?);",
        [new_id, name, description, json.dumps([int(i) for i in charts]), datetime.now()],
    )


def list_dashboards(conn: duckdb.DuckDBPyConnection):
    return conn.execute(
        "SELECT id, dashboard_name, description, charts_json, created_at FROM saved_dashboards ORDER BY created_at DESC;"
    ).df()


def load_dashboard_charts(conn: duckdb.DuckDBPyConnection, dashboard_id: int) -> list[dict]:
    row = conn.execute("SELECT charts_json FROM saved_dashboards WHERE id = ?;", [dashboard_id]).fetchone()
    if not row:
        return []
    ids = json.loads(row[0])
    if not ids:
        return []
    placeholders = ",".join(["?"] * len(ids))
    df = conn.execute(
        f"SELECT id, chart_name, spec_json FROM saved_charts WHERE id IN ({placeholders});",
        ids,
    ).df()
    specs = []
    for _, r in df.iterrows():
        specs.append({"id": int(r["id"]), "name": r["chart_name"], "spec": json.loads(r["spec_json"])})
    return specs


def next_id(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    row = conn.execute(f"SELECT COALESCE(MAX(id), 0) + 1 FROM {quote_ident(table)};").fetchone()
    return int(row[0]) if row else 1


def column_exists(conn: duckdb.DuckDBPyConnection, table: str, column: str) -> bool:
    cols = get_table_columns(conn, table)
    return any(c["name"].lower() == column.lower() for c in cols)


def count_failed_casts(conn: duckdb.DuckDBPyConnection, table: str, column: str, new_type: str) -> int:
    qt = quote_ident(table)
    qc = quote_ident(column)
    sql = f"SELECT COUNT(*) FROM {qt} WHERE {qc} IS NOT NULL AND TRY_CAST({qc} AS {new_type}) IS NULL;"
    row = conn.execute(sql).fetchone()
    return int(row[0]) if row else 0


def change_column_type(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    column: str,
    new_type: str,
    strict: bool = False,
) -> dict:
    """
    Cambia el tipo de una columna creando columna temporal + cast y renombrando.
    - strict=False usa TRY_CAST (los valores no casteables quedan NULL)
    - strict=True usa CAST y aborta si hay errores de conversión
    Devuelve métricas: {"failed_conversions": int}
    """
    qt = quote_ident(table)
    qc = quote_ident(column)
    # Nombre temporal seguro
    base_tmp = f"__tmp_{column}"
    tmp = base_tmp
    suffix = 1
    while column_exists(conn, table, tmp):
        tmp = f"{base_tmp}_{suffix}"
        suffix += 1
    qtmp = quote_ident(tmp)

    failed = 0
    conn.execute("BEGIN TRANSACTION;")
    try:
        conn.execute(f"ALTER TABLE {qt} ADD COLUMN {qtmp} {new_type};")
        cast_expr = f"CAST({qc} AS {new_type})" if strict else f"TRY_CAST({qc} AS {new_type})"
        conn.execute(f"UPDATE {qt} SET {qtmp} = {cast_expr};")
        if not strict:
            # Contar fallos (valores que pasan a NULL)
            failed = count_failed_casts(conn, table, column, new_type)
        else:
            # Validar que no haya NULLs nuevos
            check = conn.execute(
                f"SELECT COUNT(*) FROM {qt} WHERE {qc} IS NOT NULL AND {qtmp} IS NULL;"
            ).fetchone()
            if check and int(check[0]) > 0:
                raise ValueError("Errores de conversión al castear en modo estricto")
        conn.execute(f"ALTER TABLE {qt} DROP COLUMN {qc};")
        conn.execute(f"ALTER TABLE {qt} RENAME COLUMN {qtmp} TO {qc};")
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    return {"failed_conversions": failed}

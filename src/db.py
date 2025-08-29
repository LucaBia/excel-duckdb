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
    # Auditoría: eventos y lotes de ingesta
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_events (
          id BIGINT,
          event_type TEXT NOT NULL,           -- 'ingest', 'schema', 'undo_ingest'
          table_name TEXT,
          action TEXT,                        -- p.ej. 'change_type'
          details_json TEXT NOT NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_batches (
          id BIGINT,
          table_name TEXT NOT NULL,
          inserted_count BIGINT DEFAULT 0,
          updated_count BIGINT DEFAULT 0,
          keys_json TEXT,
          updated_cols_json TEXT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    # Asegurar columnas nuevas si la tabla ya existía sin ellas
    try:
        ib_info = conn.execute("PRAGMA table_info(ingest_batches);").df()
        have_cols = set(ib_info["name"].str.lower().tolist())
        if "keys_json" not in have_cols:
            conn.execute("ALTER TABLE ingest_batches ADD COLUMN keys_json TEXT;")
        if "updated_cols_json" not in have_cols:
            conn.execute("ALTER TABLE ingest_batches ADD COLUMN updated_cols_json TEXT;")
    except Exception:
        pass
    # Tabla shadow para versiones previas de filas actualizadas
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS update_versions (
          id BIGINT,
          table_name TEXT NOT NULL,
          batch_id BIGINT NOT NULL,
          keys_json TEXT NOT NULL,
          old_values_json TEXT NOT NULL,
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


def ensure_audit_column(conn: duckdb.DuckDBPyConnection, table: str) -> None:
    """Asegura columna _ingest_batch_id para poder deshacer lotes."""
    if not column_exists(conn, table, "_ingest_batch_id"):
        conn.execute(f"ALTER TABLE {quote_ident(table)} ADD COLUMN {_quote('_ingest_batch_id')} BIGINT;")


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def start_ingest_batch(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    bid = next_id(conn, "ingest_batches")
    conn.execute(
        "INSERT INTO ingest_batches(id, table_name, inserted_count, updated_count, created_at) VALUES (?, ?, 0, 0, CURRENT_TIMESTAMP);",
        [bid, table],
    )
    return bid


def finalize_ingest_batch(conn: duckdb.DuckDBPyConnection, batch_id: int, table: str, inserted: int, updated: int) -> None:
    conn.execute("UPDATE ingest_batches SET inserted_count = ?, updated_count = ? WHERE id = ?;", [int(inserted), int(updated), int(batch_id)])
    # Log evento
    ev_id = next_id(conn, "audit_events")
    details = {"batch_id": batch_id, "inserted": int(inserted), "updated": int(updated)}
    conn.execute(
        "INSERT INTO audit_events(id, event_type, table_name, action, details_json, created_at) VALUES (?, 'ingest', ?, NULL, ?, CURRENT_TIMESTAMP);",
        [ev_id, table, json.dumps(details)],
    )
    
def finalize_ingest_batch_meta(
    conn: duckdb.DuckDBPyConnection,
    batch_id: int,
    table: str,
    keys: list[str] | None,
    updated_cols: list[str] | None,
) -> None:
    # Asegurar columnas meta existen (migración perezosa por si la conexión no pasó por connect_db recientemente)
    try:
        ib_info = conn.execute("PRAGMA table_info(ingest_batches);").df()
        have_cols = set(ib_info["name"].str.lower().tolist())
        if "keys_json" not in have_cols:
            conn.execute("ALTER TABLE ingest_batches ADD COLUMN keys_json TEXT;")
        if "updated_cols_json" not in have_cols:
            conn.execute("ALTER TABLE ingest_batches ADD COLUMN updated_cols_json TEXT;")
    except Exception:
        pass
    conn.execute(
        "UPDATE ingest_batches SET keys_json = ?, updated_cols_json = ? WHERE id = ?;",
        [json.dumps(keys or []), json.dumps(updated_cols or []), int(batch_id)],
    )


def list_ingest_batches(conn: duckdb.DuckDBPyConnection, table: str):
    return conn.execute(
        "SELECT id, inserted_count, updated_count, created_at FROM ingest_batches WHERE table_name = ? ORDER BY id DESC;",
        [table],
    ).df()


def log_update_versions(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    batch_id: int,
    rows: list[dict],
    keys: list[str],
    updated_cols: list[str],
) -> int:
    """Guarda versiones previas (old values) de filas a actualizar.
    rows: lista de dicts con claves y columnas actualizadas (valores antiguos).
    """
    if not rows:
        return 0
    new_ids = []
    for r in rows:
        vid = next_id(conn, "update_versions")
        keys_json = json.dumps({k: r.get(k) for k in keys})
        old_values_json = json.dumps({c: r.get(c) for c in updated_cols})
        conn.execute(
            "INSERT INTO update_versions(id, table_name, batch_id, keys_json, old_values_json, created_at) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP);",
            [vid, table, int(batch_id), keys_json, old_values_json],
        )
        new_ids.append(vid)
    return len(new_ids)


def undo_update_batch(conn: duckdb.DuckDBPyConnection, table: str, batch_id: int) -> int:
    """Revierte actualizaciones aplicando valores antiguos registrados para el lote."""
    # Recuperar metadatos de claves y columnas actualizadas
    meta = conn.execute(
        "SELECT keys_json, updated_cols_json FROM ingest_batches WHERE id = ? AND table_name = ?;",
        [int(batch_id), table],
    ).fetchone()
    if not meta:
        return 0
    keys = json.loads(meta[0] or "[]")
    updated_cols = json.loads(meta[1] or "[]")
    if not keys or not updated_cols:
        return 0
    # Cargar versiones
    vdf = conn.execute(
        "SELECT keys_json, old_values_json FROM update_versions WHERE table_name = ? AND batch_id = ?;",
        [table, int(batch_id)],
    ).df()
    if vdf.empty:
        return 0
    # Construir DataFrame para revertir
    import pandas as _pd
    rows = []
    for _, r in vdf.iterrows():
        kd = json.loads(r["keys_json"]) if r["keys_json"] else {}
        ov = json.loads(r["old_values_json"]) if r["old_values_json"] else {}
        rows.append({**kd, **ov})
    tmpdf = _pd.DataFrame(rows)
    if tmpdf.empty:
        return 0
    conn.register("revert_tmp", tmpdf)
    # UPDATE por join de claves
    set_clause = ", ".join([f"t.{quote_ident(c)} = r.{quote_ident(c)}" for c in updated_cols])
    on_clause = " AND ".join([f"t.{quote_ident(k)} = r.{quote_ident(k)}" for k in keys])
    sql = f"""
        UPDATE {quote_ident(table)} AS t
        SET {set_clause}
        FROM revert_tmp AS r
        WHERE {on_clause};
    """
    conn.execute(sql)
    # Retornar count afectado aproximado
    return len(tmpdf)


def undo_ingest_batch(conn: duckdb.DuckDBPyConnection, table: str, batch_id: int) -> int:
    qt = quote_ident(table)
    # Contar y borrar
    cnt = conn.execute(f"SELECT COUNT(*) FROM {qt} WHERE {_quote('_ingest_batch_id')} = ?;", [int(batch_id)]).fetchone()[0]
    conn.execute(f"DELETE FROM {qt} WHERE {_quote('_ingest_batch_id')} = ?;", [int(batch_id)])
    # Log evento
    ev_id = next_id(conn, "audit_events")
    conn.execute(
        "INSERT INTO audit_events(id, event_type, table_name, action, details_json, created_at) VALUES (?, 'undo_ingest', ?, 'delete_rows', ?, CURRENT_TIMESTAMP);",
        [ev_id, table, json.dumps({"batch_id": int(batch_id), "deleted": int(cnt)})],
    )
    return int(cnt)


def log_schema_change(conn: duckdb.DuckDBPyConnection, table: str, action: str, before: dict, after: dict) -> None:
    ev_id = next_id(conn, "audit_events")
    details = {"action": action, "before": before, "after": after}
    conn.execute(
        "INSERT INTO audit_events(id, event_type, table_name, action, details_json, created_at) VALUES (?, 'schema', ?, ?, ?, CURRENT_TIMESTAMP);",
        [ev_id, table, action, json.dumps(details)],
    )


def list_schema_events(conn: duckdb.DuckDBPyConnection, table: str):
    return conn.execute(
        "SELECT id, action, details_json, created_at FROM audit_events WHERE event_type = 'schema' AND table_name = ? ORDER BY id DESC;",
        [table],
    ).df()


def update_rows_from_df(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    df,
    keys: list[str],
    cols_to_update: list[str] | None = None,
) -> int:
    """Actualiza filas de `table` usando un DataFrame con columnas clave y columnas a actualizar."""
    if df is None or len(df) == 0:
        return 0
    import pandas as _pd
    tmp_name = f"upd_tmp_{int(datetime.now().timestamp()*1000)}"
    conn.register(tmp_name, _pd.DataFrame(df))
    key_join = " AND ".join([f"t.{quote_ident(k)} = s.{quote_ident(k)}" for k in keys])
    cols = cols_to_update or [c for c in df.columns if c not in keys]
    if not cols:
        return 0
    set_clause = ", ".join([f"{quote_ident(c)} = s.{quote_ident(c)}" for c in cols])
    sql = f"""
        UPDATE {quote_ident(table)} AS t
        SET {set_clause}
        FROM {tmp_name} AS s
        WHERE {key_join};
    """
    conn.execute(sql)
    return len(df)


def insert_rows_from_df(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    df,
    batch_id: int | None = None,
) -> int:
    if df is None or len(df) == 0:
        return 0
    import pandas as _pd
    tmp_name = f"ins_tmp_{int(datetime.now().timestamp()*1000)}"
    df2 = _pd.DataFrame(df).copy()
    if batch_id is not None:
        df2["_ingest_batch_id"] = int(batch_id)
    conn.register(tmp_name, df2)
    cols = list(df2.columns)
    cols_sql = ", ".join(quote_ident(c) for c in cols)
    sql = f"INSERT INTO {quote_ident(table)} ({cols_sql}) SELECT {cols_sql} FROM {tmp_name};"
    conn.execute(sql)
    return len(df2)


def delete_rows_by_keys(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    key_df,
    keys: list[str],
) -> int:
    if key_df is None or len(key_df) == 0:
        return 0
    import pandas as _pd
    tmp_name = f"del_keys_{int(datetime.now().timestamp()*1000)}"
    conn.register(tmp_name, _pd.DataFrame(key_df)[keys])
    join_on = " AND ".join([f"t.{quote_ident(k)} = k.{quote_ident(k)}" for k in keys])
    sql = f"DELETE FROM {quote_ident(table)} t WHERE EXISTS (SELECT 1 FROM {tmp_name} k WHERE {join_on});"
    conn.execute(sql)
    return len(key_df)


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
    # tipo anterior
    before_type = None
    try:
        cols = get_table_columns(conn, table)
        for c in cols:
            if c["name"].lower() == column.lower():
                before_type = c["type"]
                break
    except Exception:
        pass
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
        # Log de esquema
        try:
            log_schema_change(
                conn,
                table,
                action="change_type",
                before={"column": column, "type": before_type},
                after={"column": column, "type": new_type},
            )
        except Exception:
            pass
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    return {"failed_conversions": failed}


def schema_add_column(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    column: str,
    col_type: str,
    default: str | None = None,
) -> None:
    qt = quote_ident(table)
    qc = quote_ident(column)
    conn.execute(f"ALTER TABLE {qt} ADD COLUMN {qc} {col_type}{(' DEFAULT ' + default) if default else ''};")
    log_schema_change(
        conn,
        table,
        action="add_column",
        before=None,
        after={"column": column, "type": col_type, "default": default},
    )


def schema_rename_column(conn: duckdb.DuckDBPyConnection, table: str, old: str, new: str) -> None:
    qt = quote_ident(table)
    qo = quote_ident(old)
    qn = quote_ident(new)
    # tipo previo
    cols = get_table_columns(conn, table)
    prev_type = next((c["type"] for c in cols if c["name"].lower() == old.lower()), None)
    conn.execute(f"ALTER TABLE {qt} RENAME COLUMN {qo} TO {qn};")
    log_schema_change(
        conn,
        table,
        action="rename_column",
        before={"column": old, "type": prev_type},
        after={"column": new, "type": prev_type},
    )


def schema_drop_column(conn: duckdb.DuckDBPyConnection, table: str, column: str) -> None:
    qt = quote_ident(table)
    qc = quote_ident(column)
    cols = get_table_columns(conn, table)
    prev_type = next((c["type"] for c in cols if c["name"].lower() == column.lower()), None)
    conn.execute(f"ALTER TABLE {qt} DROP COLUMN {qc};")
    log_schema_change(
        conn,
        table,
        action="drop_column",
        before={"column": column, "type": prev_type},
        after=None,
    )


def revert_schema_event(conn: duckdb.DuckDBPyConnection, event_id: int) -> str:
    """Revierte un evento de esquema registrado. Devuelve mensaje de resultado."""
    row = conn.execute(
        "SELECT table_name, action, details_json FROM audit_events WHERE id = ? AND event_type = 'schema';",
        [int(event_id)],
    ).fetchone()
    if not row:
        return "Evento no encontrado"
    table, action, details_json = row
    details = json.loads(details_json or "{}")
    if action == "change_type":
        before = details.get("before") or {}
        col = before.get("column")
        typ = before.get("type")
        if not col or not typ:
            return "No hay info suficiente para revertir change_type"
        change_column_type(conn, table, col, typ, strict=False)
        return f"Revertido tipo de {col} a {typ}"
    elif action == "rename_column":
        before = details.get("before") or {}
        after = details.get("after") or {}
        old = before.get("column")
        new = after.get("column")
        if not old or not new:
            return "No hay info suficiente para revertir rename"
        schema_rename_column(conn, table, new, old)
        return f"Revertido renombre: {new} → {old}"
    elif action == "add_column":
        after = details.get("after") or {}
        col = after.get("column")
        if not col:
            return "No hay info suficiente para revertir add_column"
        conn.execute(f"ALTER TABLE {quote_ident(table)} DROP COLUMN {quote_ident(col)};")
        return f"Eliminada columna agregada {col}"
    elif action == "drop_column":
        before = details.get("before") or {}
        col = before.get("column")
        typ = before.get("type")
        if not col or not typ:
            return "No hay info suficiente para revertir drop_column"
        # Re-crear columna (sin datos previos)
        conn.execute(f"ALTER TABLE {quote_ident(table)} ADD COLUMN {quote_ident(col)} {typ};")
        return f"Re-creada columna {col} de tipo {typ} (datos no recuperados)"
    return "Acción no soportada"

# src/ingest.py
from __future__ import annotations
import pandas as pd
import duckdb
from typing import Optional, Tuple, Dict
from .db import (
    ensure_table,
    add_unique_index,
    quote_ident,
    ensure_audit_column,
    start_ingest_batch,
    finalize_ingest_batch,
    finalize_ingest_batch_meta,
    log_update_versions,
)

def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def load_excel_to_df(file, start_row: int, sheet_name: Optional[str]) -> pd.DataFrame:
    """
    Lee Excel a DataFrame usando `header=start_row-1`. Si `sheet_name` es None o "",
    y el archivo tiene múltiples hojas, toma la primera hoja disponible.
    Normaliza nombres de columnas.
    `file` puede ser un buffer subido desde Streamlit.
    """
    # Cuando sheet_name es None, pandas puede devolver un dict {nombre_hoja: DataFrame}
    df_or_dict = pd.read_excel(
        file,
        sheet_name=(sheet_name if sheet_name else None),
        header=start_row - 1,
        engine="openpyxl",
    )

    if isinstance(df_or_dict, dict):
        # Tomar la primera hoja si no se especificó una en particular
        if not df_or_dict:
            raise ValueError("El archivo Excel no contiene hojas válidas.")
        first_key = next(iter(df_or_dict.keys()))
        df = df_or_dict[first_key]
    else:
        df = df_or_dict

    # Normalización de nombres de columnas
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
    )
    return df

def smart_casts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intenta castear strings que parezcan fechas y números.
    Mantiene robustez (errores='ignore' para no romper).
    """
    out = df.copy()
    # Heurística simple: si muchos valores parecen fecha -> parsea
    for c in out.columns:
        if out[c].dtype == "object":
            # Intento de fecha
            dt = pd.to_datetime(out[c], errors="ignore", dayfirst=False, infer_datetime_format=True)
            if pd.api.types.is_datetime64_any_dtype(dt):
                out[c] = dt
            else:
                # Intento de numérico
                out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", ".", regex=False), errors="ignore")
    return out

def upsert_df(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table: str,
    unique_keys: Optional[list[str]] = None,
) -> Tuple[int, Dict[str, int]]:
    """
    Inserta df en `table`. Si `unique_keys`, usa MERGE (UPSERT), si no, INSERT directo.
    Devuelve (filas_df, métricas dict).
    """
    if df.empty:
        return 0, {"inserted": 0, "updated": 0}

    # Asegurar tabla y columna de auditoría
    ensure_table(conn, table, df)
    ensure_audit_column(conn, table)
    batch_id = start_ingest_batch(conn, table)

    # Registrar DataFrame como vista temporal
    conn.register("df_tmp_streamlit", df)

    # Claves únicas / índice
    unique_keys = unique_keys or []
    add_unique_index(conn, table, unique_keys)

    cols = list(df.columns)
    cols_quoted = ", ".join(quote_ident(c) for c in cols)

    inserted_count = 0
    updated_count = 0

    if unique_keys:
        # UPSERT en 2 pasos (compatible con DuckDB sin MERGE):
        # 1) UPDATE filas existentes con join por claves
        non_key_cols = [c for c in cols if c not in unique_keys]
        if non_key_cols:
            # Registrar versiones previas antes de actualizar
            key_select = ", ".join([f"t.{quote_ident(k)} AS {quote_ident(k)}" for k in unique_keys])
            old_select = ", ".join([f"t.{quote_ident(c)} AS {quote_ident(c)}" for c in non_key_cols])
            join_on = " AND ".join([f"t.{quote_ident(k)} = s.{quote_ident(k)}" for k in unique_keys])
            pre = conn.execute(
                f"SELECT {key_select}{(', ' + old_select) if old_select else ''} FROM {quote_ident(table)} AS t JOIN df_tmp_streamlit AS s ON {join_on};"
            ).df()
            if not pre.empty:
                finalize_ingest_batch_meta(conn, batch_id, table, unique_keys, non_key_cols)
                log_update_versions(conn, table, batch_id, pre.to_dict(orient="records"), unique_keys, non_key_cols)
            set_clause = ", ".join([f"{quote_ident(c)} = s.{quote_ident(c)}" for c in non_key_cols])
            on_clause = " AND ".join([f"t.{quote_ident(k)} = s.{quote_ident(k)}" for k in unique_keys])
            update_sql = f"""
                UPDATE {quote_ident(table)} AS t
                SET {set_clause}
                FROM df_tmp_streamlit AS s
                WHERE {on_clause};
            """
            conn.execute(update_sql)
            # No cambiamos batch_id para updates; contamos luego
        # 2) INSERT filas nuevas (las que no existen por claves)
        on_exists_clause = " AND ".join([f"t.{quote_ident(k)} = s.{quote_ident(k)}" for k in unique_keys])
        insert_sql = f"""
            INSERT INTO {quote_ident(table)} ({cols_quoted}, {_quote('_ingest_batch_id')})
            SELECT {cols_quoted}, ?
            FROM df_tmp_streamlit AS s
            WHERE NOT EXISTS (
                SELECT 1 FROM {quote_ident(table)} AS t WHERE {on_exists_clause}
            );
        """
        conn.execute(insert_sql, [int(batch_id)])
        # Contar insertados de este lote
        inserted_count = conn.execute(
            f"SELECT COUNT(*) FROM {quote_ident(table)} WHERE {_quote('_ingest_batch_id')} = ?;",
            [int(batch_id)],
        ).fetchone()[0]
        # Estimar actualizados como los del DF que no fueron insertados
        updated_count = max(0, len(df) - int(inserted_count))
        finalize_ingest_batch(conn, batch_id, table, inserted=inserted_count, updated=updated_count)
        return len(df), {"inserted": int(inserted_count), "updated": int(updated_count)}
    else:
        conn.execute(
            f'INSERT INTO {quote_ident(table)} ({cols_quoted}, {_quote("_ingest_batch_id")}) SELECT {cols_quoted}, ? FROM df_tmp_streamlit;',
            [int(batch_id)],
        )
        inserted_count = conn.execute(
            f"SELECT COUNT(*) FROM {quote_ident(table)} WHERE {_quote('_ingest_batch_id')} = ?;",
            [int(batch_id)],
        ).fetchone()[0]
        finalize_ingest_batch(conn, batch_id, table, inserted=int(inserted_count), updated=0)
        return len(df), {"inserted": int(inserted_count)}

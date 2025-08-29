# app.py
from __future__ import annotations
import streamlit as st
import plotly.express as px
import pandas as pd
from src.db import (
    connect_db,
    quote_ident,
    get_table_columns,
    count_failed_casts,
    change_column_type,
    list_ingest_batches,
    undo_ingest_batch,
    undo_update_batch,
    list_schema_events,
    schema_add_column,
    schema_rename_column,
    schema_drop_column,
    revert_schema_event,
    save_view,
    list_views,
    load_view_spec,
    save_chart,
    list_charts,
    save_dashboard,
    list_dashboards,
    load_dashboard_charts,
    ensure_audit_column,
    start_ingest_batch,
    finalize_ingest_batch,
    finalize_ingest_batch_meta,
    log_update_versions,
    update_rows_from_df,
    insert_rows_from_df,
    delete_rows_by_keys,
)
from src.ingest import load_excel_to_df, smart_casts, upsert_df

st.set_page_config(page_title="Excel → DuckDB (Local)", layout="wide")


def _sanitize_for_duckdb(df: pd.DataFrame) -> pd.DataFrame:
    """Reemplaza NaN/NaT/pd.NA por None y ajusta dtypes para registrar en DuckDB."""
    try:
        out = df.copy()
        for c in out.columns:
            out[c] = out[c].where(pd.notna(out[c]), None)
            # Evitar pandas nullable Int64 con None → usa object
            dtype_name = str(out[c].dtype)
            if dtype_name.lower() in ("int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8", "Int64"):
                out[c] = out[c].astype(object)
        return out
    except Exception:
        return df

# --- Sidebar: Instancia ---
st.sidebar.header("Instancia (archivo .duckdb)")
db_path = st.sidebar.text_input("Ruta del archivo .duckdb", value="data/local.duckdb")
if st.sidebar.button("Abrir/Crear instancia"):
    st.session_state["conn"] = connect_db(db_path)
    st.sidebar.success(f"Conectado a {db_path}")

conn = st.session_state.get("conn")
if not conn:
    st.info("👈 Elige una ruta y pulsa **Abrir/Crear instancia** en la barra lateral.")
    st.stop()

st.title("Excel → DuckDB (local)")
st.caption("100% local, sin servidor. Carga, normaliza, inserta, filtra y explora.")

# --- Sección: Ingesta ---
with st.expander("📥 Subir Excel e ingestar", expanded=False):
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        file = st.file_uploader("Archivo Excel", type=["xlsx", "xls"], accept_multiple_files=False)
    with c2:
        start_row = st.number_input("Fila de inicio de datos", min_value=1, value=1, step=1)
    with c3:
        sheet_name = st.text_input("Nombre de hoja (opcional)", value="")

    table_name = st.text_input("Tabla destino (SQL)", value="ventas")

    if file:
        # Previsualización
        preview_df = load_excel_to_df(file, start_row, sheet_name or None).head(20)
        st.write("**Previsualización (primeras 20 filas):**")
        st.dataframe(preview_df, width='stretch')

        # Claves únicas (opcional)
        unique_keys = st.multiselect(
            "Claves únicas para idempotencia (opcional)",
            options=list(preview_df.columns),
            help="Si defines claves, se hará MERGE (UPSERT). Si no, se insertará todo."
        )

        if st.button("Procesar e insertar", type="primary"):
            try:
                df = load_excel_to_df(file, start_row, sheet_name or None)
                df = smart_casts(df)
                n, metrics = upsert_df(conn, df, table_name, unique_keys=unique_keys)
                st.success(f"OK: {n} filas procesadas → {metrics}")
            except Exception as e:
                st.error(f"Error de ingesta: {e}")

# --- Sección: Exploración ---
with st.expander("🔎 Explorar con SQL", expanded=False):
    default_sql = f"SELECT * FROM {quote_ident(table_name)} LIMIT 100;"
    sql = st.text_area("Consulta SQL", value=default_sql, height=140)
    if st.button("Ejecutar consulta"):
        try:
            res = conn.execute(sql).df()
            st.dataframe(res, width='stretch')
            st.session_state["last_query_df"] = res
        except Exception as e:
            st.error(str(e))

# --- Sección: Visualizador de tablas ---
with st.expander("🗂️ Visualizador de tablas (todas las filas)", expanded=False):
    if st.button("Actualizar", key="refresh_browse"):
        st.rerun()
    try:
        tables_df = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name;"
        ).df()
        table_options = tables_df["table_name"].tolist()
    except Exception:
        table_options = []

    if not table_options:
        st.info("No hay tablas todavía. Ingresa datos en la sección de arriba.")
    else:
        sel_table = st.selectbox("Tabla", options=table_options, index=(table_options.index(table_name) if table_name in table_options else 0), key="browse_table")
        # Paginación simple
        total_rows = conn.execute(f"SELECT COUNT(*) FROM {quote_ident(sel_table)};").fetchone()[0]
        c1, c2, c3 = st.columns(3)
        with c1:
            page_size = st.number_input("Filas por página", 50, 5000, 200, step=50)
        with c2:
            max_page = max(1, (total_rows + page_size - 1) // page_size)
            page = st.number_input("Página", 1, int(max_page), 1)
        with c3:
            st.metric("Total filas", total_rows)
        offset = (int(page) - 1) * int(page_size)
        df_browse = conn.execute(
            f"SELECT * FROM {quote_ident(sel_table)} LIMIT ? OFFSET ?;",
            [int(page_size), int(offset)],
        ).df()
        st.dataframe(df_browse, width='stretch', height=400)

# --- Sección: Editor de datos ---
with st.expander("✏️ Editar datos (inline)", expanded=False):
    if st.button("Actualizar", key="refresh_edit"):
        st.rerun()
    # Elegir tabla y claves
    try:
        tables_df_ed = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name;"
        ).df()
        table_opts_ed = tables_df_ed["table_name"].tolist()
    except Exception:
        table_opts_ed = []
    if not table_opts_ed:
        st.info("No hay tablas para editar.")
    else:
        t_edit = st.selectbox("Tabla a editar", options=table_opts_ed, key="edit_table")
        cols_meta_ed = get_table_columns(conn, t_edit)
        col_names_ed = [c["name"] for c in cols_meta_ed]
        # Sugerir clave por defecto
        default_keys = [c for c in col_names_ed if c.lower() == "id"] or []
        keys = st.multiselect("Columnas clave (para identificar filas)", options=col_names_ed, default=default_keys, help="Se usan para detectar filas nuevas/actualizadas/eliminadas.")
        if not keys:
            st.warning("Selecciona al menos una columna clave.")
        # Paginación
        total_rows_e = conn.execute(f"SELECT COUNT(*) FROM {quote_ident(t_edit)};").fetchone()[0]
        c1, c2, c3 = st.columns(3)
        with c1:
            page_size_e = st.number_input("Filas por página", 20, 1000, 50, step=10, key="edit_ps")
        with c2:
            max_page_e = max(1, (total_rows_e + page_size_e - 1) // page_size_e)
            page_e = st.number_input("Página", 1, int(max_page_e), 1, key="edit_pg")
        with c3:
            st.metric("Total filas", total_rows_e)
        offset_e = (int(page_e) - 1) * int(page_size_e)
        df_orig = conn.execute(
            f"SELECT * FROM {quote_ident(t_edit)} LIMIT ? OFFSET ?;",
            [int(page_size_e), int(offset_e)],
        ).df()
        # Excluir columnas protegidas del editor
        protected_cols = ["_ingest_batch_id"]
        df_view = df_orig.drop(columns=protected_cols, errors='ignore')
        allow_add = st.checkbox("Permitir añadir filas nuevas", value=False)
        allow_delete = st.checkbox("Permitir eliminar filas en esta página", value=False, help="No hay undo para borrados desde el editor (aún).")
        edited = st.data_editor(df_view, num_rows=("dynamic" if allow_add else "fixed"), key="editor_df")
        if st.button("Guardar cambios", type="primary"):
            try:
                import pandas as _pd
                if not keys:
                    st.error("Debes seleccionar al menos una clave.")
                else:
                    # Limitar a filas con claves no nulas
                    edited_valid = edited.dropna(subset=keys) if not edited.empty else edited
                    orig_valid = df_view.dropna(subset=keys) if not df_view.empty else df_view
                    # Sanitizar para DuckDB (None en vez de NaN)
                    edited_valid = _sanitize_for_duckdb(edited_valid)
                    orig_valid = _sanitize_for_duckdb(orig_valid)
                    # Índices por claves
                    idx_orig = orig_valid.set_index(keys, drop=False)
                    idx_edit = edited_valid.set_index(keys, drop=False)
                    # Nuevas y borradas
                    new_keys = idx_edit.index.difference(idx_orig.index)
                    del_keys = idx_orig.index.difference(idx_edit.index)
                    new_rows = idx_edit.loc[new_keys].reset_index(drop=True) if len(new_keys) else _pd.DataFrame(columns=edited_valid.columns)
                    del_rows = idx_orig.loc[del_keys][keys].reset_index(drop=True) if (allow_delete and len(del_keys)) else _pd.DataFrame(columns=keys)
                    # Detectar cambios en las comunes
                    common_keys = idx_edit.index.intersection(idx_orig.index)
                    upd_rows = _pd.DataFrame(columns=edited_valid.columns)
                    updated_cols = []
                    if len(common_keys):
                        left = idx_orig.loc[common_keys]
                        right = idx_edit.loc[common_keys]
                        non_key_cols = [c for c in col_names_ed if c not in keys and c not in protected_cols]
                        if non_key_cols:
                            # Comparar valores, tratando NaN como iguales
                            comp = (left[non_key_cols].fillna("__NaN__") != right[non_key_cols].fillna("__NaN__"))
                            diff_mask = comp.any(axis=1)
                            if diff_mask.any():
                                # Columnas cambiadas: union de difs por columna
                                changed_cols = [c for c in non_key_cols if comp[c].any()]
                                updated_cols = changed_cols
                                cols_for_update = keys + updated_cols
                                upd_rows = right.loc[diff_mask, cols_for_update].reset_index(drop=True)
                                upd_rows = _sanitize_for_duckdb(upd_rows)
                    # Preparar auditoría
                    ensure_audit_column(conn, t_edit)
                    batch_id = start_ingest_batch(conn, t_edit)
                    inserted_count = 0
                    updated_count = 0
                    # Log versiones de updates antes de aplicar
                    if not (upd_rows is None or upd_rows.empty):
                        # Construir filas previas con valores antiguos
                        pre_rows = []
                        l_prev = left.loc[diff_mask] if 'diff_mask' in locals() else _pd.DataFrame(columns=col_names_ed)
                        for _, r in l_prev.iterrows():
                            prev = {k: r[k] for k in keys}
                            for c in updated_cols:
                                prev[c] = r[c]
                            pre_rows.append(prev)
                        if pre_rows:
                            finalize_ingest_batch_meta(conn, batch_id, t_edit, keys, updated_cols)
                            log_update_versions(conn, t_edit, batch_id, pre_rows, keys, updated_cols)
                        updated_count = update_rows_from_df(conn, t_edit, upd_rows, keys, cols_to_update=updated_cols)
                    # Inserts
                    if not (new_rows is None or new_rows.empty):
                        new_rows = _sanitize_for_duckdb(new_rows)
                        inserted_count = insert_rows_from_df(conn, t_edit, new_rows, batch_id)
                    # Deletes
                    if allow_delete and not del_rows.empty:
                        delete_rows_by_keys(conn, t_edit, del_rows, keys)
                    finalize_ingest_batch(conn, batch_id, t_edit, inserted=inserted_count, updated=updated_count)
                    st.success(f"Cambios guardados. Insertados: {inserted_count}, Actualizados: {updated_count}. Para deshacer inserts/updates, usa Auditoría → Lotes.")
            except Exception as e:
                st.error(f"No se pudo guardar: {e}")

# --- Sección: Exportar ---
with st.expander("📤 Exportar datos", expanded=False):
    if st.button("Actualizar", key="refresh_export"):
        st.rerun()
    src = st.radio("Fuente", ["Tabla", "Último resultado"], horizontal=True)
    if src == "Último resultado" and "last_query_df" in st.session_state:
        exp_df = st.session_state["last_query_df"]
        st.caption(f"Filas: {len(exp_df)}")
        fmt = st.selectbox("Formato", ["CSV", "Excel", "Parquet"], index=0, key="fmt_result")
        fn = st.text_input("Nombre de archivo", value="export")
        if st.button("Generar archivo"):
            import io
            data = None
            mime = "text/csv"
            filename = fn
            try:
                if fmt == "CSV":
                    buf = io.StringIO()
                    exp_df.to_csv(buf, index=False)
                    data = buf.getvalue().encode("utf-8")
                    filename += ".csv"
                elif fmt == "Excel":
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                        exp_df.to_excel(writer, index=False)
                    data = buf.getvalue()
                    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    filename += ".xlsx"
                else:
                    try:
                        import pyarrow  # noqa: F401
                        buf = io.BytesIO()
                        exp_df.to_parquet(buf, index=False)
                        data = buf.getvalue()
                        mime = "application/octet-stream"
                        filename += ".parquet"
                    except Exception:
                        st.warning("Parquet no disponible, exportando a CSV")
                        buf = io.StringIO()
                        exp_df.to_csv(buf, index=False)
                        data = buf.getvalue().encode("utf-8")
                        filename += ".csv"
            except Exception as e:
                st.error(f"Error exportando: {e}")
                data = None
            if data is not None:
                st.download_button("Descargar", data=data, file_name=filename, mime=mime)
    else:
        # Exportar tabla completa
        try:
            tables_df = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name;"
            ).df()
            table_options = tables_df["table_name"].tolist()
        except Exception:
            table_options = []
        if not table_options:
            st.info("No hay tablas para exportar.")
        else:
            texp = st.selectbox("Tabla", options=table_options, key="export_table")
            fmt = st.selectbox("Formato", ["CSV", "Excel", "Parquet", "CSV (ZIP, por chunks)"], index=0, key="fmt_tab")
            fn = st.text_input("Nombre de archivo", value=f"{texp}", key="fn_tab")
            if fmt == "CSV (ZIP, por chunks)":
                chunk_size = st.number_input("Tamaño de chunk", min_value=10000, max_value=1000000, value=100000, step=10000, key="chunk_sz")
            if st.button("Generar archivo de tabla"):
                import io
                try:
                    data = None
                    mime = "text/csv"
                    filename = fn
                    if fmt == "CSV":
                        df_all = conn.execute(f"SELECT * FROM {quote_ident(texp)};").df()
                        buf = io.StringIO()
                        df_all.to_csv(buf, index=False)
                        data = buf.getvalue().encode("utf-8")
                        filename += ".csv"
                    elif fmt == "Excel":
                        df_all = conn.execute(f"SELECT * FROM {quote_ident(texp)};").df()
                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                            df_all.to_excel(writer, index=False)
                        data = buf.getvalue()
                        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        filename += ".xlsx"
                    elif fmt == "Parquet":
                        try:
                            import pyarrow  # noqa: F401
                            df_all = conn.execute(f"SELECT * FROM {quote_ident(texp)};").df()
                            buf = io.BytesIO()
                            df_all.to_parquet(buf, index=False)
                            data = buf.getvalue()
                            mime = "application/octet-stream"
                            filename += ".parquet"
                        except Exception:
                            st.warning("Parquet no disponible, exportando a CSV")
                            df_all = conn.execute(f"SELECT * FROM {quote_ident(texp)};").df()
                            buf = io.StringIO()
                            df_all.to_csv(buf, index=False)
                            data = buf.getvalue().encode("utf-8")
                            filename += ".csv"
                    else:  # CSV (ZIP, por chunks)
                        import zipfile
                        zip_buf = io.BytesIO()
                        with zipfile.ZipFile(zip_buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                            csv_buf = io.StringIO()
                            # Cabecera + chunks
                            offset = 0
                            wrote_header = False
                            while True:
                                chunk_df = conn.execute(
                                    f"SELECT * FROM {quote_ident(texp)} LIMIT ? OFFSET ?;",
                                    [int(chunk_size), int(offset)],
                                ).df()
                                if chunk_df.empty:
                                    break
                                chunk_df.to_csv(csv_buf, index=False, header=(not wrote_header))
                                wrote_header = True
                                offset += int(chunk_size)
                            zf.writestr(f"{texp}.csv", csv_buf.getvalue().encode('utf-8'))
                        data = zip_buf.getvalue()
                        mime = "application/zip"
                        filename += ".zip"
                    st.download_button("Descargar tabla", data=data, file_name=filename, mime=mime)
                except Exception as e:
                    st.error(f"Error exportando: {e}")

# --- Sección: Esquema (cambiar tipos de columnas) ---
with st.expander("🧩 Esquema: cambiar tipo de columna", expanded=False):
    if st.button("Actualizar", key="refresh_schema"):
        st.rerun()
    try:
        tables_df3 = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name;"
        ).df()
        table_options3 = tables_df3["table_name"].tolist()
    except Exception:
        table_options3 = []

    if not table_options3:
        st.info("No hay tablas para editar esquema.")
    else:
        t3 = st.selectbox(
            "Tabla",
            options=table_options3,
            index=(table_options3.index(table_name) if table_name in table_options3 else 0),
            key="schema_table",
        )
        cols_meta3 = get_table_columns(conn, t3)
        col_names3 = [c["name"] for c in cols_meta3]
        col_sel = st.selectbox("Columna", options=col_names3, key="schema_col")
        cur_type = next((c["type"] for c in cols_meta3 if c["name"] == col_sel), "?")
        st.caption(f"Tipo actual: {cur_type}")

        # Tipos comunes de DuckDB
        type_opts = [
            "BOOLEAN",
            "BIGINT",
            "INTEGER",
            "DOUBLE",
            "DECIMAL(18,2)",
            "VARCHAR",
            "DATE",
            "TIMESTAMP",
        ]
        new_type = st.selectbox("Nuevo tipo", options=type_opts, index=type_opts.index("VARCHAR") if cur_type.upper() != "VARCHAR" else 5)
        mode = st.radio("Modo de conversión", ["Seguro (TRY_CAST)", "Estricto (CAST)"] , horizontal=True)

        cpa, cpb = st.columns(2)
        with cpa:
            if st.button("Verificar posibles fallos"):
                try:
                    failed = count_failed_casts(conn, t3, col_sel, new_type)
                    if failed > 0:
                        st.warning(f"{failed} filas fallarían la conversión (quedarían NULL en modo seguro).")
                    else:
                        st.success("No se detectaron fallos de conversión.")
                except Exception as e:
                    st.error(f"Error verificando: {e}")
        with cpb:
            if st.button("Aplicar cambio de tipo", type="primary"):
                try:
                    strict = mode.startswith("Estricto")
                    res = change_column_type(conn, t3, col_sel, new_type, strict=strict)
                    if strict:
                        st.success("Tipo cambiado en modo estricto.")
                    else:
                        st.success(f"Tipo cambiado. Fallos de conversión: {res.get('failed_conversions', 0)}")
                except Exception as e:
                    st.error(f"No se pudo cambiar el tipo: {e}")

        st.divider()
        st.subheader("Otras operaciones de esquema")
        c1, c2, c3 = st.columns(3)
        with c1:
            new_col = st.text_input("Añadir columna: nombre", key="add_col_name")
            new_type2 = st.selectbox("Tipo", ["VARCHAR","INTEGER","BIGINT","DOUBLE","DATE","TIMESTAMP","BOOLEAN","DECIMAL(18,2)"], key="add_col_type")
            if st.button("Añadir columna"):
                try:
                    schema_add_column(conn, t3, new_col, new_type2)
                    st.success(f"Columna {new_col} añadida")
                except Exception as e:
                    st.error(f"No se pudo añadir: {e}")
        with c2:
            oldn = st.text_input("Renombrar: nombre actual", key="ren_old")
            newn = st.text_input("Renombrar: nombre nuevo", key="ren_new")
            if st.button("Renombrar columna"):
                try:
                    schema_rename_column(conn, t3, oldn, newn)
                    st.success(f"Renombrada {oldn} → {newn}")
                except Exception as e:
                    st.error(f"No se pudo renombrar: {e}")
        with c3:
            delc = st.text_input("Eliminar columna: nombre", key="drop_col")
            if st.button("Eliminar columna"):
                try:
                    schema_drop_column(conn, t3, delc)
                    st.success(f"Eliminada columna {delc}")
                except Exception as e:
                    st.error(f"No se pudo eliminar: {e}")

# --- Sección: Builder de filtros (sin SQL) + Vistas guardadas ---
with st.expander("🧰 Builder de filtros (sin SQL) y Vistas", expanded=False):
    if st.button("Actualizar", key="refresh_filters"):
        st.rerun()
    # Elegir tabla base
    try:
        tables_df2 = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name;"
        ).df()
        table_options2 = tables_df2["table_name"].tolist()
    except Exception:
        table_options2 = []

    if not table_options2:
        st.info("No hay tablas para filtrar todavía.")
    else:
        tb = st.selectbox("Tabla base", options=table_options2, index=(table_options2.index(table_name) if table_name in table_options2 else 0), key="filter_table")
        # Columnas y tipos
        try:
            cols_meta = get_table_columns(conn, tb)
        except Exception:
            cols_meta = []
        col_names = [c["name"] for c in cols_meta]
        types_map = {c["name"]: c["type"].upper() for c in cols_meta}

        # Construcción de condiciones dinámicas
        st.subheader("Condiciones")
        logic = st.radio("Operador entre condiciones", ["AND", "OR"], horizontal=True, key="logic_op")
        if "filters" not in st.session_state:
            st.session_state["filters"] = []

        def render_condition(idx: int):
            cols = st.columns([3, 3, 4, 1])
            with cols[0]:
                col = st.selectbox("Columna", options=col_names, key=f"f_col_{idx}", index=0 if col_names else None)
            if not col:
                return None
            typ = types_map.get(col, "VARCHAR")
            # Operadores según tipo
            if "INT" in typ or "DOUBLE" in typ or "DECIMAL" in typ or "HUGEINT" in typ:
                ops = ["=", "!=", ">", ">=", "<", "<=", "BETWEEN"]
            elif "TIMESTAMP" in typ or "DATE" in typ:
                ops = [">=", "<=", "BETWEEN", "=", "!="]
            else:
                ops = ["=", "!=", "CONTAINS", "STARTS WITH", "ENDS WITH", "IN"]
            with cols[1]:
                op = st.selectbox("Operador", options=ops, key=f"f_op_{idx}")
            with cols[2]:
                if op == "BETWEEN":
                    v1 = st.text_input("Desde", key=f"f_v1_{idx}")
                    v2 = st.text_input("Hasta", key=f"f_v2_{idx}")
                    val = {"v1": v1, "v2": v2}
                elif op == "IN":
                    v = st.text_input("Valores (separados por coma)", key=f"f_v_{idx}")
                    val = [s.strip() for s in v.split(",") if s.strip()]
                else:
                    val = st.text_input("Valor", key=f"f_v_{idx}")
            with cols[3]:
                remove = st.button("✖", key=f"f_rm_{idx}")
            if remove:
                st.session_state["filters"].pop(idx)
                st.rerun()
            return {"column": col, "op": op, "value": val}

        # Render existing filters
        if st.session_state["filters"]:
            for i in range(len(st.session_state["filters"])):
                st.session_state["filters"][i] = render_condition(i)

        cA, cB = st.columns([1, 3])
        with cA:
            if st.button("Añadir condición"):
                st.session_state["filters"].append({"column": None, "op": "=", "value": ""})
                st.rerun()

        # Construir WHERE parametrizado y ejecutar
        where_clauses = []
        params: list = []
        for f in [x for x in st.session_state["filters"] if x and x.get("column")]:
            col = f["column"]
            op = f["op"]
            val = f["value"]
            if op == "BETWEEN" and isinstance(val, dict):
                where_clauses.append(f"{quote_ident(col)} BETWEEN ? AND ?")
                params.extend([val.get("v1"), val.get("v2")])
            elif op == "IN" and isinstance(val, list) and val:
                placeholders = ",".join(["?"] * len(val))
                where_clauses.append(f"{quote_ident(col)} IN ({placeholders})")
                params.extend(val)
            elif op == "CONTAINS":
                where_clauses.append(f"{quote_ident(col)} ILIKE ?")
                params.append(f"%{val}%")
            elif op == "STARTS WITH":
                where_clauses.append(f"{quote_ident(col)} ILIKE ?")
                params.append(f"{val}%")
            elif op == "ENDS WITH":
                where_clauses.append(f"{quote_ident(col)} ILIKE ?")
                params.append(f"%{val}")
            else:
                where_clauses.append(f"{quote_ident(col)} {op} ?")
                params.append(val)

        # Construcción correcta del separador AND/OR
        if where_clauses:
            sep = f" {logic} "
            where_sql = sep.join(where_clauses)
        else:
            where_sql = ""
        q_base = f"SELECT * FROM {quote_ident(tb)}"
        q = q_base + (f" WHERE {where_sql}" if where_sql else "") + " LIMIT 5000"  # cap para UI
        try:
            df_f = conn.execute(q, params).df()
            st.dataframe(df_f, width='stretch', height=350)
            st.session_state["last_query_df"] = df_f
        except Exception as e:
            st.error(f"Error ejecutando filtros: {e}")

        # Guardar vista
        with st.container():
            st.subheader("Guardar vista")
            view_name = st.text_input("Nombre de la vista")
            if st.button("Guardar vista", type="primary"):
                try:
                    spec = {"table": tb, "logic": logic, "filters": st.session_state["filters"]}
                    save_view(conn, tb, view_name or f"vista_{tb}", spec)
                    st.success("Vista guardada")
                except Exception as e:
                    st.error(f"No se pudo guardar la vista: {e}")

        # Listar y aplicar vistas guardadas
        st.subheader("Vistas guardadas")
        try:
            views_df = list_views(conn, tb)
            if not views_df.empty:
                view_sel = st.selectbox("Elegir vista", options=views_df["id"].tolist(), format_func=lambda vid: f"#{vid} – " + views_df.set_index("id").loc[vid, "view_name"])
                if st.button("Aplicar vista seleccionada"):
                    spec = load_view_spec(conn, int(view_sel))
                    if spec:
                        st.session_state["filters"] = spec.get("filters", [])
                        st.rerun()
            else:
                st.info("No hay vistas guardadas para esta tabla.")
        except Exception as e:
            st.error(f"No se pudieron listar vistas: {e}")

# --- Sección: Gráficas y Dashboards ---
with st.expander("📊 Gráficas y Dashboards", expanded=False):
    if st.button("Actualizar", key="refresh_dash"):
        st.rerun()
    # Fuente: usar resultado filtrado si existe, si no, tabla destino
    src_choice = st.radio("Fuente de datos", ["Tabla destino", "Último filtrado/consulta"], horizontal=True)
    if src_choice == "Último filtrado/consulta" and "last_query_df" in st.session_state:
        data_df = st.session_state["last_query_df"]
    else:
        try:
            data_df = conn.execute(f"SELECT * FROM {quote_ident(table_name)} LIMIT 5000;").df()
        except Exception:
            data_df = pd.DataFrame()

    if data_df.empty:
        st.info("No hay datos para graficar todavía.")
    else:
        cols = list(data_df.columns)
        c1, c2, c3 = st.columns(3)
        with c1:
            mark = st.selectbox("Tipo", ["line", "bar", "scatter"], index=0, key="chart_mark")
        with c2:
            col_x = st.selectbox("Eje X", cols, index=0, key="chart_x")
        with c3:
            col_y = st.selectbox("Métrica Y", cols, index=min(1, len(cols)-1), key="chart_y")
        agg = st.selectbox("Agregación", ["sum", "avg", "count", "min", "max"], index=0, key="chart_agg")

        try:
            q = f"""
            SELECT {quote_ident(col_x)} AS x,
                   {agg}({quote_ident(col_y)}) AS y
            FROM {quote_ident(table_name)}
            GROUP BY 1
            ORDER BY 1
            """
            chart_df = conn.execute(q).df()
            if mark == "line":
                fig = px.line(chart_df, x="x", y="y", markers=True)
            elif mark == "bar":
                fig = px.bar(chart_df, x="x", y="y")
            else:
                fig = px.scatter(chart_df, x="x", y="y")
            st.plotly_chart(fig, width='stretch', key='main_chart')
        except Exception as e:
            st.error(str(e))

        # Guardar gráfico
        with st.container():
            chart_name = st.text_input("Nombre del gráfico")
            if st.button("Guardar gráfico", type="primary"):
                try:
                    spec = {
                        "mark": mark,
                        "x": col_x,
                        "y": col_y,
                        "agg": agg,
                        "table": table_name,
                    }
                    save_chart(conn, table_name, chart_name or f"graf_{table_name}", spec)
                    st.success("Gráfico guardado")
                except Exception as e:
                    st.error(f"No se pudo guardar: {e}")

        # Dashboards: crear y ver
        st.subheader("Dashboards")
        try:
            charts_df = list_charts(conn, table_name)
        except Exception:
            charts_df = pd.DataFrame()

        if charts_df.empty:
            st.info("No hay gráficos guardados aún.")
        else:
            sel_charts = st.multiselect(
                "Elegir gráficos para el dashboard",
                options=charts_df["id"].tolist(),
                format_func=lambda cid: f"#{cid} – " + charts_df.set_index("id").loc[cid, "chart_name"],
            )
            d_name = st.text_input("Nombre del dashboard")
            d_desc = st.text_input("Descripción (opcional)")
            if st.button("Guardar dashboard"):
                try:
                    save_dashboard(conn, d_name or "dashboard", sel_charts, d_desc)
                    st.success("Dashboard guardado")
                except Exception as e:
                    st.error(f"No se pudo guardar dashboard: {e}")

        st.subheader("Ver dashboards")
        try:
            ddf = list_dashboards(conn)
        except Exception:
            ddf = pd.DataFrame()
        if ddf.empty:
            st.info("No hay dashboards guardados.")
        else:
            dash_sel = st.selectbox(
                "Dashboard",
                options=ddf["id"].tolist(),
                format_func=lambda did: f"#{did} – " + ddf.set_index("id").loc[did, "dashboard_name"],
                key="dash_select",
            )
            if st.button("Renderizar dashboard"):
                try:
                    charts_specs = load_dashboard_charts(conn, int(dash_sel))
                    # Render simple en columnas
                    cols = st.columns(2)
                    for i, c in enumerate(charts_specs):
                        spec = c["spec"]
                        mark = spec.get("mark", "line")
                        col_x, col_y, agg = spec.get("x"), spec.get("y"), spec.get("agg", "sum")
                        q = f"""
                        SELECT {quote_ident(col_x)} AS x,
                               {agg}({quote_ident(col_y)}) AS y
                        FROM {quote_ident(table_name)}
                        GROUP BY 1
                        ORDER BY 1
                        """
                        cdf = conn.execute(q).df()
                        with cols[i % 2]:
                            st.markdown(f"**{c['name']}**")
                            if mark == "line":
                                fig = px.line(cdf, x="x", y="y", markers=True)
                            elif mark == "bar":
                                fig = px.bar(cdf, x="x", y="y")
                            else:
                                fig = px.scatter(cdf, x="x", y="y")
                            st.plotly_chart(fig, width='stretch', key=f"dash_chart_{c['id']}_{i}")
                except Exception as e:
                    st.error(f"Error al renderizar dashboard: {e}")

# --- Sección: Auditoría ---
with st.expander("🧾 Auditoría (ingesta y esquema)", expanded=False):
    # Ingesta: listar lotes y deshacer
    try:
        tables_df4 = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name;"
        ).df()
        table_options4 = tables_df4["table_name"].tolist()
    except Exception:
        table_options4 = []
    if not table_options4:
        st.info("No hay tablas para auditar.")
    else:
        t4 = st.selectbox("Tabla", options=table_options4, key="audit_table")
        st.subheader("Lotes de ingesta")
        try:
            batches = list_ingest_batches(conn, t4)
            if batches.empty:
                st.info("Sin lotes registrados.")
            else:
                st.dataframe(batches, width='stretch', height=240)
                bid = st.number_input("ID de lote", min_value=1, step=1, key="undo_bid")
                c_undo_i, c_undo_u = st.columns(2)
                with c_undo_i:
                    if st.button("Deshacer INSERTs"):
                        try:
                            deleted = undo_ingest_batch(conn, t4, int(bid))
                            st.success(f"Eliminadas {deleted} filas del lote #{int(bid)}")
                        except Exception as e:
                            st.error(f"No se pudo deshacer: {e}")
                with c_undo_u:
                    if st.button("Deshacer UPDATEs"):
                        try:
                            from src.db import undo_update_batch
                            restored = undo_update_batch(conn, t4, int(bid))
                            st.success(f"Revertidas {restored} filas actualizadas del lote #{int(bid)}")
                        except Exception as e:
                            st.error(f"No se pudo revertir updates: {e}")
        except Exception as e:
            st.error(f"Error listando lotes: {e}")

        st.subheader("Cambios de esquema")
        try:
            se = list_schema_events(conn, t4)
            if se.empty:
                st.info("Sin eventos de esquema.")
            else:
                st.dataframe(se, width='stretch', height=200)
                st.caption("Para revertir cambios de tipo, usa el panel de Esquema seleccionando el tipo anterior.")
        except Exception as e:
            st.error(f"Error listando eventos de esquema: {e}")

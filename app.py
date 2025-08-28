# app.py
from __future__ import annotations
import streamlit as st
import plotly.express as px
import pandas as pd
from src.db import connect_db, quote_ident
from src.ingest import load_excel_to_df, smart_casts, upsert_df

st.set_page_config(page_title="Excel → DuckDB (Local)", layout="wide")

# --- Sidebar: Instancia ---
st.sidebar.header("Instancia (archivo .duckdb)")
db_path = st.sidebar.text_input("Ruta del archivo .duckdb", value="data/local.duckdb")
if st.sidebar.button("Abrir/Crear instancia", use_container_width=True):
    st.session_state["conn"] = connect_db(db_path)
    st.sidebar.success(f"Conectado a {db_path}")

conn = st.session_state.get("conn")
if not conn:
    st.info("👈 Elige una ruta y pulsa **Abrir/Crear instancia** en la barra lateral.")
    st.stop()

st.title("Ingesta de Excel → SQL local (DuckDB)")
st.caption("100% local, sin servidor. Carga, normaliza, inserta y explora.")

# --- Sección: Ingesta ---
with st.expander("📥 Subir Excel e ingestar", expanded=True):
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
        st.dataframe(preview_df, use_container_width=True)

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
with st.expander("🔎 Explorar con SQL", expanded=True):
    default_sql = f"SELECT * FROM {quote_ident(table_name)} LIMIT 100;"
    sql = st.text_area("Consulta SQL", value=default_sql, height=140)
    if st.button("Ejecutar consulta"):
        try:
            res = conn.execute(sql).df()
            st.dataframe(res, use_container_width=True)
            st.session_state["last_query_df"] = res
        except Exception as e:
            st.error(str(e))

# --- Sección: Gráficas ---
with st.expander("📊 Gráfica rápida (agregación)", expanded=True):
    # Fuente: última tabla consultada o tabla destino
    src_choice = st.radio("Fuente de datos", ["Tabla destino", "Último resultado de consulta"], horizontal=True)
    if src_choice == "Último resultado de consulta" and "last_query_df" in st.session_state:
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
        col_x = st.selectbox("Eje X", cols, index=0)
        col_y = st.selectbox("Métrica Y (numérica)", cols, index=min(1, len(cols)-1))
        agg = st.selectbox("Agregación", ["sum", "avg", "count", "min", "max"], index=0)
        # Construir query de agregación
        try:
            q = f"""
            SELECT {quote_ident(col_x)} AS x,
                   {agg}({quote_ident(col_y)}) AS y
            FROM {quote_ident(table_name)}
            GROUP BY 1
            ORDER BY 1
            """
            chart_df = conn.execute(q).df()
            fig = px.line(chart_df, x="x", y="y", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(str(e))
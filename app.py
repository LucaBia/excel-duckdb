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
    save_view,
    list_views,
    load_view_spec,
    save_chart,
    list_charts,
    save_dashboard,
    list_dashboards,
    load_dashboard_charts,
)
from src.ingest import load_excel_to_df, smart_casts, upsert_df

st.set_page_config(page_title="Excel → DuckDB (Local)", layout="wide")

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
with st.expander("🔎 Explorar con SQL", expanded=True):
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
with st.expander("🗂️ Visualizador de tablas (todas las filas)", expanded=True):
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
        sel_table = st.selectbox("Tabla", options=table_options, index=(table_options.index(table_name) if table_name in table_options else 0))
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

# --- Sección: Esquema (cambiar tipos de columnas) ---
with st.expander("🧩 Esquema: cambiar tipo de columna", expanded=False):
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

# --- Sección: Builder de filtros (sin SQL) + Vistas guardadas ---
with st.expander("🧰 Builder de filtros (sin SQL) y Vistas", expanded=True):
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
with st.expander("📊 Gráficas y Dashboards", expanded=True):
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
            mark = st.selectbox("Tipo", ["line", "bar", "scatter"], index=0)
        with c2:
            col_x = st.selectbox("Eje X", cols, index=0)
        with c3:
            col_y = st.selectbox("Métrica Y", cols, index=min(1, len(cols)-1))
        agg = st.selectbox("Agregación", ["sum", "avg", "count", "min", "max"], index=0)

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

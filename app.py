# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from functools import lru_cache

# =========================
# CONFIGURACIÓN
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parquet_dir = BASE_DIR

radios = [20140903, 20141005]

# =========================
# SHAPE (RELATIVO!!)
# =========================

shp_path = os.path.join(BASE_DIR, "CABA+suptechosOK-5-2.shp")

@lru_cache
def load_shape():
    gdf = gpd.read_file(shp_path).to_crs(4326)
    gdf["REDCODE"] = gdf["REDCODE"].astype(int)
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf["geometry"] = gdf.geometry.simplify(0.0001, preserve_topology=True)
    return gdf

gdf = load_shape()
geojson = gdf.__geo_interface__

# =========================
# MAPA
# =========================

def build_map(selected_radio=None):
    fig = go.Figure()

    # base
    fig.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        locations=gdf["REDCODE"],
        z=[1]*len(gdf),
        featureidkey="properties.REDCODE",
        colorscale=[[0, "#ff9999"], [1, "#ff9999"]],
        marker_opacity=0.4,
        marker_line_width=0.3,
        showscale=False
    ))

    # radios con info
    gdf_info = gdf[gdf["REDCODE"].isin(radios)]

    fig.add_trace(go.Choroplethmapbox(
        geojson=gdf_info.__geo_interface__,
        locations=gdf_info["REDCODE"],
        z=[1]*len(gdf_info),
        featureidkey="properties.REDCODE",
        colorscale=[[0, "#cc0000"], [1, "#cc0000"]],
        marker_opacity=0.8,
        marker_line_width=1,
        showscale=False
    ))

    # seleccionado
    if selected_radio is not None:
        selected = gdf[gdf["REDCODE"] == selected_radio]

        fig.add_trace(go.Choroplethmapbox(
            geojson=selected.__geo_interface__,
            locations=selected["REDCODE"],
            z=[1],
            featureidkey="properties.REDCODE",
            colorscale=[[0, "gold"], [1, "gold"]],
            marker_opacity=1,
            marker_line_width=2,
            showscale=False
        ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=10,  # más abierto
        mapbox_center={
            "lat": gdf.geometry.centroid.y.mean(),
            "lon": gdf.geometry.centroid.x.mean()
        },
        margin=dict(l=0, r=0, t=40, b=0),
        title="Census radios",
        uirevision="keep-zoom"
    )

    return fig

# =========================
# DASH APP
# =========================

app = Dash(__name__)
server = app.server  # 👈 IMPORTANTE PARA RENDER

app.layout = html.Div([
    dcc.Store(id="selected_radio_store"),

    html.Div(style={"display": "flex", "height": "100vh"}, children=[

        html.Div(style={"width": "45%"}, children=[
            dcc.Graph(id="map", style={"height": "100%"}, config={"scrollZoom": True})
        ]),

        html.Div(style={"width": "55%", "padding": "10px"}, children=[
            html.Label("Month"),
            dcc.Dropdown(
                id="month_selector",
                options=[{"label": "January", "value": 1},
                         {"label": "July", "value": 7}],
                value=1
            ),

            html.Br(),

            html.Label("Electrification (%)"),
            dcc.Slider(0, 100, 10, value=0, id="slider_elec"),

            html.Label("Retrofitting (%)"),
            dcc.Slider(0, 100, 10, value=0, id="slider_retro"),

            html.Label("PV (%)"),
            dcc.Slider(0, 100, 10, value=0, id="slider_fv"),

            dcc.Checklist(
                id="bess_enable",
                options=[{"label": "Activate BESS", "value": "ON"}],
                value=[]
            ),

            html.Br(),

            html.Div(id="energy_counters", style={"margin": "10px 0", "fontWeight": "bold"}),

            dcc.Graph(id="graph", style={"height": "35vh"}),
            dcc.Graph(id="soc_graph", style={"height": "35vh"})
        ])
    ])
])

# =========================
# CALLBACKS
# =========================

@app.callback(
    Output("selected_radio_store", "data"),
    Input("map", "clickData"),
    prevent_initial_call=True
)
def map_click(clickData):
    if clickData:
        return int(clickData["points"][0]["location"])
    return None

@app.callback(
    Output("map", "figure"),
    Input("selected_radio_store", "data")
)
def update_map(selected_radio):
    return build_map(selected_radio)

@app.callback(
    Output("graph", "figure"),
    Output("soc_graph", "figure"),
    Output("energy_counters", "children"),
    Input("selected_radio_store", "data"),
    Input("month_selector", "value"),
    Input("slider_elec", "value"),
    Input("slider_retro", "value"),
    Input("slider_fv", "value"),
    Input("bess_enable", "value")
)
def update_curves_parquet(radio, mes, elec_pct, retro_pct, fv_pct, bess_enable):

    if radio is None:
        return go.Figure(), go.Figure(), ""

    fname = os.path.join(parquet_dir, f"radio_{radio}_mes{mes}.parquet")

    if not os.path.exists(fname):
        return go.Figure(), go.Figure(), "Archivo no encontrado"

    df = pd.read_parquet(fname)

    row = df[
        (df["electrification"] == elec_pct) &
        (df["retrofitting"] == retro_pct) &
        (df["pv"] == fv_pct) &
        (df["bess"] == ("ON" if "ON" in bess_enable else "OFF"))
    ]

    if row.empty:
        return go.Figure(), go.Figure(), "No data"

    curva_base = np.array(row["curva_base"].iloc[0])
    curva_fv = np.array(row["curva_fv"].iloc[0])
    curva_total = np.array(row["curva_total"].iloc[0])
    soc = np.array(row["soc"].iloc[0])

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=curva_fv, name="PV"))
    fig.add_trace(go.Scatter(y=curva_total, name="Total"))
    fig.add_trace(go.Scatter(y=curva_base, name="Base"))

    fig.update_layout(title=f"Radio {radio}", height=300)

    fig_soc = go.Figure()
    fig_soc.add_trace(go.Scatter(y=soc, name="SOC"))

    return fig, fig_soc, "OK"

# =========================
# RUN LOCAL
# =========================

if __name__ == "__main__":
    app.run(debug=True)
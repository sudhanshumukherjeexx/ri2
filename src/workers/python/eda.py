import json

import plotly.express as px
import plotly.graph_objects as go

# Replaces the ~17 duplicated *Analyzer classes in the original app's
# utilities/viz.py (one class per chart type, each re-implementing
# "render -> write PNG -> GPT-vision analyze") with a single spec table +
# dispatcher. Chart construction is reused almost verbatim from the
# original plotly.express/graph_objects calls; only the Streamlit/PNG/
# GPT-vision plumbing around them is gone (AI insight is now a shared,
# text-based component instead of a per-chart screenshot analyzer).
CHART_SPECS = {
    "box": {
        "label": "Box Plot",
        "category": "Basic",
        "fields": [
            {"name": "y", "label": "Value", "type": "numeric"},
            {"name": "x", "label": "Group by", "type": "categorical", "optional": True},
        ],
    },
    "histogram": {
        "label": "Histogram",
        "category": "Basic",
        "fields": [{"name": "x", "label": "Value", "type": "numeric"}],
    },
    "scatter": {
        "label": "Scatter Plot",
        "category": "Basic",
        "fields": [
            {"name": "x", "label": "X", "type": "numeric"},
            {"name": "y", "label": "Y", "type": "numeric"},
            {"name": "color", "label": "Color by", "type": "categorical", "optional": True},
        ],
    },
    "bar": {
        "label": "Bar Chart",
        "category": "Basic",
        "fields": [
            {"name": "x", "label": "Category", "type": "categorical"},
            {"name": "y", "label": "Value (blank = count)", "type": "numeric", "optional": True},
        ],
    },
    "pie": {
        "label": "Pie Chart",
        "category": "Basic",
        "fields": [
            {"name": "names", "label": "Category", "type": "categorical"},
            {"name": "values", "label": "Value (blank = count)", "type": "numeric", "optional": True},
        ],
    },
    "line": {
        "label": "Line Plot",
        "category": "Basic",
        "fields": [
            {"name": "x", "label": "X", "type": "any"},
            {"name": "y", "label": "Y", "type": "numeric"},
        ],
    },
    "violin": {
        "label": "Violin Plot",
        "category": "Advanced",
        "fields": [
            {"name": "y", "label": "Value", "type": "numeric"},
            {"name": "x", "label": "Group by", "type": "categorical", "optional": True},
        ],
    },
    "contour": {
        "label": "Contour Plot",
        "category": "Advanced",
        "fields": [
            {"name": "x", "label": "X", "type": "numeric"},
            {"name": "y", "label": "Y", "type": "numeric"},
        ],
    },
    "histcontour": {
        "label": "2D Hist Contour",
        "category": "Advanced",
        "fields": [
            {"name": "x", "label": "X", "type": "numeric"},
            {"name": "y", "label": "Y", "type": "numeric"},
        ],
    },
    "scatter3d": {
        "label": "3D Scatter",
        "category": "Advanced",
        "fields": [
            {"name": "x", "label": "X", "type": "numeric"},
            {"name": "y", "label": "Y", "type": "numeric"},
            {"name": "z", "label": "Z", "type": "numeric"},
        ],
    },
    "line3d": {
        "label": "3D Line",
        "category": "Advanced",
        "fields": [
            {"name": "x", "label": "X", "type": "numeric"},
            {"name": "y", "label": "Y", "type": "numeric"},
            {"name": "z", "label": "Z", "type": "numeric"},
        ],
    },
    "polarscatter": {
        "label": "Polar Scatter",
        "category": "Specialized",
        "fields": [
            {"name": "theta", "label": "Angle (theta)", "type": "any"},
            {"name": "r", "label": "Radius (r)", "type": "numeric"},
        ],
    },
    "polarbar": {
        "label": "Polar Bar",
        "category": "Specialized",
        "fields": [
            {"name": "theta", "label": "Angle (theta)", "type": "categorical"},
            {"name": "r", "label": "Radius (r)", "type": "numeric"},
        ],
    },
    "scattergeo": {
        "label": "Scatter Map",
        "category": "Geospatial",
        "fields": [
            {"name": "lat", "label": "Latitude", "type": "numeric"},
            {"name": "lon", "label": "Longitude", "type": "numeric"},
        ],
    },
    "choropleth": {
        "label": "Choropleth Map",
        "category": "Geospatial",
        "fields": [
            {"name": "locations", "label": "Location code", "type": "categorical"},
            {"name": "values", "label": "Value", "type": "numeric"},
            {
                "name": "locationmode",
                "label": "Location type",
                "type": "choice",
                "options": ["USA-states", "country names", "ISO-3"],
            },
        ],
    },
    "bubblemap": {
        "label": "Bubble Map",
        "category": "Geospatial",
        "fields": [
            {"name": "lat", "label": "Latitude", "type": "numeric"},
            {"name": "lon", "label": "Longitude", "type": "numeric"},
            {"name": "size", "label": "Bubble size", "type": "numeric", "optional": True},
        ],
    },
}

MAX_POINTS = 5000


@register("eda_specs")
def _eda_specs(args):
    return CHART_SPECS


# Chart kinds where each row renders as its own point/marker -- subsampling
# only reduces visual density, it doesn't change what the chart "means".
# Every other kind (histogram, bar, pie, contour/histcontour density, box,
# violin, polar bar, choropleth) aggregates or counts rows, so subsampling
# would silently change the displayed numbers, not just the resolution.
SAMPLEABLE_KINDS = {"scatter", "scatter3d", "line", "line3d", "polarscatter", "scattergeo", "bubblemap"}


def _maybe_sample(df, kind):
    if kind in SAMPLEABLE_KINDS and len(df) > MAX_POINTS:
        return df.sample(MAX_POINTS, random_state=42)
    return df


@register("eda_figure")
def _eda_figure(args):
    source_key = args["source_key"]
    kind = args["kind"]
    fields = args.get("fields", {})

    if kind not in CHART_SPECS:
        raise ValueError(f"Unknown chart kind: {kind}")

    df = _maybe_sample(get_df(source_key), kind)

    if kind == "box":
        fig = px.box(df, y=fields["y"], x=fields.get("x") or None)
    elif kind == "histogram":
        fig = px.histogram(df, x=fields["x"])
    elif kind == "scatter":
        fig = px.scatter(df, x=fields["x"], y=fields["y"], color=fields.get("color") or None)
    elif kind == "bar":
        if fields.get("y"):
            fig = px.bar(df, x=fields["x"], y=fields["y"])
        else:
            counts = df[fields["x"]].value_counts().reset_index()
            counts.columns = [fields["x"], "count"]
            fig = px.bar(counts, x=fields["x"], y="count")
    elif kind == "pie":
        if fields.get("values"):
            fig = px.pie(df, names=fields["names"], values=fields["values"])
        else:
            counts = df[fields["names"]].value_counts().reset_index()
            counts.columns = [fields["names"], "count"]
            fig = px.pie(counts, names=fields["names"], values="count")
    elif kind == "line":
        fig = px.line(df.sort_values(fields["x"]), x=fields["x"], y=fields["y"])
    elif kind == "violin":
        fig = px.violin(df, y=fields["y"], x=fields.get("x") or None, box=True)
    elif kind == "contour":
        fig = px.density_contour(df, x=fields["x"], y=fields["y"])
    elif kind == "histcontour":
        fig = go.Figure(go.Histogram2dContour(x=df[fields["x"]], y=df[fields["y"]]))
    elif kind == "scatter3d":
        fig = px.scatter_3d(df, x=fields["x"], y=fields["y"], z=fields["z"])
    elif kind == "line3d":
        fig = go.Figure(
            go.Scatter3d(x=df[fields["x"]], y=df[fields["y"]], z=df[fields["z"]], mode="lines")
        )
    elif kind == "polarscatter":
        fig = px.scatter_polar(df, theta=fields["theta"], r=fields["r"])
    elif kind == "polarbar":
        fig = px.bar_polar(df, theta=fields["theta"], r=fields["r"])
    elif kind == "scattergeo":
        fig = px.scatter_geo(df, lat=fields["lat"], lon=fields["lon"])
    elif kind == "choropleth":
        fig = px.choropleth(
            df,
            locations=fields["locations"],
            color=fields["values"],
            locationmode=fields.get("locationmode") or "USA-states",
        )
    elif kind == "bubblemap":
        fig = px.scatter_geo(df, lat=fields["lat"], lon=fields["lon"], size=fields.get("size") or None)

    fig.update_layout(margin=dict(t=40, b=40))
    return json.loads(fig.to_json())

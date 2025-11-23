import psycopg2
import pandas as pd
import plotly.graph_objects as go

# --------------------------------
# CONNECT TO POSTGRES
# --------------------------------
conn = psycopg2.connect(
    host="localhost",
    port="5432",
    database="options_data",
    user="postgres",
    password="postgres"
)

# --------------------------------
# LOAD ALL DISTINCT DATES FIRST
# --------------------------------
dates = pd.read_sql("SELECT DISTINCT asofdate FROM options ORDER BY asofdate;", conn)
dates["asofdate"] = pd.to_datetime(dates["asofdate"])

frames = []
first_day_df = None

for idx, row in dates.iterrows():
    day = row["asofdate"]

    query = """
    SELECT
        (data->'attributes'->>'strike')::float AS strike,
        (data->'attributes'->>'exp_date') AS exp_date
    FROM options
    WHERE asofdate = %s
    """

    df = pd.read_sql(query, conn, params=[day])
    df["exp_date"] = pd.to_datetime(df["exp_date"])

    if df.empty:
        continue

    # Save the first day's data for initial frame
    if first_day_df is None:
        first_day_df = df.copy()

    # Add Plotly frame
    frames.append(go.Frame(
        data=[go.Scatter(
            x=df["strike"],
            y=df["exp_date"],
            mode="markers",
            marker=dict(size=6)
        )],
        name=str(day.date())
    ))

conn.close()

# --------------------------------
# PLOT INITIAL FIG
# --------------------------------
fig = go.Figure(
    data=[go.Scatter(
        x=first_day_df["strike"],
        y=first_day_df["exp_date"],
        mode="markers",
        marker=dict(size=6)
    )],
    frames=frames,
)

# Set layout
fig.update_layout(
    title="Strike vs Expiration — Animated by Date",
    xaxis_title="Strike",
    yaxis_title="Expiration Date",
    updatemenus=[
        {
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 200, "redraw": True}}],
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                }
            ]
        }
    ]
)

# Add slider
fig.update_layout(
    sliders=[{
        "steps": [
            {"method": "animate",
             "args": [[f.name], {"mode": "immediate", "frame": {"duration": 0}}],
             "label": f.name}
            for f in frames
        ]
    }]
)

fig.show()

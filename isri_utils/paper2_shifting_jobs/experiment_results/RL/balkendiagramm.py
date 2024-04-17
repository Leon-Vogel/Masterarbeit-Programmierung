import time
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SA_min_mean = np.mean([0.889, 0.937, 0.924, 0.906, 0.911, 0.913, 0.942, 0.909, 0.903])
SA_mean_mean = np.mean([0.93, 0.957, 0.945, 0.94, 0.943, 0.954, 0.963, 0.946, 0.94])
SA_std_mean = np.mean([0.018, 0.013, 0.012, 0.016, 0.013, 0.02, 0.011, 0.016, 0.019])

RL_min_mean_seen = np.mean([0.859, 0.912, 0.906, 0.877, 0.890])
RL_mean_mean_seen = np.mean([0.892, 0.940, 0.932, 0.930, 0.919,])
RL_std_mean_seen = np.mean([0.016, 0.021, 0.016, 0.022, 0.016,])

RL_min_mean_UNseen = np.mean([0.876, 0.917, 0.896, 0.859,])
RL_mean_mean_UNseen = np.mean([0.936, 0.952, 0.938, 0.909,])
RL_std_mean_UNseen = np.mean([0.026, 0.026, 0.027, 0.023,])

fig = go.Figure()

categories = ['SA', 'RL_unseen', 'RL_seen']
values_min = [SA_min_mean, RL_min_mean_UNseen, RL_min_mean_seen]
values_mean = [SA_mean_mean, RL_mean_mean_UNseen, RL_mean_mean_seen]
values_std = [SA_std_mean, RL_std_mean_UNseen, RL_std_mean_seen]
pastel_colors = ['#AECDC2', '#FFD7A5', '#FF9FB2']
pastel_colors = ['#808080', '#1E90FF', '#104E8B']
legend_names = categories

fig = make_subplots(rows=1, cols=3, subplot_titles=('min', 'mean', 'std'), horizontal_spacing=0.15)

fig.add_trace(go.Bar(x=categories, y=values_min, marker_color=pastel_colors, showlegend=False), row=1, col=1)
fig.add_trace(go.Bar(x=categories, y=values_mean, marker_color=pastel_colors, showlegend=False), row=1, col=2)
# fig.add_trace(go.Bar(x=categories, y=values_std, marker_color=pastel_colors, name=legend_names, showlegend=True),
#               row=1, col=3)
for cat, value, color in zip(categories, values_std , pastel_colors):
    fig.add_trace(go.Bar(x=[cat], y=[value], marker_color=color, name=cat), row=1, col=3)


fig.update_layout(
    height=350,
    plot_bgcolor='rgba(0,0,0,0)',  # Transparente Hintergrundfarbe
    paper_bgcolor='rgba(0,0,0,0)',  # Transparente Hintergrundfarbe
    bargap=0.1,
    xaxis=dict(showticklabels=False),
    xaxis2=dict(showticklabels=False),
    xaxis3=dict(showticklabels=False),
    legend=dict(font=dict(size=14)),
    margin=dict(l=10, r=10, b=10, t=10)
)
y1_tickvals = [0.86, 0.88, 0.90, 0.92]
y2_tickvals = [0.89, 0.91, 0.93, 0.95]
y3_tickvals = [0.011, 0.016, 0.021, 0.027]
# y1_tickvals = [0.862, 0.88, 0.896, 0.912, 0.918]
# y2_tickvals = [0.9, 0.915, 0.93, 0.945, 0.96]
# y3_tickvals = [0.011, 0.015, 0.019, 0.023, 0.027]

fig.update_yaxes(range=[0.858, 0.922], tickvals=y1_tickvals, row=1, col=1)
fig.update_yaxes(range=[0.889, 0.951], tickvals=y2_tickvals, row=1, col=2)
fig.update_yaxes(range=[0.011, 0.027], tickvals=y3_tickvals, row=1, col=3)

fig.show()

fig.write_image("balkendiagramm_ta30.pdf")
time.sleep(1)
fig.write_image("balkendiagramm_ta30.pdf")
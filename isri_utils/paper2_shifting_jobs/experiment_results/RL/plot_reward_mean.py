import pandas as pd
import time
import plotly.graph_objects as go

PATH = r'dist1.02_3obj_smallerR_POL_DR_ta30\t_epslen75.csv'
df = pd.read_csv(PATH)
print(df)

fig = go.Figure()

fig.add_trace(go.Scatter(x=df['Step'], y=df['Value'], mode='lines', name='reward_mean'))

fig.update_layout(
    # title='Value vs Step',
    xaxis=dict(title='step',  showgrid=False, showline=True, linewidth=2, linecolor='black'),
    yaxis=dict(title='reward',  showgrid=False, showline=True, linewidth=2, linecolor='black', tickvals=[8, 12, 16]),
    plot_bgcolor='white',  # Hintergrundfarbe des Diagramms
    paper_bgcolor='white',  # Hintergrundfarbe des gesamten Bereichs
    height=200,
    margin=dict(l=10, r=10, b=10, t=10)  # Margins um die Grafik herum. Dies kann je nach Ihren Anforderungen angepasst werden.
)
fig.write_image("reward_mean_ta30.pdf")
time.sleep(1)
fig.write_image("reward_mean_ta30.pdf")
# fig.show()
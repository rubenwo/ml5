import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt


@st.cache(allow_output_mutation=True)
def get_data():
    c = pd.read_csv("./model/coordinates.csv", index_col=0, squeeze=True)
    # coordinates = coordinates.rename(columns={0: 'X', 1: 'Y'})
    c = c.add_prefix('X')
    y = pd.read_csv("./model/prediction.csv", index_col=0)
    y = y.add_prefix('Y')
    i = pd.read_csv("./model/indices.csv", index_col=0, squeeze=True, header=0)
    # indices = pd.Series.from_csv('./model/indices.csv', header=0)
    return i, c, y


def clamp_min(n, minn):
    return max(n, minn)


def clamp_max(n, maxn):
    return min(n, maxn)


indices, coordinates, y_pred = get_data()

name_input = st.sidebar.selectbox('Learning Object Name', indices.index.values.tolist())
zoom_input = st.sidebar.slider('Select the Zoom parameter', 1, 10)

idx = indices[name_input]
coordinate = coordinates.loc[idx, :]


def plot_clusters_st_alt(X, target):
    zv_x_min, zv_x_max = abs((-80 - (target[0] - 30)) / zoom_input), abs((80 - (target[0] + 30)) / zoom_input)
    zv_y_min, zv_y_max = abs((-80 - (target[1] - 30)) / zoom_input), abs((80 - (target[1] + 30)) / zoom_input)
    x_min, x_max = target[0] - (30 + zv_x_min), target[0] + (30 + zv_x_max)
    y_min, y_max = target[1] - (30 + zv_y_min), target[1] + (30 + zv_y_max)

    plot = alt.Chart(X).mark_point(clip=True).encode(
        x=alt.X('X0', scale=alt.Scale(domain=(clamp_min(x_min, -80), clamp_max(x_max, 80)))),
        y=alt.Y('X1', scale=alt.Scale(domain=(clamp_min(y_min, -80), clamp_max(y_max, 80)))),
        # x='X0',
        # y='X1',
        color=alt.Color('Y', scale=alt.Scale(scheme='dark2'))
    ).properties(
        height=800,
        width=800
    )

    target_df = pd.DataFrame([(target + ("Learning Object",))], columns=["X0", "X1", "label"])
    plot2 = alt.Chart(target_df).mark_point(filled=True, size=150).encode(
        x='X0',
        y='X1',
        color=alt.value('red')
    )

    plot3 = alt.Chart(target_df).mark_text(align="left", baseline="middle", fontSize=18, fontStyle="bold").encode(
        x='X0',
        y='X1',
        text="label"
    )

    st.altair_chart((plot + plot2 + plot3), use_container_width=True)


coordinates['Pred'] = y_pred['Y0']
plot_clusters_st_alt(coordinates, (coordinate[0], coordinate[1]))

import streamlit as st
from src.data import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from urllib.request import urlopen
import json


def app():
    st.markdown("## Exploratory Data Analysis")
    st.write("\n")
    data = fetch_data()

    # app, out = st.columns(2)
    config, chart=st.columns((4,6))

    country_filter, state_filter, county_check = None, None, None
    # Now this will show the filtered row in the dataframe as you change the inputs
    config_expander = config.expander('Configuration')
    # Add user input 
    country_check = config_expander.checkbox('Country')
    if country_check:
        country_filter = config_expander.selectbox('Select Country', ('United States', ""))

    state_check = config_expander.checkbox('State')
    if state_check:
        state_filter = config_expander.selectbox("Select State", data["State"].unique())

    county_check = config_expander.checkbox('County')
    if county_check:
        county_filter = config_expander.selectbox("Select County", data["County"].unique())

    if state_check and county_check:
        data = data[(data["State"] == state_filter) & (data["County"] == county_filter)]
        config.dataframe(data)
        config.info("(Rows, Columns) : {}".format(data.shape))

    elif state_check:
        data = data[(data["State"] == state_filter)]
        config.dataframe(data)
        config.info("Shape (Rows, Columns) : {}".format(data.shape))

    elif county_check:
        data = data[(data["County"] == county_check)]
        config.dataframe(data)
        config.info("Shape (Rows, Columns) : {}".format(data.shape))

    else:
        config.dataframe(data)
        config.info("Shape (Rows, Columns) : {}".format(data.shape))

    values = ['Per_Capita_Income','Perc_Diff_Income_County_vs_State',
              'Per_Capita_GDP', 'UNP_Rate',
              'higher_degree', 'Population_Density']

    predictor_var1 = "Crime_Density_Per_1000"#chart.selectbox("Select the Social First Indicator Trend", values, index=values.index("Crime_Density_Per_1000"))
    predictor_var2 = chart.selectbox("Select the Social Second Indicator Trend", values, index=values.index("UNP_Rate"))

    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), ax=ax, annot=True)
    config.write(fig)

    temp1 = data.groupby(by=["Year"]).sum()[predictor_var1]
    temp2 = data.groupby(by=["Year"]).sum()[predictor_var2]

    x_index = list(temp1.index)
    y1 = temp1.values
    y2 = temp2.values

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=x_index, y=y1, name=predictor_var1),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=x_index, y=y2, name=predictor_var2),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title_text="{} v/s {}".format(predictor_var1, predictor_var2)
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Year")

    # Set y-axes titles
    fig.update_yaxes(title_text="{}".format(predictor_var1), secondary_y=False)
    fig.update_yaxes(title_text="{}".format(predictor_var2), secondary_y=True)

    chart.plotly_chart(fig)


    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)

    df_pred = data.groupby(["GeoFIPS"])[predictor_var1].agg('mean').to_frame()
    df_pred["GeoFIPS"] = df_pred.index
    # print(df_pred.head())

    fig2 = px.choropleth_mapbox(df_pred, geojson=counties, locations='GeoFIPS', color=predictor_var1,
                            color_continuous_scale="Viridis",
                            range_color=(0, 12),
                            mapbox_style="carto-positron",
                            zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                            opacity=0.5,
                            labels={'unemp':'unemployment rate'}
                            )
    fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    chart.plotly_chart(fig2)

    df_pred2 = data.groupby(["GeoFIPS"])[predictor_var2].agg('mean').to_frame()
    df_pred2["GeoFIPS"] = df_pred2.index
    # print(df_pred.head())

    fig3 = px.choropleth_mapbox(df_pred2, geojson=counties, locations='GeoFIPS', color=predictor_var2,
                            color_continuous_scale="Viridis",
                            range_color=(0, 12),
                            mapbox_style="carto-positron",
                            zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
                            opacity=0.5,
                            labels={'unemp':'unemployment rate'}
                            )
    fig3.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    chart.plotly_chart(fig3)
import streamlit as st
import pickle
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from src.data import *


def generate_dummies(x, dummies, var):
        vector = []
        dums = dict(zip(dummies.columns, [0]*dummies.shape[1]))
        for k, v in dums.items():
            if k == x[var]:
                vector.append(1)
            else:
                vector.append(0)
        temp = pd.Series(dict(zip(dummies.columns, vector)))
        return temp


def app():
    st.markdown("## Crime Predictor Page!")
    st.write("\n")
    model, sc, metro_dummies, state_dummies = None, None, None, None

    with open('models/model.pkl', 'rb') as handle:
        model = pickle.load(handle)

    with open('models/standardScaler.pkl', 'rb') as handle:
        sc = pickle.load(handle)

    with open("../crime_analytics/models/dense_state.txt", "rb") as fp:
        dense_state = pickle.load(fp)

    geo_df = pd.read_csv("models/county_data.csv")
    geo_df = geo_df[geo_df['State'].isin(dense_state)].reset_index(drop=True)   
    
    State_dummies = pd.read_csv('models/State_dummies.csv')
    Metro_dummies = pd.read_csv('models/Metro_dummies.csv')

    past_data = fetch_data()
    
    num_cols = ['Per_Capita_Income', 'Perc_Diff_Income_County_vs_State', 'Per_Capita_GDP', 'UNP_Rate', 'higher_degree', 'Population_Density']


    col1, col2 = st.columns(2)
    state = col1.selectbox("Select State", geo_df["State"].unique())

    county = col2.selectbox("Select County", geo_df[geo_df["State"] == state]["County"].unique())

    county_area = geo_df[(geo_df["State"] == state) & (geo_df["County"] == county)]["County_Area"].unique()[0]

    population = col1.number_input("Population", value=54773.0)  
    gdp = col2.number_input("GDP", value=1222286.0)  
    per_capita_income = col1.number_input("Per Capita Income", value=33348.0)
    state_avg_income = col2.number_input("State Average Income", value=42590)
    unp_rate = col1.number_input("Unemployment Rate", value=8.8)
    higher_degree = col2.number_input("Higher Degree Percentage", value=26.571573)
    year = col1.selectbox('Year', range(2019, 2050))

    population_density = population / county_area

    per_capita_gdp = gdp/population

    diff_income = per_capita_income - state_avg_income

    perc_diff_income = diff_income/per_capita_income*100

    metro = geo_df[(geo_df["State"] == state) & (geo_df["County"] == county)]["Metro"].unique()[0]

    test_dict = {'Per_Capita_Income': per_capita_income,
                 'Perc_Diff_Income_County_vs_State': perc_diff_income,
                 'Per_Capita_GDP': per_capita_gdp, 'UNP_Rate': unp_rate,
                 'higher_degree': higher_degree, 'Population_Density': population_density,
                 'Metro': metro, "State": state}

    x_test = pd.Series(test_dict)

    predict = st.button("Predict")
    if predict:
        x_test_scaled = sc.transform(x_test[num_cols].values.reshape(1,-1))
        x_test.loc[num_cols] = x_test_scaled[0]

        metro_dum = generate_dummies(x_test, Metro_dummies, "Metro")
        state_dum = generate_dummies(x_test, State_dummies, "State")
        del x_test["State"]
        del x_test["Metro"]
        x_test = pd.concat([x_test, metro_dum, state_dum])

        pred = round(model.predict(x_test.values.reshape(1,-1))[0], 4)

        past_data = past_data[past_data["State"] == state]
        temp = past_data.groupby(by=["Year"]).agg('mean')["Crime_Density_Per_1000"]

        # print(temp)

        predictor_var1 = 'Crime_Density_Per_1000'

        x_index = list(temp.index)
        x_index.append(year)
        y = list(temp.values)
        y.append(pred)

        fig = px.line(x = x_index, #Columns from the data frame
                    y = y,
                    title = "Crime Density per 1000 Trend",
                    markers=True
                    )
        fig.add_scatter(x = [fig.data[0].x[-1]],
                        y = [fig.data[0].y[-1]],
                        text="Prediction",
                        mode='markers+text',
                        marker=dict(color='red', size=10),
                        textfont=dict(color='green', size=10),
                        textposition='top left',
                        showlegend=False)
        # fig.update_traces(line_color = "blue")
        st.plotly_chart(fig)

        last_datapoint = (round(y[-2], 4) * population)/1000
        pred = (pred*population)/1000

        st.metric("Total Crime", "{} Predicted".format(int(pred)), "{} Last Available Data".format(int(last_datapoint)))
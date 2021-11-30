# Crime Analytics

Crime in society has been on the rise globally, being particularly pronounced in urban areas, and is a major concern for law enforcement and social services. Scholars, institutions, and governments have undertaken studies to understand the causes of crime, and varied opinions have emerged, commonly identified causes include increased urbanization, nuclear families, population densities, education, income disparities and unemployment. A vast body of studies have concluded that social indicators are a key determinant of crime. However, despite the overwhelming amount of material on this subject, it is difficult to get a comprehensive depiction of the causal relationships between crime and social indicators. Further there is lack of visual presentation of data and crime models, particularly a model that can be applied to entire US or any state or county of US. Our project collects crime data and data on predictors from several sources, performs data cleaning, exploratory analysis and modeling to provide analytics and a model to predict crime in the US.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Crime Analytics.

## Python Packages
Pre-requisite: This dashboard assumes Anaconda is already installed with Python==3.8. 

If not, follow the below steps:

1. Download and install [Anaconda](https://www.anaconda.com/products/individual). Once that is working...

2. Create your Anaconda env: 
```bash
conda create --name gatech_env python=3.8
```

3. Activate the anaconda environment: 
```bash
conda activate gatech_env
```

We need below packages to be installed in the created conda environment.
```python
streamlit
numpy
pandas
plotly
sqlalchemy
scikit-learn
scipy
seaborn
matplotlib
jupyterlab
statsmodels
```
To install run the below command:
```bash
pip install -r requirements.txt
```
## For Running the application
```bash
streamlit run main_app.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
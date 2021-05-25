import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
#visualization
import matplotlib.pyplot as plt
#for analysis 
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

	

st.title("A simple data science app")
def main():
	
	Opt = st.sidebar.selectbox('Select any of the below',('EDA','Visualization','Classifier'))
	
	if Opt=='EDA':
		st.subheader("Exploratory Data Analysis")

		data = st.file_uploader("Upload Dataset :",type = ('csv','xslx','txt','json'))
		if data is not None:
			st.success("Data successfully loaded") 
			df = pd.read_csv(data)
			st.dataframe(df.head(50))

			if st.checkbox("Display shape"):
				st.write(df.shape)
			if st.checkbox("Display Columns"):
				st.write(df.columns)
			if st.checkbox("Select multiple columns"):
				selected_columns = st.multiselect('Select preferred columns:',df.columns)
				df1=df[selected_columns]
				st.dataframe(df1)
			if st.checkbox("Display Summary"):
				st.write(df.describe().T)
			if st.checkbox("Display Null Values"):
				st.write(df.isnull().sum())
			if st.checkbox("Display Datatype"):
				st.write(df.dtypes)
			if st.checkbox("Display Corelation"):
				st.write(df.Corelation)
	elif Opt=='Visualization':
		st.subheader("Visualization")

		data = st.file_uploader("Upload Dataset :",type = ('csv','xslx','txt','json'))
		if data is not None:
			st.success("Data successfully loaded") 
			df = pd.read_csv(data)
			st.dataframe(df.head(50))

		if st.checkbox('Select multiple columns to plot'):
			selected_columns = st.multiselect('Select your preferred columns',df.columns)
			df1 = df[selected_columns]
			st.dataframe(df1)
		if st.checkbox('Display heatmap'):
			st.write(sns.heatmap(df.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
			st.set_option('deprecation.showPyplotGlobalUse', False)
			st.pyplot()

		if st.checkbox('Display Pairplot'):
			st.write(sns.pairplot(df1,diag_kind='kde'))
			st.pyplot()
		if st.checkbox('Display PieChart'):
			all_columns = df.columns.to_list()
			pie_columns=st.selectbox("Selecet column to display",all_columns)
			pieChart = df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
			st.write(pieChart)


	elif Opt=='Classifier':
		st.subheader("Classifier")
main()

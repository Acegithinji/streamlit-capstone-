import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
@st.cache_data
def load_data():
    file_path = "healthcare-stroke-data.csv"  
    return pd.read_csv(file_path)

data = load_data()

# Title and description
st.title("Stroke Data Analysis Dashboard")
st.markdown("""
This dashboard allows you to explore the relationships between stroke occurrence and various features.
Use the filters below to interact with the data.
""")

# Filters
st.sidebar.header("Filters")

# Dropdowns for categorical variables
gender_filter = st.sidebar.multiselect("Gender", data["gender"].unique(), default=data["gender"].unique())
work_type_filter = st.sidebar.multiselect("Work Type", data["work_type"].unique(), default=data["work_type"].unique())
residence_filter = st.sidebar.multiselect("Residence Type", data["Residence_type"].unique(), default=data["Residence_type"].unique())
smoking_filter = st.sidebar.multiselect("Smoking Status", data["smoking_status"].unique(), default=data["smoking_status"].unique())

# Sliders for numerical variables
age_range = st.sidebar.slider("Age Range", int(data["age"].min()), int(data["age"].max()), (0, 100))
glucose_range = st.sidebar.slider("Average Glucose Level", float(data["avg_glucose_level"].min()), float(data["avg_glucose_level"].max()), (50.0, 200.0))
bmi_range = st.sidebar.slider("BMI Range", float(data["bmi"].min(skipna=True)), float(data["bmi"].max(skipna=True)), (10.0, 50.0))

# Checkbox for excluding null BMI
exclude_null_bmi = st.sidebar.checkbox("Exclude Null BMI", value=True)

# Filter the dataset
filtered_data = data[
    (data["gender"].isin(gender_filter)) &
    (data["work_type"].isin(work_type_filter)) &
    (data["Residence_type"].isin(residence_filter)) &
    (data["smoking_status"].isin(smoking_filter)) &
    (data["age"].between(age_range[0], age_range[1])) &
    (data["avg_glucose_level"].between(glucose_range[0], glucose_range[1]))
]
if exclude_null_bmi:
    filtered_data = filtered_data[filtered_data["bmi"].notnull()]
filtered_data = filtered_data[filtered_data["bmi"].between(bmi_range[0], bmi_range[1])]

# Overview of the filtered dataset
st.subheader("Filtered Dataset")
st.write(filtered_data.describe())

# Visualization: Bar chart for categorical features
st.subheader("Stroke Occurrence by Categorical Features")
cat_feature = st.selectbox("Select Categorical Feature", ["gender", "work_type", "Residence_type", "smoking_status"])
fig, ax = plt.subplots()
sns.countplot(data=filtered_data, x=cat_feature, hue="stroke", ax=ax)
st.pyplot(fig)

# Visualization: Scatter plot for numerical features
st.subheader("Scatter Plot: Numerical Features vs Stroke")
num_x = st.selectbox("X-Axis Feature", ["age", "avg_glucose_level", "bmi"])
num_y = st.selectbox("Y-Axis Feature", ["age", "avg_glucose_level", "bmi"], index=1)
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_data, x=num_x, y=num_y, hue="stroke", palette="coolwarm", ax=ax)
st.pyplot(fig)

# Visualization: Correlation heatmap
st.subheader("Correlation Heatmap (Numerical Features)")
corr_data = filtered_data[["age", "avg_glucose_level", "bmi", "stroke"]].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_data, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

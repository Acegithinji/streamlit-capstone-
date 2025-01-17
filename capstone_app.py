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
st.title("Stroke Data Analysis Dashboard 🧠")
st.markdown(""" **Stroke Data Analysis Dashboard**!  
""")

# Overview of the dataset
st.subheader("Dataset Overview 📊")
st.write(data.describe())

# Slicer: Gender
st.subheader("Slicer: Gender")
selected_gender = st.multiselect("Select Gender", options=data["gender"].unique(), default=data["gender"].unique())

# Slicer: Work Type
st.subheader("Slicer: Work Type")
selected_work_type = st.multiselect("Select Work Type", options=data["work_type"].unique(), default=data["work_type"].unique())

# Slicer: Residence Type
st.subheader("Slicer: Residence Type")
selected_residence = st.multiselect("Select Residence Type", options=data["Residence_type"].unique(), default=data["Residence_type"].unique())

# Slicer: Smoking Status
st.subheader("Slicer: Smoking Status")
selected_smoking_status = st.multiselect("Select Smoking Status", options=data["smoking_status"].unique(), default=data["smoking_status"].unique())

# Sliders for numerical variables
st.subheader("Slicers for Numerical Variables")
age_range = st.slider("Select Age Range", int(data["age"].min()), int(data["age"].max()), (0, 100))
glucose_range = st.slider("Select Average Glucose Level Range", float(data["avg_glucose_level"].min()), float(data["avg_glucose_level"].max()), (50.0, 200.0))
bmi_range = st.slider("Select BMI Range", float(data["bmi"].min(skipna=True)), float(data["bmi"].max(skipna=True)), (10.0, 50.0))

# Filter the dataset
filtered_data = data[
    (data["gender"].isin(selected_gender)) &
    (data["work_type"].isin(selected_work_type)) &
    (data["Residence_type"].isin(selected_residence)) &
    (data["smoking_status"].isin(selected_smoking_status)) &
    (data["age"].between(age_range[0], age_range[1])) &
    (data["avg_glucose_level"].between(glucose_range[0], glucose_range[1])) &
    (data["bmi"].between(bmi_range[0], bmi_range[1], inclusive="both"))
]

# Visualization: Stroke Occurrence by Categorical Features
st.subheader("Stroke Occurrence by Categorical Features")
cat_feature = st.selectbox("Select Categorical Feature for Analysis", ["gender", "work_type", "Residence_type", "smoking_status", "ever_married"])
fig, ax = plt.subplots()
sns.countplot(data=filtered_data, x=cat_feature, hue="stroke", ax=ax)
ax.set_title(f"Stroke Occurrence by {cat_feature.capitalize()}")
st.pyplot(fig)

# Visualization: Stroke occurrence vs. Marital Status
st.subheader("Stroke Occurrence by Marital Status 💍")
fig, ax = plt.subplots()
sns.countplot(data=filtered_data, x="ever_married", hue="stroke", palette="pastel", ax=ax)
ax.set_title("Stroke Occurrence by Marital Status")
st.pyplot(fig)

# Visualization: Scatter plot for numerical features
st.subheader("Scatter Plot: Numerical Features vs Stroke")
num_x = st.selectbox("X-Axis Feature", ["age", "avg_glucose_level", "bmi"])
num_y = st.selectbox("Y-Axis Feature", ["age", "avg_glucose_level", "bmi"], index=1)
fig, ax = plt.subplots()
sns.scatterplot(data=filtered_data, x=num_x, y=num_y, hue="stroke", palette="coolwarm", ax=ax)
ax.set_title(f"Scatter Plot of {num_x.capitalize()} vs {num_y.capitalize()}")
st.pyplot(fig)

# Visualization: Correlation heatmap
st.subheader("Correlation Heatmap (Numerical Features) 🔥")
corr_data = filtered_data[["age", "avg_glucose_level", "bmi", "stroke"]].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_data, annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)


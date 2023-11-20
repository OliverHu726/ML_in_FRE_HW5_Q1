import numpy as np
import pandas as pd
import streamlit as st

file_path = "https://raw.githubusercontent.com/OliverHu726/ML_in_FRE_HW5_Q1/main/submissions.csv"
df = pd.read_csv(file_path)

Accurate_status = []
for n in range(len(df)):
    if df["Ground Truth"][n] == df["Predictions"][n]:
        Accurate_status.append(1)
    else:
        Accurate_status.append(0)
df['Accurate'] = Accurate_status
df['Pclass'] = df['Pclass'].astype(str)
total_acc = sum(Accurate_status) / len(Accurate_status)

def calculate_accuracy(data, category):
  acc_metric = data.groupby(category).mean()["Accurate"]
  return acc_metric
  
st.title('Titanic ML Project Analysis')
# Dropdown menu
selected_category = st.selectbox('Select a category:', ['Sex', 'Pclass'])
# Calculate accuracy based on the selected category
acc_metric = calculate_accuracy(df, selected_category)
# Display results
st.write(f'Total Accuracy: {total_acc}')
for i in range(acc_metric.shape[0]):
    st.write({acc_metric.index[i], ' Accuracy: ', {acc_metric[i]})


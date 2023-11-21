import pandas as pd
import numpy as np
import streamlit as st

file_path = "https://raw.githubusercontent.com/OliverHu726/ML_in_FRE_HW5_Q1/main/submissions.csv"
df = pd.read_csv(file_path)
df['Pclass'] = df['Pclass'].astype(str)

Accurate_status = []
for n in range(len(df)):
    if df["Ground Truth"][n] == df["Predictions"][n]:
        Accurate_status.append(1)
    else:
        Accurate_status.append(0)
df["Accurate"] = Accurate_status
total_acc = sum(Accurate_status) / len(Accurate_status)

def calculate_accuracy(category):
  acc_metric = df.groupby(category)['Accurate'].mean()
  return acc_metric
  
st.title('dh3517_ML_in_FRE_HW5_Q1')
# Dropdown menu
selected_category = st.selectbox('Select a category:', ['Sex', 'Pclass'])
# Calculate accuracy based on the selected category
acc_metric = calculate_accuracy(selected_category)
# Display results
st.write('Total Accuracy: ', total_acc)
for i in range(acc_metric.shape[0]):
    st.write(acc_metric.index[i], ' Accuracy: ', acc_metric[i])


import pandas as pd

df=pd.read_csv("chest_pain_patients_test.csv", engine='python')
print(df.columns)

df_subset = df[['Past_Medical_History', 'Outpatient_Medications']]

print(df_subset.columns)

df.to_csv("chest_pain_patients_test_subset.csv", index=False)

df_subset.drop(columns="Past_Medical_History").to_csv("chest_pain_patients_test_problems_only.csv", index=False)
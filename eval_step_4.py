import pandas as pd

df = pd.read_csv("./test_result_csv/mode_3.csv")

# Filter only the MRI 
mask = df["sid"].str.contains("AnatCorrLungs", case=True, na=False)

df = df[mask]

# df = df[~mask] # Filter only the CT

df.to_csv("./test_result_csv/mode_3_mri.csv", index=False)
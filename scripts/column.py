import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv("models/parse/Ecoform_Dataset_with_ComfortIndex_clean_V1 (1).csv")  # Or pd.read_csv() if it's a CSV

# 1. Find the most common zone
top_zone = df['Zone (string)'].mode()[0]
zone_mask = df['Zone (string)'] == top_zone

# 2. Filter by dB values
# Define pass/fail thresholds (customize as needed)
pass_min, pass_max = 33, 36   # Scores 'close to 35'
fail_min = 36                 # 'Fail' is 36 or higher

pass_rows = df[zone_mask & (df['L(a)eq (dB)'] >= pass_min) & (df['L(a)eq (dB)'] < pass_max)]
fail_rows = df[zone_mask & (df['L(a)eq (dB)'] >= fail_min)]

# 3. Determine 10% sample size
total_top_zone = zone_mask.sum()
sample_size = max(1, int(np.round(0.1 * total_top_zone)))

# Split the sample between pass/fail (as balanced as possible)
half_sample = sample_size // 2
pass_sample = pass_rows.sample(n=min(half_sample, len(pass_rows)), random_state=42)
fail_sample = fail_rows.sample(n=min(sample_size - len(pass_sample), len(fail_rows)), random_state=42)

# Concatenate the indices to update
replace_indices = pd.concat([pass_sample, fail_sample]).index

# 4. Replace [zone] with 'Greenzone' for these samples
df.loc[replace_indices, 'Zone (string)'] = 'Greenzone'

# 5. Save the result
df.to_excel("models/parse/parsed_data.xlsx", index=False)

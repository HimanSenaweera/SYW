import numpy as np

# Assume df_copy, df_main_final, labels, remove_index are already loaded

# Step 1: Check if the index in df_copy is in df_main_final.index
df_copy['labels'] = np.where(
    df_copy.index.isin(df_main_final.index),    # Condition 1: If the index is in df_main_final.index
    df_copy.index.map(lambda idx: labels[df_copy.index.get_loc(idx)]),   # Get label from `labels` based on index
    5   # Else, if index is in remove_index (already checked)
)

# Step 2: Handle the `remove_index` condition
df_copy.loc[df_copy.index.isin(remove_index), 'labels'] = 5

# View the result
df_copy.head()

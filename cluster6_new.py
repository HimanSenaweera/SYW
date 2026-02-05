# Step 1: Set the 'labels' in df_copy to match the values from y_pred based on index
df_copy.loc[df_copy.index.isin(y_pred_index), 'labels'] = y_pred.loc[y_pred_index, 0]

# Step 2: Handle the 'remove_index' condition (set 'labels' to 5 for indices in remove_index)
df_copy.loc[df_copy.index.isin(remove_index), 'labels'] = 5

# View the result
df_copy.head()

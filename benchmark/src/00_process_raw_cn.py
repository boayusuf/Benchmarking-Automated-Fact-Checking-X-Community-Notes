import pandas as pd

# Load the 3 files (from inside folders)
notes_1 = pd.read_csv('notes-00000/notes-00000.tsv', sep='\t')
notes_2 = pd.read_csv('notes-00001/notes-00001.tsv', sep='\t')
status = pd.read_csv('noteStatusHistory-00000/noteStatusHistory-00000.tsv', sep='\t')

# Combine the two notes files
notes = pd.concat([notes_1, notes_2], ignore_index=True)

# Join them together
df = notes.merge(status, on='noteId', how='inner')

# Keep only notes with clear verdicts
df_clean = df[df['currentStatus'].isin(['CURRENTLY_RATED_HELPFUL', 'CURRENTLY_RATED_NOT_HELPFUL'])]

# Show results
print(f"Total notes with clear verdicts: {len(df_clean)}")
print(f"\nHELPFUL: {(df_clean['currentStatus'] == 'CURRENTLY_RATED_HELPFUL').sum()}")
print(f"NOT_HELPFUL: {(df_clean['currentStatus'] == 'CURRENTLY_RATED_NOT_HELPFUL').sum()}")
print(f"\nFirst few rows:")
print(df_clean.head())

# Save the cleaned dataset
df_clean.to_csv('community_notes_clean.csv', index=False)
print(f"\n✓ Saved to community_notes_clean.csv")


import pandas as pd

df = pd.read_csv('community_notes_clean.csv')

# See what columns exist
print("Columns:", df.columns.tolist())
print("\nFirst row:")
print(df.iloc[0])

# Check if there's text data
if 'summary' in df.columns:
    print(f"\nExample note summary:\n{df['summary'].iloc[0]}")


    import pandas as pd

df = pd.read_csv('community_notes_clean.csv')

# Show 20 random summaries
print("Sample Community Note summaries:\n")
for i, summary in enumerate(df['summary'].sample(20)):
    print(f"{i+1}. {summary[:150]}...\n")
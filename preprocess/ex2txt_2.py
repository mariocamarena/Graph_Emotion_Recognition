import pandas as pd

# Read the Excel file, skipping the first row
df = pd.read_excel("C:/Users/Mario/OneDrive/Desktop/gcn/graph_emotionv1/CARLA_Data/keys/keysR_PostPilot_final.xlsx", skiprows=1, header=None)

# Open the output file
with open("C:/Users/Mario/OneDrive/Desktop/gcn/graph_emotionv1/CARLA_Data/keys/customIDSvR2.txt", 'w') as f:
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Extract the first number from the first column
        subject_num = row[0].split('_')[0]
        
        # Construct the new string for the first column
        new_first_col = f"C:/Users/Mario/OneDrive/Desktop/gcn/graph_emotionv1/CARLA_Data/subject_screenshots/subject_{subject_num}/{row[0]}.png"
        
        # Write the new string and the second column to the output file
        f.write(f"{new_first_col} {row[1]}\n")

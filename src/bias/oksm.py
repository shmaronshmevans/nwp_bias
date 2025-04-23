import os
import pandas as pd
from datetime import timedelta

# Path to your directory of broken parquet files
path = "/home/aevans/nwp_bias/src/landtype/NY_cartopy/oksm_v3/"
files = os.listdir(path)

for file in files:
    print(f"Fixing: {file}")
    df = pd.read_parquet(f"{path}{file}").reset_index()

    # Ensure 'time' is datetime and extract the base date
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = df['time'].dt.date  # Just the date portion

    # Create corrected time by adding minutes from 'TIME' column
    df['time'] = pd.to_datetime(df['date'].astype(str)) + pd.to_timedelta(pd.to_numeric(df['TIME']), unit='m')

    # Clean up
    df = df.drop(columns=['date'])  # Optional
    df = df.set_index(['station', 'time']).sort_index()

    # Overwrite fixed file
    df.to_parquet(f"{path}{file}")
    print(f"{file} fixed.")

# directory = "/home/aevans/nwp_bias/src/landtype/NY_cartopy/"

# # Step 1: Find all .parquet files
# parquet_files = [
#     os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".parquet")
# ]


# def clean_parquet(path):
#     df_raw = pd.read_parquet(path)

#     # Row 1: timestamp parts (year, month, day, hour, etc.)
#     timestamp_row = df_raw.iloc[1].astype(str).tolist()

#     # Row 2: actual column names
#     col_names = df_raw.iloc[2].tolist()

#     # Data starts at row 3
#     df_data = df_raw.iloc[3:].copy()
#     df_data.columns = col_names
#     df_data = df_data.reset_index(drop=True)

#     # Add timestamp components from row 1 (assuming positions)
#     df_data["YEAR"] = int(timestamp_row[1])
#     df_data["MONTH"] = int(timestamp_row[2])
#     df_data["DAY"] = int(timestamp_row[3])
#     # Extract hour, minute, second from the 'TIME' column
    # Ensure 'time' is datetime and extract the base date
    # df['time'] = pd.to_datetime(df['time'])
    # df['date'] = df['time'].dt.date  # Just the date portion

    # # Create corrected time by adding minutes from 'TIME' column
    # df['time'] = pd.to_datetime(df['date'].astype(str)) + pd.to_timedelta(pd.to_numeric(df['TIME']), unit='m')

    # # Clean up
    # df = df.drop(columns=['date'])  # Optional

#     # Create 'time' column
#     df_data["time"] = pd.to_datetime(
#         df_data[["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"]], errors="coerce"
#     )

#     # Drop timestamp parts
#     df_data = df_data.drop(columns=["YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "SECOND"])

#     # Rename station ID
#     if "STID" in df_data.columns:
#         df_data = df_data.rename(columns={"STID": "station"})

#     return df_data


# # Step 2: Clean and combine all files
# df_list = [clean_parquet(f) for f in parquet_files]
# combined_df = pd.concat(df_list, ignore_index=True)

# # Step 3: Drop rows without valid 'station' or 'time'
# combined_df = combined_df.dropna(subset=["station", "time"])

# # Step 4: Set index
# combined_df = combined_df.set_index(["station", "time"]).sort_index()

# # Step 5: Save
# combined_df.to_parquet(os.path.join(directory, "combined_indexed_2024.parquet"))

# print("✅ Parquet files cleaned and combined with correct timestamp logic.")

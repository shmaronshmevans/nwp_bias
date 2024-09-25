import sys
import subprocess
import pandas as pd
import numpy as np


def run_script_recursively(
    batch_size,
    station,
    epochs,
    weight_decay,
    fh,
    nwp_model,
    climate_division_name,
    model_path,
):
    # Prepare the arguments to pass to the script
    command = [
        sys.executable,  # Python interpreter to call the script
        "/home/aevans/nwp_bias/src/machine_learning/src/seq2seq/switchboard.py",  # Path to your main script
        "--batch_size",
        str(batch_size),
        "--station",
        station,
        "--epochs",
        str(epochs),
        "--weight_decay",
        str(weight_decay),
        "--fh",
        str(fh),
        "--nwp_model",
        nwp_model,
        "--climate_division_name",
        climate_division_name,
        "--model_path",
        model_path,
    ]

    # Run the script with the arguments
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the output or handle errors
    print(f"Output for station {station} and forecast hour {fh}:")
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error occurred for station {station} and forecast hour {fh}:")
        print(result.stderr)


if __name__ == "__main__":
    c = "Mohawk Valley"

    nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
    df = nysm_clim[nysm_clim["climate_division_name"] == c]
    stations = df["stid"].unique()

    for f in np.arange(1, 19):
        for s in stations:
            run_script_recursively(
                batch_size=500,
                station=s,
                epochs=300,
                weight_decay=0.1,
                fh=f,
                nwp_model="HRRR",
                climate_division_name=c,
                model_path=f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/s2s/{c}_tp.pth",
            )

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F


class ZScoreNormalization:
    """Apply Z-score normalization to images."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, images: torch.Tensor):
        """Calculate mean and standard deviation for each image channel."""
        self.mean = images.mean(
            dim=(0, 1, 2), keepdim=True
        )  # Mean across batch, height, and width
        self.std = images.std(
            dim=(0, 1, 2), keepdim=True
        )  # Std across batch, height, and width

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Normalize the image using the precomputed mean and std."""
        image = torch.tensor(image, dtype=torch.float32)
        # Normalize by Z-score formula: (x - mean) / std
        if self.mean is not None and self.std is not None:
            image = (image - self.mean) / self.std
        return image


class SequenceDatasetMultiTask(Dataset):
    """Dataset class for multi-task learning with station-specific data."""

    def __init__(
        self,
        dataframe,
        target,
        features,
        nwp_features,
        sequence_length,
        forecast_steps,
        device,
        nwp_model,
        metvar,
        image_list_cols,
        dataframe_ls,  # Ensure this is passed
        transform=ZScoreNormalization(),
    ):
        self.nysm_df = dataframe  # Assign the correct dataframe
        self.nwp_dataframe_ls = dataframe_ls  # Assign the list of NWP dataframes
        self.image_df = image_list_cols
        self.features = features
        self.nwp_features = nwp_features
        self.target = target
        self.sequence_length = sequence_length
        self.forecast_steps = forecast_steps
        self.device = device
        self.nwp_model = nwp_model
        self.metvar = metvar
        self.transform = transform
        self.X = torch.tensor(dataframe[features].values, dtype=torch.float32).to(
            device
        )
        self.P_ls = dataframe[image_list_cols].values.tolist()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.nwp_model == "HRRR":
            ## Construct the input values
            x_start = i
            x_end = i + self.sequence_length
            x = self.X[x_start:x_end, :]

            y_start = i + self.sequence_length
            y_end = y_start + self.forecast_steps

            # Ensure `y_start + forecast_steps` does not exceed length
            times_y = self.nysm_df["valid_time"].iloc[y_start:y_end]

            # Initialize y tensor
            y = torch.zeros(
                (self.forecast_steps, x.shape[1]),
                dtype=torch.float32,
                device=self.device,
            )

            for n, t in enumerate(times_y):
                match = self.nwp_dataframe_ls[n].loc[
                    self.nwp_dataframe_ls[n]["valid_time"] == t
                ]
                if match.empty:
                    # PAD ROW
                    nwp_ = pd.DataFrame(
                        np.zeros((1, len(self.nwp_features))), columns=self.nwp_features
                    )
                    target_values = pd.DataFrame(
                        np.zeros((1, len(self.target))), columns=self.target
                    )
                else:
                    # finish x values
                    nwp_ = self.nwp_dataframe_ls[n].iloc[int(y_start + n), :]
                    # Ensure tensor conversion and dimension match
                    nwp_values = torch.tensor(
                        nwp_[self.nwp_features].values, dtype=torch.float32
                    ).to(self.device)
                    x = torch.cat([x, nwp_values], dim=1)

                    # y values
                    nwp_t = self.nwp_dataframe_ls[n].iloc[int(y_start + n), :]
                    target_values = (
                        torch.tensor(nwp_t[self.target].values, dtype=torch.float32)
                        .to(self.device)
                        .squeeze(0)
                    )
                    target_values = target_values.unsqueeze(1)
                    # Fix indexing issue for y assignment
                    y[n - 1, :] = target_values

            ##### Padding (if necessary)
            if x.shape[0] < (self.sequence_length + self.forecast_steps):
                _x = torch.zeros(
                    (
                        (self.sequence_length + self.forecast_steps) - x.shape[0],
                        x.shape[1],
                    ),
                    device=self.device,
                )
                x = torch.cat((x, _x), 0)

            if y.shape[0] < self.forecast_steps:
                _y = torch.zeros(
                    (self.forecast_steps - y.shape[0], y.shape[1]), device=self.device
                )
                y = torch.cat((y, _y), dim=0)

        # Check if the selected NWP model is GFS
        if self.nwp_model == "GFS":
            # Define start and end indices for input sequence `x`
            x_start = i
            x_end = i + self.sequence_length
            x = self.X[x_start:x_end, :]  # Extract sequence from `self.X`

            # Define start and end indices for target sequence `y`
            y_start = (i + self.sequence_length) + 1
            y_end = y_start + int(self.forecast_steps / 3)

            # Extract corresponding timestamps for `y` values
            times_y = self.nysm_df["valid_time"].iloc[y_start:y_end]

            # Initialize `y` tensor with zeros to store target values
            y = torch.zeros(
                (int(self.forecast_steps / 3), x.shape[1]),
                dtype=torch.float32,
                device=self.device,
            )

            # Iterate over target time indices
            for n, t in enumerate(times_y):
                # Find corresponding row in NWP dataframe for time `t`
                match = self.nwp_dataframe_ls[n].loc[
                    self.nwp_dataframe_ls[n]["valid_time"] == t
                ]

                # If no match is found, use a zero-filled placeholder tensor
                if match.empty:
                    nwp_values = torch.zeros(
                        (1, self.X.shape[1]),  # Shape matches feature dimensions
                        device=self.device,
                    )
                else:
                    try:
                        # Extract NWP values for the given target time and convert to tensor
                        nwp_values = (
                            torch.tensor(
                                self.nwp_dataframe_ls[n]
                                .iloc[int(y_start + n)][self.nwp_features]
                                .to_numpy(dtype=np.float32),
                                dtype=torch.float32,
                            )
                            .to(self.device)
                            .unsqueeze(0)  # Add batch dimension
                        )
                    except:
                        break  # If an error occurs, exit the loop

                    # Append the extracted NWP values to `x`
                    x = torch.vstack([x, nwp_values])

                    # Extract and process target values
                    try:
                        target_values = torch.tensor(
                            self.nwp_dataframe_ls[n].iloc[int(y_start + n)][
                                self.target
                            ],
                            dtype=torch.float32,
                        ).to(self.device)
                        target_values = target_values.unsqueeze(
                            0
                        )  # Add batch dimension
                    except:
                        continue  # Skip to the next iteration if target extraction fails

                    # Store target values in `y` at the corresponding index
                    y[n - 1, :] = target_values

            # Ensure `x` has the required length by padding with zeros if necessary
            if x.shape[0] < (self.sequence_length + int(self.forecast_steps / 3)):
                _x = torch.zeros(
                    (
                        (self.sequence_length + int(self.forecast_steps / 3))
                        - x.shape[0],
                        self.X.shape[1],
                    ),
                    device=self.device,
                )
                x = torch.cat((x, _x), 0)

            # Ensure `y` has the required length by padding with zeros if necessary
            if y.shape[0] < int(self.forecast_steps / 3):
                _y = torch.zeros(
                    (int(self.forecast_steps / 3) - y.shape[0], 1), device=self.device
                )
                y = torch.cat((y, _y), 0)

        if self.nwp_model == "NAM":
            x_start = i
            x_end = i + self.sequence_length
            y_start = i + self.sequence_length
            y_end = y_start + self.forecast_steps
            x = self.X[x_start:x_end, :]

            # Ensure `y_start + forecast_steps` does not exceed length
            times_y = self.nysm_df["valid_time"].iloc[y_start:y_end]
            y = torch.zeros(
                (int(self.forecast_steps / 3), x.shape[1]),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(1)

            for n, t in enumerate(times_y):
                match = self.nwp_dataframe_ls[n].loc[
                    self.nwp_dataframe_ls[n]["valid_time"] == t
                ]
                if match.empty:
                    # PAD ROW
                    nwp_ = pd.DataFrame(
                        np.zeros((1, len(self.nwp_features))), columns=self.nwp_features
                    )
                    target_values = pd.DataFrame(
                        np.zeros((1, len(self.target))), columns=self.target
                    )
                else:
                    # finish x values
                    nwp_ = self.nwp_dataframe_ls[n].iloc[int(y_start + n), :]
                    # Ensure tensor conversion and dimension match
                    nwp_values = torch.tensor(
                        nwp_[self.nwp_features].values, dtype=torch.float32
                    ).to(self.device)
                    x = torch.cat([x, nwp_values], dim=1)

                    # y values
                    nwp_t = self.nwp_dataframe_ls[n].iloc[int(y_start + n), :]
                    target_values = torch.tensor(
                        nwp_t[self.target].values, dtype=torch.float32
                    ).to(self.device)
                    target_values = target_values.unsqueeze(1)
                    # Fix indexing issue for y assignment
                    y[n - 1, :, :] = target_values

            if x.shape[0] < (self.sequence_length + self.forecast_steps):
                _x = torch.zeros(
                    (
                        (self.sequence_length + self.forecast_steps) - x.shape[0],
                        self.X.shape[1],
                    ),
                    device=self.device,
                )
                x = torch.cat((x, _x), 0)

            if y.shape[0] < self.forecast_steps:
                _y = torch.zeros(
                    (self.forecast_steps - y.shape[0], 1),
                    device=self.device,
                )
                y = torch.cat((y, _y), 0)

        idx = min(i + self.sequence_length, len(self.P_ls) - 1)
        img_name = self.P_ls[idx]  # This avoids an out-of-range error
        images = []
        for img in img_name:
            # Load the image
            image = np.load(img).astype(np.float32)

            # Apply transform if available
            if self.transform:
                image = self.transform(image)

            # Convert to tensor and move to device
            images.append(image.clone().detach().to(torch.float32).to(self.device))

        # Stack images into a single tensor
        images = torch.stack(images)

        # Expected shape
        expected_shape = (1, 121, 6, 11)

        # Compute padding values (ensure non-negative values)
        padding = [
            max(0, expected_shape[3] - images.shape[3]),  # Pad width (last dimension)
            max(0, expected_shape[2] - images.shape[2]),  # Pad height (third dimension)
            max(0, expected_shape[1] - images.shape[1]),  # Pad depth
            max(0, expected_shape[0] - images.shape[0]),
        ]  # Pad batch/channel

        # Apply padding if necessary
        if any(p > 0 for p in padding):
            images = F.pad(
                images, (0, padding[0], 0, padding[1], 0, padding[2], 0, padding[3])
            )

        return x, images, y
        # return x, y

#!/usr/bin/env python
import torch
from torch.utils.data import Dataset


class Energy(Dataset):
    """

    Examples
    --------
    >>> energy = Energy()
    >>> energy[200]
    """
    def __init__(self, lag=20):
        super(Energy).__init__()
        x = pd.read_csv("data/train.csv")

        # get timestamps
        time_info = x[["Month", "DayOfTheMonth", "Hour", "Minute"]]
        time_info.insert(0, "year", 2012)
        time_info.columns = ["year", "month", "day", "hour", "minute"]

        # standardize the covariate columns
        z = (x - x.mean()) / x.std()
        z["Forecast"] = x["Forecast"]
        z["time"] = pd.to_datetime(time_info)
        z = z.sort_values("time")

        self.z = z

    def __len__(self):
        return len(self.z)

    def __getitem__(self, ix):
        # use push-back imputation if request an early timepoint
        if ix < lag:
            df_buffer = self.z.iloc[list(np.zeros(lag - ix))]
            cur = df_buffer.append(self.z[:ix])
        else:
            cur = self.z[(ix - lag):ix]

        # feature that's the difference between current and past times
        delta = cur.iloc[-1]["time"] - past["time"]
        cur.insert(0, "delta", [v.total_seconds() / 1e6 for v in delta])

        # split x and y
        y = cur["Forecast"].values
        x = cur.drop(["time", "Forecast"], axis=1).values
        return x, y

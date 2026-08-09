import numpy as np
import pandas as pd
import holidays


# ============================================================
# Configuration
# ============================================================

# City C / Sapporo dataset:
# d = 0 corresponds to 2019-09-15
ORIGIN_DATE = pd.Timestamp("2019-09-15")


# ============================================================
# Calendar features
# ============================================================

def add_calendar_features(
    df: pd.DataFrame,
    origin_date: str | pd.Timestamp = ORIGIN_DATE,
    country: str = "JP",
    events: list[dict] | None = None,
) -> pd.DataFrame:
    """
    Add calendar-related features to the mobility dataframe.

    Required column
    ---------------
    d : int
        Day index.

    Added columns
    -------------
    date
    month
    season
    is_holiday
    event_code
    is_event

    Season encoding
    ---------------
    0 = Winter
    1 = Spring
    2 = Summer
    3 = Autumn

    Event format
    ------------
    events = [
        {
            "name": "rugbyworldcup",
            "start": "2019-09-20",
            "end": "2019-11-02",
        }
    ]
    """

    df = df.copy()
    origin_date = pd.Timestamp(origin_date)

    # --------------------------------------------------------
    # Day index -> actual date
    # --------------------------------------------------------

    df["date"] = (
        origin_date
        + pd.to_timedelta(df["d"].astype(int), unit="D")
    )

    df["month"] = df["date"].dt.month.astype(np.int16)

    # --------------------------------------------------------
    # Season
    # --------------------------------------------------------

    m = df["month"].to_numpy()

    season = np.zeros(len(df), dtype=np.int16)

    # Spring: March-May
    season[(m >= 3) & (m <= 5)] = 1

    # Summer: June-August
    season[(m >= 6) & (m <= 8)] = 2

    # Autumn: September-November
    season[(m >= 9) & (m <= 11)] = 3

    # Winter remains 0
    df["season"] = season

    # --------------------------------------------------------
    # Japanese national holidays
    # --------------------------------------------------------

    country_holidays = holidays.country_holidays(country)

    df["is_holiday"] = (
        df["date"]
        .dt.date
        .map(lambda x: x in country_holidays)
        .astype(np.int8)
    )

    # --------------------------------------------------------
    # Event features
    # --------------------------------------------------------

    df["event_code"] = 0

    if events:
        for i, event in enumerate(events, start=1):

            start = pd.Timestamp(event["start"])
            end = pd.Timestamp(event["end"])

            mask = (
                (df["date"] >= start)
                & (df["date"] <= end)
            )

            df.loc[mask, "event_code"] = i

    df["is_event"] = (
        df["event_code"] > 0
    ).astype(np.int8)

    return df


# ============================================================
# Prior construction
# ============================================================

def build_priors(
    train_df: pd.DataFrame,
    keys: list[str],
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Build frequency and historical-count priors.

    Parameters
    ----------
    train_df:
        Training dataframe containing at least the grouping
        keys and the `count` column.

    keys:
        Columns used to define the prior.

    alpha:
        Smoothing parameter for the non-zero frequency prior.

    Returns
    -------
    DataFrame containing:

        keys + ["freq_nz", "base_log"]

    freq_nz
        Smoothed probability that count > 0.

    base_log
        Mean log1p(count) for the corresponding group.
    """

    grouped = train_df.groupby(keys, as_index=False)

    agg = grouped["count"].agg(
        n="size",
        nz=lambda s: int(
            (s.to_numpy() > 0).sum()
        ),
        base_log=lambda s: float(
            np.log1p(s.to_numpy()).mean()
        ),
    )

    # Laplace-style smoothing
    agg["freq_nz"] = (
        (agg["nz"] + alpha)
        / (agg["n"] + 2 * alpha)
    )

    return agg[
        keys + ["freq_nz", "base_log"]
    ]


# ============================================================
# Hierarchical prior backoff
# ============================================================

def attach_priors_with_backoff(
    df: pd.DataFrame,
    priA: pd.DataFrame,
    keysA: list[str],
    priB: pd.DataFrame,
    keysB: list[str],
    priC: pd.DataFrame,
    keysC: list[str],
) -> pd.DataFrame:
    """
    Attach hierarchical priors using:

        A -> B -> C -> default

    More specific priors are preferred when available.
    Missing values fall back to progressively more general
    priors.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Most specific prior: A
    # --------------------------------------------------------

    df = df.merge(
        priA,
        on=keysA,
        how="left",
    )

    # --------------------------------------------------------
    # Medium prior: B
    # --------------------------------------------------------

    df = df.merge(
        priB,
        on=keysB,
        how="left",
        suffixes=("", "_B"),
    )

    # --------------------------------------------------------
    # General prior: C
    # --------------------------------------------------------

    df = df.merge(
        priC,
        on=keysC,
        how="left",
        suffixes=("", "_C"),
    )

    # --------------------------------------------------------
    # base_log backoff
    #
    # A -> B -> C -> 0
    # --------------------------------------------------------

    df["base_log"] = (
        df["base_log"]
        .combine_first(df["base_log_B"])
        .combine_first(df["base_log_C"])
        .fillna(0.0)
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # freq_nz backoff
    #
    # A -> B -> C -> 0
    # --------------------------------------------------------

    df["freq_nz"] = (
        df["freq_nz"]
        .combine_first(df["freq_nz_B"])
        .combine_first(df["freq_nz_C"])
        .fillna(0.0)
        .astype(np.float32)
    )

    # Remove helper columns generated by the merges
    drop_cols = [
        column
        for column in df.columns
        if column.endswith("_B")
        or column.endswith("_C")
    ]

    df.drop(
        columns=drop_cols,
        inplace=True,
    )

    return df


# ============================================================
# Historical baseline
# ============================================================

def fit_baseline_hist(
    train_df: pd.DataFrame,
):
    """
    Fit the historical mean baseline.

    Baseline grouping:

        (weekday, t, x, y)

    For every spatial grid and time slot, the baseline is the
    historical average count observed on the same weekday.
    """

    hist = train_df.groupby(
        ["weekday", "t", "x", "y"]
    )["count"].mean()

    return hist


def predict_baseline_hist(
    df_split: pd.DataFrame,
    hist_series,
) -> np.ndarray:
    """
    Generate predictions from the historical mean baseline.

    If a (weekday, t, x, y) combination was not observed in
    training, the global historical mean is used instead.
    """

    keys = list(
        zip(
            df_split["weekday"],
            df_split["t"],
            df_split["x"],
            df_split["y"],
        )
    )

    global_mean = float(
        hist_series.mean()
    )

    predictions = np.array(
        [
            hist_series.get(
                key,
                global_mean,
            )
            for key in keys
        ],
        dtype=float,
    )

    return predictions
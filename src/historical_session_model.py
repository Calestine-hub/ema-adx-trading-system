"""
EMA-ADX TRADING SYSTEM - V3.0
B3.2 SESSION BOUNDARY EVIDENCE ENGINE

Purpose
-------
Inspect the ACTUAL M1 candles around recurring gap patterns
identified by B3.1.

B3.2 is diagnostic only.

It does NOT:
    - modify raw M1 data
    - fill missing prices
    - remove candles
    - modify session_validator.py
    - declare final broker session rules
    - generate trading signals

B3.2 asks:

    "When a recurring gap occurs, what do the actual M1
     candles immediately before and after that gap look like?"

IMPORTANT
---------
B3.1 family files and gap files do not necessarily contain a
"boundary_family" column in the gap table.

Therefore this version reconstructs family membership from
the actual family time pattern.
"""


import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOL_FILES = {
    "XAUUSDm": "XAUUSDm_M1_202108170000_202608131827.csv",
    "GBPUSDm": "GBPUSDm_M1_202108170000_202608131847.csv",
    "USDCHFm": "USDCHFm_M1_202108170000_202608131849.csv",
}

DATA_DIR = Path("data/raw")

B31_DIR = Path("data/session_model_b31")

OUTPUT_DIR = Path(
    "data/session_boundary_evidence_b32"
)

CONTEXT_MINUTES = 10

MIN_FAMILY_OCCURRENCES = 20

MAX_EXAMPLES_PER_FAMILY = 12


# ============================================================
# FILE LOADER
# ============================================================

def detect_separator(filepath):

    with open(
        filepath,
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as f:

        first_line = f.readline()

    if "\t" in first_line:
        return "\t", "TAB"

    if "," in first_line:
        return ",", "COMMA"

    if ";" in first_line:
        return ";", "SEMICOLON"

    raise ValueError(
        f"Could not detect separator: {filepath}"
    )


def load_m1(filepath):

    filepath = Path(filepath)

    if not filepath.exists():

        raise FileNotFoundError(
            f"File does not exist:\n{filepath}"
        )

    separator, separator_name = (
        detect_separator(filepath)
    )

    print(
        f"\nLoading M1: {filepath.name}"
    )

    print(
        f"Separator: {separator_name}"
    )

    df = pd.read_csv(
        filepath,
        sep=separator,
        encoding="utf-8-sig",
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    required = [
        "<DATE>",
        "<TIME>",
        "<OPEN>",
        "<HIGH>",
        "<LOW>",
        "<CLOSE>",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    df["Datetime"] = pd.to_datetime(
        df["<DATE>"].astype(str).str.strip()
        + " "
        + df["<TIME>"].astype(str).str.strip(),
        format="%Y.%m.%d %H:%M:%S",
        errors="coerce",
    )

    if df["Datetime"].isna().any():

        raise ValueError(
            "Invalid Datetime values detected."
        )

    df = (
        df.sort_values("Datetime")
        .drop_duplicates(
            subset=["Datetime"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        f"Loaded {len(df):,} M1 candles."
    )

    print(
        f"First: {df['Datetime'].iloc[0]}"
    )

    print(
        f"Last:  {df['Datetime'].iloc[-1]}"
    )

    return df


# ============================================================
# FIND B3.1 FILES
# ============================================================

def find_b31_file(
    symbol,
    preferred_names,
):

    symbol_dir = (
        B31_DIR / symbol
    )

    if not symbol_dir.exists():

        raise FileNotFoundError(
            f"B3.1 directory not found:\n"
            f"{symbol_dir}"
        )

    for name in preferred_names:

        candidate = (
            symbol_dir / name
        )

        if candidate.exists():

            return candidate

    # --------------------------------------------------------
    # Fallback: inspect available CSV files
    # --------------------------------------------------------

    csv_files = list(
        symbol_dir.glob("*.csv")
    )

    if not csv_files:

        raise FileNotFoundError(
            f"No CSV files found in:\n"
            f"{symbol_dir}"
        )

    return csv_files


# ============================================================
# LOAD B3.1 FAMILY FILE
# ============================================================

def load_family_file(symbol):

    result = find_b31_file(
        symbol,
        [
            "recurring_boundary_families.csv",
            "boundary_families.csv",
            "recurring_candidates.csv",
            "families.csv",
        ],
    )

    if isinstance(result, list):

        print(
            "\nAvailable B3.1 CSV files:"
        )

        for file in result:

            print(
                f"    {file.name}"
            )

        # Prefer files whose names indicate
        # recurring/family information.
        candidates = [
            f
            for f in result
            if (
                "family" in f.name.lower()
                or
                "candidate" in f.name.lower()
            )
        ]

        if not candidates:

            raise FileNotFoundError(
                "Could not identify B3.1 family CSV."
            )

        filepath = candidates[0]

    else:

        filepath = result

    print(
        f"\nLoading B3.1 family file:"
    )

    print(
        f"    {filepath}"
    )

    families = pd.read_csv(
        filepath
    )

    print(
        "\nB3.1 family columns:"
    )

    print(
        families.columns.tolist()
    )

    return families


# ============================================================
# NORMALIZE FAMILY TABLE
# ============================================================

def normalize_family_table(
    families,
):

    df = families.copy()

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Locate family identifier
    # --------------------------------------------------------

    family_candidates = [
        "Family",
        "family",
        "boundary_family",
        "family_id",
        "Family_ID",
    ]

    family_column = None

    for column in family_candidates:

        if column in df.columns:

            family_column = column
            break

    # If no explicit family ID exists, create one.
    if family_column is None:

        df["family_id"] = [
            f"F{i + 1:03d}"
            for i in range(len(df))
        ]

        family_column = "family_id"

    df["family_id"] = (
        df[family_column]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Locate previous/current time fields
    # --------------------------------------------------------

    previous_candidates = [
        "Prev",
        "Previous",
        "previous",
        "previous_time",
        "median_previous_time",
    ]

    current_candidates = [
        "Current",
        "current",
        "current_time",
        "median_current_time",
    ]

    previous_column = None
    current_column = None

    for column in previous_candidates:

        if column in df.columns:

            previous_column = column
            break

    for column in current_candidates:

        if column in df.columns:

            current_column = column
            break

    # --------------------------------------------------------
    # Some B3.1 versions may store minute-of-day values
    # --------------------------------------------------------

    if previous_column is None:

        numeric_candidates = [
            "median_previous_minute",
            "previous_minute",
        ]

        for column in numeric_candidates:

            if column in df.columns:

                previous_column = column
                break

    if current_column is None:

        numeric_candidates = [
            "median_current_minute",
            "current_minute",
        ]

        for column in numeric_candidates:

            if column in df.columns:

                current_column = column
                break

    if (
        previous_column is None
        or
        current_column is None
    ):

        raise ValueError(
            "\nCould not identify the B3.1 "
            "previous/current time columns.\n\n"
            f"Available columns:\n"
            f"{df.columns.tolist()}"
        )

    df["_previous_raw"] = (
        df[previous_column]
    )

    df["_current_raw"] = (
        df[current_column]
    )

    # --------------------------------------------------------
    # Convert time representation
    # --------------------------------------------------------

    df["previous_minute"] = (
        df["_previous_raw"]
        .apply(parse_time_to_minutes)
    )

    df["current_minute"] = (
        df["_current_raw"]
        .apply(parse_time_to_minutes)
    )

    # --------------------------------------------------------
    # Occurrence count
    # --------------------------------------------------------

    occurrence_candidates = [
        "Occ",
        "occurrences",
        "Occurrences",
        "occurrence_count",
    ]

    occurrence_column = None

    for column in occurrence_candidates:

        if column in df.columns:

            occurrence_column = column
            break

    if occurrence_column:

        df["occurrences"] = pd.to_numeric(
            df[occurrence_column],
            errors="coerce",
        )

    else:

        df["occurrences"] = 0

    # --------------------------------------------------------
    # Optional fields
    # --------------------------------------------------------

    for target, candidates in {

        "unique_dates": [
            "Dates",
            "dates",
            "unique_dates",
        ],

        "unique_years": [
            "Years",
            "years",
            "unique_years",
        ],

        "boundary_strength": [
            "Score",
            "score",
            "boundary_strength",
        ],

        "persistence_score": [
            "Score",
            "score",
            "persistence_score",
        ],

    }.items():

        found = None

        for column in candidates:

            if column in df.columns:

                found = column
                break

        if found:

            df[target] = pd.to_numeric(
                df[found],
                errors="coerce",
            )

        else:

            df[target] = pd.NA

    return df


# ============================================================
# TIME PARSER
# ============================================================

def parse_time_to_minutes(value):

    if pd.isna(value):

        return None

    # Numeric minute-of-day
    if isinstance(
        value,
        (int, float),
    ):

        if 0 <= float(value) < 1440:

            return int(
                round(float(value))
            )

    text = str(value).strip()

    # HH:MM
    if ":" in text:

        parts = text.split(":")

        try:

            hour = int(parts[0])
            minute = int(parts[1])

            return (
                hour * 60
                + minute
            )

        except ValueError:

            pass

    # Try numeric string
    try:

        numeric = float(text)

        if 0 <= numeric < 1440:

            return int(
                round(numeric)
            )

    except ValueError:

        pass

    return None


# ============================================================
# LOAD GAP FILE
# ============================================================

def load_gap_file(symbol):

    result = find_b31_file(
        symbol,
        [
            "all_gaps.csv",
            "gap_table.csv",
            "gaps.csv",
            "timestamp_gaps.csv",
        ],
    )

    if isinstance(result, list):

        candidates = [
            f
            for f in result
            if "gap" in f.name.lower()
        ]

        if not candidates:

            raise FileNotFoundError(
                "Could not identify B3.1 gap CSV."
            )

        filepath = candidates[0]

    else:

        filepath = result

    print(
        f"\nLoading B3.1 gap file:"
    )

    print(
        f"    {filepath}"
    )

    gaps = pd.read_csv(
        filepath
    )

    print(
        "\nB3.1 gap columns:"
    )

    print(
        gaps.columns.tolist()
    )

    # --------------------------------------------------------
    # Normalize previous/current columns
    # --------------------------------------------------------

    previous_column = None
    current_column = None

    for column in [
        "previous",
        "Previous",
        "previous_time",
    ]:

        if column in gaps.columns:

            previous_column = column
            break

    for column in [
        "current",
        "Current",
        "current_time",
    ]:

        if column in gaps.columns:

            current_column = column
            break

    if (
        previous_column is None
        or
        current_column is None
    ):

        raise ValueError(
            "\nCould not identify previous/current "
            "timestamp columns in B3.1 gap file.\n\n"
            f"Available columns:\n"
            f"{gaps.columns.tolist()}"
        )

    gaps["previous"] = pd.to_datetime(
        gaps[previous_column]
    )

    gaps["current"] = pd.to_datetime(
        gaps[current_column]
    )

    # Calculate gap directly rather than trusting
    # a saved B3.1 field.
    gaps["gap_minutes"] = (
        (
            gaps["current"]
            - gaps["previous"]
        )
        .dt.total_seconds()
        / 60
    )

    gaps["previous_minute"] = (
        gaps["previous"]
        .dt.hour * 60
        + gaps["previous"].dt.minute
    )

    gaps["current_minute"] = (
        gaps["current"]
        .dt.hour * 60
        + gaps["current"].dt.minute
    )

    return gaps


# ============================================================
# FAMILY MEMBERSHIP
# ============================================================

def match_family_occurrences(
    gaps,
    family,
):

    previous_minute = (
        family["previous_minute"]
    )

    current_minute = (
        family["current_minute"]
    )

    if pd.isna(
        previous_minute
    ) or pd.isna(
        current_minute
    ):

        return pd.DataFrame()

    matched = gaps[
        (
            gaps["previous_minute"]
            == int(previous_minute)
        )
        &
        (
            gaps["current_minute"]
            == int(current_minute)
        )
    ].copy()

    return matched


# ============================================================
# M1 CONTEXT
# ============================================================

def inspect_m1_context(
    m1,
    previous_time,
    current_time,
):

    start = (
        previous_time
        - pd.Timedelta(
            minutes=CONTEXT_MINUTES
        )
    )

    end = (
        current_time
        + pd.Timedelta(
            minutes=CONTEXT_MINUTES
        )
    )

    context = m1[
        (
            m1["Datetime"]
            >= start
        )
        &
        (
            m1["Datetime"]
            <= end
        )
    ]

    before = context[
        context["Datetime"]
        <= previous_time
    ]

    after = context[
        context["Datetime"]
        >= current_time
    ]

    expected_context = (
        CONTEXT_MINUTES + 1
    )

    before_count = len(
        before
    )

    after_count = len(
        after
    )

    return {
        "before_count":
            before_count,

        "after_count":
            after_count,

        "before_complete":
            before_count
            >= expected_context,

        "after_complete":
            after_count
            >= expected_context,
    }


# ============================================================
# FAMILY EVIDENCE
# ============================================================

def analyze_family(
    m1,
    gaps,
    family,
):

    occurrences = (
        match_family_occurrences(
            gaps,
            family,
        )
    )

    if occurrences.empty:

        return None, pd.DataFrame()

    before_counts = []
    after_counts = []

    durations = []

    details = []

    for index, row in (
        occurrences.iterrows()
    ):

        context = inspect_m1_context(
            m1,
            row["previous"],
            row["current"],
        )

        before_counts.append(
            context[
                "before_count"
            ]
        )

        after_counts.append(
            context[
                "after_count"
            ]
        )

        durations.append(
            row["gap_minutes"]
        )

        if index < MAX_EXAMPLES_PER_FAMILY:

            details.append(
                {
                    "family_id":
                        family["family_id"],

                    "previous":
                        row["previous"],

                    "current":
                        row["current"],

                    "gap_minutes":
                        row["gap_minutes"],

                    "before_count":
                        context[
                            "before_count"
                        ],

                    "after_count":
                        context[
                            "after_count"
                        ],

                    "before_complete":
                        context[
                            "before_complete"
                        ],

                    "after_complete":
                        context[
                            "after_complete"
                        ],
                }
            )

    occurrence_count = len(
        occurrences
    )

    before_availability = (
        sum(
            count >= CONTEXT_MINUTES
            for count in before_counts
        )
        / occurrence_count
    )

    after_availability = (
        sum(
            count >= CONTEXT_MINUTES
            for count in after_counts
        )
        / occurrence_count
    )

    median_gap = (
        pd.Series(
            durations
        ).median()
    )

    gap_std = (
        pd.Series(
            durations
        ).std()
    )

    # --------------------------------------------------------
    # Conservative classification
    # --------------------------------------------------------

    if (
        median_gap <= 5
        and
        gap_std == 0
        and
        before_availability >= 0.80
        and
        after_availability >= 0.80
    ):

        classification = (
            "SHORT_RECURRING_GAP"
        )

    elif (
        gap_std <= 2
        and
        before_availability >= 0.80
        and
        after_availability >= 0.80
    ):

        classification = (
            "CONSISTENT_BOUNDARY_CANDIDATE"
        )

    else:

        classification = (
            "VARIABLE_BOUNDARY"
        )

    summary = {
        "family_id":
            family["family_id"],

        "occurrences":
            occurrence_count,

        "b31_occurrences":
            family["occurrences"],

        "unique_dates":
            family["unique_dates"],

        "unique_years":
            family["unique_years"],

        "previous_time":
            format_minutes(
                family[
                    "previous_minute"
                ]
            ),

        "current_time":
            format_minutes(
                family[
                    "current_minute"
                ]
            ),

        "median_gap_minutes":
            median_gap,

        "gap_std":
            gap_std,

        "before_context_availability":
            before_availability,

        "after_context_availability":
            after_availability,

        "boundary_strength":
            family[
                "boundary_strength"
            ],

        "classification":
            classification,
    }

    return (
        summary,
        pd.DataFrame(details),
    )


# ============================================================
# FORMAT TIME
# ============================================================

def format_minutes(value):

    if pd.isna(value):

        return ""

    value = int(value) % 1440

    return (
        f"{value // 60:02d}:"
        f"{value % 60:02d}"
    )


# ============================================================
# PROCESS SYMBOL
# ============================================================

def process_symbol(
    symbol,
):

    print("\n")
    print("#" * 110)

    print(
        f"B3.2 SESSION BOUNDARY EVIDENCE — "
        f"{symbol}"
    )

    print("#" * 110)

    # --------------------------------------------------------
    # Load raw M1
    # --------------------------------------------------------

    m1 = load_m1(
        DATA_DIR
        / SYMBOL_FILES[symbol]
    )

    # --------------------------------------------------------
    # Load B3.1 family definitions
    # --------------------------------------------------------

    families_raw = (
        load_family_file(
            symbol
        )
    )

    families = (
        normalize_family_table(
            families_raw
        )
    )

    # --------------------------------------------------------
    # Filter meaningful families
    # --------------------------------------------------------

    families = families[
        families["occurrences"]
        >= MIN_FAMILY_OCCURRENCES
    ].copy()

    print(
        f"\nFamilies selected for evidence: "
        f"{len(families):,}"
    )

    # --------------------------------------------------------
    # Load gaps
    # --------------------------------------------------------

    gaps = load_gap_file(
        symbol
    )

    print(
        f"\nGap records available: "
        f"{len(gaps):,}"
    )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    summaries = []
    all_details = []

    print(
        "\nInspecting actual M1 candles..."
    )

    for index, family in (
        families.iterrows()
    ):

        summary, details = (
            analyze_family(
                m1,
                gaps,
                family,
            )
        )

        if summary is not None:

            summaries.append(
                summary
            )

        if not details.empty:

            all_details.append(
                details
            )

        if (
            (index + 1) % 25 == 0
        ):

            print(
                f"    Processed "
                f"{index + 1:,}/"
                f"{len(families):,} families"
            )

    evidence = pd.DataFrame(
        summaries
    )

    if all_details:

        examples = pd.concat(
            all_details,
            ignore_index=True,
        )

    else:

        examples = pd.DataFrame()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_DIR
        / symbol
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence_file = (
        output_dir
        / "boundary_evidence.csv"
    )

    examples_file = (
        output_dir
        / "boundary_examples.csv"
    )

    evidence.to_csv(
        evidence_file,
        index=False,
    )

    examples.to_csv(
        examples_file,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 110)

    print(
        f"B3.2 EVIDENCE SUMMARY — {symbol}"
    )

    print("=" * 110)

    if evidence.empty:

        print(
            "\nNo evidence records generated."
        )

        return

    counts = (
        evidence[
            "classification"
        ]
        .value_counts()
    )

    for classification, count in (
        counts.items()
    ):

        print(
            f"{classification:<35}"
            f"{count:>8,}"
        )

    print("-" * 110)

    print(
        f"Families inspected: "
        f"{len(evidence):,}"
    )

    # --------------------------------------------------------
    # Strong candidates
    # --------------------------------------------------------

    print("\n")
    print(
        "CONSISTENT BOUNDARY CANDIDATES"
    )

    print("-" * 110)

    strong = evidence[
        evidence[
            "classification"
        ]
        == "CONSISTENT_BOUNDARY_CANDIDATE"
    ]

    if strong.empty:

        print(
            "None."
        )

    else:

        for _, row in (
            strong
            .sort_values(
                "occurrences",
                ascending=False,
            )
            .head(25)
            .iterrows()
        ):

            print(
                f"{row['family_id']:<18}"
                f"{row['occurrences']:>7,} occurrences | "
                f"{row['previous_time']} -> "
                f"{row['current_time']} | "
                f"gap "
                f"{row['median_gap_minutes']:.0f} min | "
                f"std "
                f"{row['gap_std']:.2f} | "
                f"before "
                f"{row['before_context_availability']:.1%} | "
                f"after "
                f"{row['after_context_availability']:.1%}"
            )

    print("\n")
    print(
        "SHORT RECURRING GAPS"
    )

    print("-" * 110)

    short = evidence[
        evidence[
            "classification"
        ]
        == "SHORT_RECURRING_GAP"
    ]

    if short.empty:

        print(
            "None."
        )

    else:

        for _, row in (
            short
            .sort_values(
                "occurrences",
                ascending=False,
            )
            .head(25)
            .iterrows()
        ):

            print(
                f"{row['family_id']:<18}"
                f"{row['occurrences']:>7,} occurrences | "
                f"{row['previous_time']} -> "
                f"{row['current_time']} | "
                f"gap "
                f"{row['median_gap_minutes']:.0f} min"
            )

    print("\n")
    print(
        "VARIABLE BOUNDARIES"
    )

    print("-" * 110)

    variable = evidence[
        evidence[
            "classification"
        ]
        == "VARIABLE_BOUNDARY"
    ]

    if variable.empty:

        print(
            "None."
        )

    else:

        for _, row in (
            variable
            .sort_values(
                "occurrences",
                ascending=False,
            )
            .head(25)
            .iterrows()
        ):

            print(
                f"{row['family_id']:<18}"
                f"{row['occurrences']:>7,} occurrences | "
                f"{row['previous_time']} -> "
                f"{row['current_time']} | "
                f"median gap "
                f"{row['median_gap_minutes']:.0f} min | "
                f"std "
                f"{row['gap_std']:.2f}"
            )

    print("\n")
    print(
        "Saved:"
    )

    print(
        f"    {evidence_file}"
    )

    print(
        f"    {examples_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 110)

    print(
        "       EMA-ADX TRADING SYSTEM - V3.0"
    )

    print(
        "       B3.2 SESSION BOUNDARY EVIDENCE"
    )

    print("=" * 110)

    print("\n")
    print(
        "Diagnostic-only stage."
    )

    print(
        "Raw M1 data will NOT be modified."
    )

    print(
        "session_validator.py will NOT be modified."
    )

    for symbol in SYMBOL_FILES:

        try:

            process_symbol(
                symbol
            )

        except Exception as error:

            print("\n")
            print("=" * 110)

            print(
                f"B3.2 ERROR — {symbol}"
            )

            print("=" * 110)

            print(
                repr(error)
            )

    print("\n")
    print("=" * 110)

    print(
        "B3.2 SESSION BOUNDARY EVIDENCE COMPLETE"
    )

    print("=" * 110)

    print("\n")
    print(
        "DO NOT COMMIT B3.2 YET."
    )

    print(
        "Review the evidence output first."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

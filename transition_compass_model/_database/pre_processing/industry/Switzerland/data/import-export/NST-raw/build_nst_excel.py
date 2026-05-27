import os

import pandas as pd

YEAR = 2024

current_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(current_dir, "..")

flow_map = {
    "EXP": "NST4_VK_EXP_en_v1.csv",
    "IMP": "NST4_VK_IMP_en_v1.csv",
}

for flow, csv_file in flow_map.items():
    df = pd.read_csv(os.path.join(current_dir, csv_file), sep=";", low_memory=False)

    df = df[df["year"] == YEAR]

    df = (
        df.groupby(["NST4", "NST4_txt"], as_index=False)[["Quantity_kg", "Value_CHF"]]
        .sum()
        .sort_values("NST4")
        .reset_index(drop=True)
    )

    df["NST4_txt"] = df["NST4_txt"].str.replace("\xa0", " ", regex=False)
    df["kt"] = df["Quantity_kg"] / 1e6
    df["chf-mio"] = df["Value_CHF"] / 1e6

    data = df[["NST4_txt", "kt", "chf-mio"]].copy()
    data.columns = ["Product group (NST4)", "kt", "chf-mio"]

    header_rows = [
        [f"Swiss {flow.lower()}ports by NST4 product group, {YEAR}", None, None],
        ["Source: Federal Office for Customs and Border Security FOCBS", None, None],
        ["Product group (NST4)", "kt", "chf-mio"],
    ]
    footer_rows = [
        ["Source: FOCBS / opendata.swiss", None, None],
        [f"Year: {YEAR}", None, None],
        ["Units: kt (thousand tonnes), chf-mio (million CHF)", None, None],
    ]

    header_df = pd.DataFrame(header_rows, columns=data.columns)
    footer_df = pd.DataFrame(footer_rows, columns=data.columns)
    out_df = pd.concat([header_df, data, footer_df], ignore_index=True)

    out_path = os.path.join(out_dir, f"2_8_VT_NST_{flow}_en.xlsx")
    out_df.to_excel(out_path, index=False, header=False)
    print(f"Saved {out_path}  ({len(data)} product groups)")


if __name__ == "__main__":
    pass

# src/relationships.py

import pandas as pd


def suggest_relationships(datasets: dict) -> pd.DataFrame:
    """
    Gợi ý quan hệ giữa các bảng bằng cách tìm các cột trùng tên.
    """
    relationships = []
    table_names = list(datasets.keys())

    for i in range(len(table_names)):
        for j in range(i + 1, len(table_names)):
            left_table = table_names[i]
            right_table = table_names[j]

            left_df = datasets[left_table]
            right_df = datasets[right_table]

            common_cols = set(left_df.columns).intersection(set(right_df.columns))

            for col in common_cols:
                left_unique = left_df[col].nunique(dropna=True)
                right_unique = right_df[col].nunique(dropna=True)

                relationships.append({
                    "Bảng trái": left_table,
                    "Bảng phải": right_table,
                    "Khóa gợi ý": col,
                    "Unique bên trái": left_unique,
                    "Unique bên phải": right_unique,
                })

    return pd.DataFrame(relationships)


def merge_two_tables(
    datasets: dict,
    left_table: str,
    right_table: str,
    join_key: str,
    join_type: str = "left") -> pd.DataFrame:
    """
    Merge 2 bảng theo key và kiểu join được chọn.
    """
    left_df = datasets[left_table]
    right_df = datasets[right_table]

    merged_df = left_df.merge(
        right_df,
        on=join_key,
        how=join_type,
        suffixes=("", f"_{right_table}")
    )

    return merged_df
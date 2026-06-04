"""
Pandas cross-validation fixture generator for tabgo.

Generates deterministic test cases exercising pandas DataFrame/Series operations
and stores inputs + expected outputs as JSON fixtures for Go cross-validation tests.
"""

import pandas as pd
import numpy as np
import json
import math


def _safe(x):
    """Convert a value to JSON-safe form."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        if np.isnan(x):
            return None
        if np.isinf(x):
            return None
        return float(x)
    if isinstance(x, float):
        if math.isnan(x):
            return None
        if math.isinf(x):
            return None
        return x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_safe(v) for v in x.tolist()]
    if isinstance(x, pd.Series):
        return [_safe(v) for v in x.tolist()]
    if isinstance(x, list):
        return [_safe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _safe(v) for k, v in x.items()}
    if isinstance(x, pd.DataFrame):
        return _df_to_dict(x)
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return x


def _df_to_dict(df):
    """Convert DataFrame to a dict of column->list for JSON."""
    result = {}
    for col in df.columns:
        result[str(col)] = [_safe(v) for v in df[col].tolist()]
    return result


def _series_to_list(s):
    """Convert Series to list for JSON."""
    return [_safe(v) for v in s.tolist()]


def generate():
    test_cases = []

    # =========================================================================
    # CATEGORY 1: DataFrame Creation & Properties (10 cases)
    # =========================================================================

    # 1.1: Create from dict with mixed types
    df = pd.DataFrame({"A": [1, 2, 3], "B": [1.5, 2.5, 3.5], "C": ["x", "y", "z"]})
    test_cases.append({
        "name": "creation_from_dict_mixed",
        "category": "creation",
        "input": {
            "columns": {"A": [1, 2, 3], "B": [1.5, 2.5, 3.5], "C": ["x", "y", "z"]}
        },
        "expected": {
            "shape": [3, 3],
            "columns": sorted(list(df.columns)),
            "size": int(df.size),
            "ndim": int(df.ndim),
            "empty": bool(df.empty),
            "values": _df_to_dict(df)
        }
    })

    # 1.2: Create from list of records
    records = [{"A": 10, "B": 20}, {"A": 30, "B": 40}]
    df = pd.DataFrame(records)
    test_cases.append({
        "name": "creation_from_records",
        "category": "creation",
        "input": {
            "records": records,
            "column_names": ["A", "B"]
        },
        "expected": {
            "shape": [2, 2],
            "columns": sorted(list(df.columns)),
            "values": _df_to_dict(df)
        }
    })

    # 1.3: Empty DataFrame
    df = pd.DataFrame()
    test_cases.append({
        "name": "creation_empty",
        "category": "creation",
        "input": {},
        "expected": {
            "shape": [0, 0],
            "empty": True,
            "size": 0,
            "ndim": 2
        }
    })

    # 1.4: Single column
    df = pd.DataFrame({"X": [10, 20, 30, 40]})
    test_cases.append({
        "name": "creation_single_column",
        "category": "creation",
        "input": {"columns": {"X": [10, 20, 30, 40]}},
        "expected": {
            "shape": [4, 1],
            "columns": ["X"],
            "values": _df_to_dict(df)
        }
    })

    # 1.5: Single row
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    test_cases.append({
        "name": "creation_single_row",
        "category": "creation",
        "input": {"columns": {"A": [1], "B": [2], "C": [3]}},
        "expected": {
            "shape": [1, 3],
            "columns": sorted(list(df.columns)),
            "values": _df_to_dict(df)
        }
    })

    # 1.6: With None/NaN values - count nulls
    df = pd.DataFrame({"A": [1.0, None, 3.0], "B": [None, 2.0, 3.0], "C": [1.0, 2.0, None]})
    test_cases.append({
        "name": "creation_with_nulls",
        "category": "creation",
        "input": {
            "columns": {"A": [1.0, None, 3.0], "B": [None, 2.0, 3.0], "C": [1.0, 2.0, None]}
        },
        "expected": {
            "shape": [3, 3],
            "null_counts": {"A": 1, "B": 1, "C": 1},
            "non_null_counts": {"A": 2, "B": 2, "C": 2}
        }
    })

    # 1.7: Wide DataFrame
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8], "E": [9, 10]})
    test_cases.append({
        "name": "creation_wide",
        "category": "creation",
        "input": {"columns": {"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8], "E": [9, 10]}},
        "expected": {
            "shape": [2, 5],
            "columns": sorted(["A", "B", "C", "D", "E"]),
            "size": 10,
            "values": _df_to_dict(df)
        }
    })

    # 1.8: All nulls in one column
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [None, None, None]})
    test_cases.append({
        "name": "creation_all_nulls_column",
        "category": "creation",
        "input": {"columns": {"A": [1.0, 2.0, 3.0], "B": [None, None, None]}},
        "expected": {
            "shape": [3, 2],
            "null_counts": {"A": 0, "B": 3},
            "non_null_counts": {"A": 3, "B": 0}
        }
    })

    # 1.9: Large numeric DataFrame
    df = pd.DataFrame({"X": list(range(1, 11)), "Y": list(range(10, 0, -1))})
    test_cases.append({
        "name": "creation_numeric_range",
        "category": "creation",
        "input": {"columns": {"X": list(range(1, 11)), "Y": list(range(10, 0, -1))}},
        "expected": {
            "shape": [10, 2],
            "values": _df_to_dict(df)
        }
    })

    # 1.10: Float precision
    df = pd.DataFrame({"A": [0.1 + 0.2, 1.0/3.0, 1e-15]})
    test_cases.append({
        "name": "creation_float_precision",
        "category": "creation",
        "input": {"columns": {"A": [0.1 + 0.2, 1.0/3.0, 1e-15]}},
        "expected": {
            "shape": [3, 1],
            "values": _df_to_dict(df)
        }
    })

    # =========================================================================
    # CATEGORY 2: Aggregation (20 cases)
    # =========================================================================

    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50], "C": [1.5, 2.5, 3.5, 4.5, 5.5]})

    # 2.1: sum
    test_cases.append({
        "name": "agg_sum",
        "category": "aggregation",
        "input": {"columns": {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50], "C": [1.5, 2.5, 3.5, 4.5, 5.5]}},
        "expected": {"sum": _safe(df.sum(numeric_only=True).to_dict())}
    })

    # 2.2: mean
    test_cases.append({
        "name": "agg_mean",
        "category": "aggregation",
        "input": {"columns": {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50], "C": [1.5, 2.5, 3.5, 4.5, 5.5]}},
        "expected": {"mean": _safe(df.mean(numeric_only=True).to_dict())}
    })

    # 2.3: std (ddof=1)
    test_cases.append({
        "name": "agg_std",
        "category": "aggregation",
        "input": {"columns": {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50], "C": [1.5, 2.5, 3.5, 4.5, 5.5]}},
        "expected": {"std": _safe(df.std(numeric_only=True).to_dict())}
    })

    # 2.4: var (ddof=1)
    test_cases.append({
        "name": "agg_var",
        "category": "aggregation",
        "input": {"columns": {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50], "C": [1.5, 2.5, 3.5, 4.5, 5.5]}},
        "expected": {"var": _safe(df.var(numeric_only=True).to_dict())}
    })

    # 2.5: min
    test_cases.append({
        "name": "agg_min",
        "category": "aggregation",
        "input": {"columns": {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50], "C": [1.5, 2.5, 3.5, 4.5, 5.5]}},
        "expected": {"min": _safe(df.min(numeric_only=True).to_dict())}
    })

    # 2.6: max
    test_cases.append({
        "name": "agg_max",
        "category": "aggregation",
        "input": {"columns": {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50], "C": [1.5, 2.5, 3.5, 4.5, 5.5]}},
        "expected": {"max": _safe(df.max(numeric_only=True).to_dict())}
    })

    # 2.7: median
    test_cases.append({
        "name": "agg_median",
        "category": "aggregation",
        "input": {"columns": {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50], "C": [1.5, 2.5, 3.5, 4.5, 5.5]}},
        "expected": {"median": _safe(df.median(numeric_only=True).to_dict())}
    })

    # 2.8: count
    test_cases.append({
        "name": "agg_count",
        "category": "aggregation",
        "input": {"columns": {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50], "C": [1.5, 2.5, 3.5, 4.5, 5.5]}},
        "expected": {"count": {"A": 5, "B": 5, "C": 5}}
    })

    # 2.9: describe
    desc = df.describe()
    test_cases.append({
        "name": "agg_describe",
        "category": "aggregation",
        "input": {"columns": {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50], "C": [1.5, 2.5, 3.5, 4.5, 5.5]}},
        "expected": {"describe": {
            col: {stat: _safe(desc.loc[stat, col]) for stat in desc.index}
            for col in desc.columns
        }}
    })

    # 2.10: sum with NaN
    df_nan = pd.DataFrame({"A": [1.0, np.nan, 3.0, np.nan, 5.0], "B": [10.0, 20.0, np.nan, 40.0, 50.0]})
    test_cases.append({
        "name": "agg_sum_nan",
        "category": "aggregation",
        "input": {"columns": {"A": [1.0, None, 3.0, None, 5.0], "B": [10.0, 20.0, None, 40.0, 50.0]}},
        "expected": {"sum": _safe(df_nan.sum(numeric_only=True).to_dict())}
    })

    # 2.11: mean with NaN
    test_cases.append({
        "name": "agg_mean_nan",
        "category": "aggregation",
        "input": {"columns": {"A": [1.0, None, 3.0, None, 5.0], "B": [10.0, 20.0, None, 40.0, 50.0]}},
        "expected": {"mean": _safe(df_nan.mean(numeric_only=True).to_dict())}
    })

    # 2.12: count with NaN
    test_cases.append({
        "name": "agg_count_nan",
        "category": "aggregation",
        "input": {"columns": {"A": [1.0, None, 3.0, None, 5.0], "B": [10.0, 20.0, None, 40.0, 50.0]}},
        "expected": {"count": {"A": 3, "B": 4}}
    })

    # 2.13: std with NaN
    test_cases.append({
        "name": "agg_std_nan",
        "category": "aggregation",
        "input": {"columns": {"A": [1.0, None, 3.0, None, 5.0], "B": [10.0, 20.0, None, 40.0, 50.0]}},
        "expected": {"std": _safe(df_nan.std(numeric_only=True).to_dict())}
    })

    # 2.14: var with NaN
    test_cases.append({
        "name": "agg_var_nan",
        "category": "aggregation",
        "input": {"columns": {"A": [1.0, None, 3.0, None, 5.0], "B": [10.0, 20.0, None, 40.0, 50.0]}},
        "expected": {"var": _safe(df_nan.var(numeric_only=True).to_dict())}
    })

    # 2.15: min with NaN
    test_cases.append({
        "name": "agg_min_nan",
        "category": "aggregation",
        "input": {"columns": {"A": [1.0, None, 3.0, None, 5.0], "B": [10.0, 20.0, None, 40.0, 50.0]}},
        "expected": {"min": _safe(df_nan.min(numeric_only=True).to_dict())}
    })

    # 2.16: max with NaN
    test_cases.append({
        "name": "agg_max_nan",
        "category": "aggregation",
        "input": {"columns": {"A": [1.0, None, 3.0, None, 5.0], "B": [10.0, 20.0, None, 40.0, 50.0]}},
        "expected": {"max": _safe(df_nan.max(numeric_only=True).to_dict())}
    })

    # 2.17: median with NaN
    test_cases.append({
        "name": "agg_median_nan",
        "category": "aggregation",
        "input": {"columns": {"A": [1.0, None, 3.0, None, 5.0], "B": [10.0, 20.0, None, 40.0, 50.0]}},
        "expected": {"median": _safe(df_nan.median(numeric_only=True).to_dict())}
    })

    # 2.18: describe with NaN
    desc_nan = df_nan.describe()
    test_cases.append({
        "name": "agg_describe_nan",
        "category": "aggregation",
        "input": {"columns": {"A": [1.0, None, 3.0, None, 5.0], "B": [10.0, 20.0, None, 40.0, 50.0]}},
        "expected": {"describe": {
            col: {stat: _safe(desc_nan.loc[stat, col]) for stat in desc_nan.index}
            for col in desc_nan.columns
        }}
    })

    # 2.19: Even number of elements - median
    df_even = pd.DataFrame({"A": [1, 2, 3, 4], "B": [10, 20, 30, 40]})
    test_cases.append({
        "name": "agg_median_even",
        "category": "aggregation",
        "input": {"columns": {"A": [1, 2, 3, 4], "B": [10, 20, 30, 40]}},
        "expected": {"median": _safe(df_even.median(numeric_only=True).to_dict())}
    })

    # 2.20: Single element
    df_single = pd.DataFrame({"A": [42.0], "B": [3.14]})
    test_cases.append({
        "name": "agg_single_element",
        "category": "aggregation",
        "input": {"columns": {"A": [42.0], "B": [3.14]}},
        "expected": {
            "sum": _safe(df_single.sum(numeric_only=True).to_dict()),
            "mean": _safe(df_single.mean(numeric_only=True).to_dict()),
            "min": _safe(df_single.min(numeric_only=True).to_dict()),
            "max": _safe(df_single.max(numeric_only=True).to_dict()),
            "count": {"A": 1, "B": 1}
        }
    })

    # =========================================================================
    # CATEGORY 3: GroupBy (15 cases)
    # =========================================================================

    gdf = pd.DataFrame({
        "Group": ["A", "A", "B", "B", "C"],
        "Value": [10, 20, 30, 40, 50],
        "Score": [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    gb_input = {
        "columns": {"Group": ["A", "A", "B", "B", "C"], "Value": [10, 20, 30, 40, 50], "Score": [1.0, 2.0, 3.0, 4.0, 5.0]},
        "group_by": ["Group"],
        "value_columns": ["Value", "Score"]
    }

    # 3.1: groupby sum
    gs = gdf.groupby("Group")[["Value", "Score"]].sum()
    test_cases.append({
        "name": "groupby_sum",
        "category": "groupby",
        "input": gb_input,
        "expected": {"result": {col: {idx: _safe(gs.loc[idx, col]) for idx in gs.index} for col in gs.columns}}
    })

    # 3.2: groupby mean
    gm = gdf.groupby("Group")[["Value", "Score"]].mean()
    test_cases.append({
        "name": "groupby_mean",
        "category": "groupby",
        "input": gb_input,
        "expected": {"result": {col: {idx: _safe(gm.loc[idx, col]) for idx in gm.index} for col in gm.columns}}
    })

    # 3.3: groupby std
    gstd = gdf.groupby("Group")[["Value", "Score"]].std()
    test_cases.append({
        "name": "groupby_std",
        "category": "groupby",
        "input": gb_input,
        "expected": {"result": {col: {idx: _safe(gstd.loc[idx, col]) for idx in gstd.index} for col in gstd.columns}}
    })

    # 3.4: groupby count
    gc = gdf.groupby("Group")[["Value", "Score"]].count()
    test_cases.append({
        "name": "groupby_count",
        "category": "groupby",
        "input": gb_input,
        "expected": {"result": {col: {idx: _safe(gc.loc[idx, col]) for idx in gc.index} for col in gc.columns}}
    })

    # 3.5: groupby min
    gmin = gdf.groupby("Group")[["Value", "Score"]].min()
    test_cases.append({
        "name": "groupby_min",
        "category": "groupby",
        "input": gb_input,
        "expected": {"result": {col: {idx: _safe(gmin.loc[idx, col]) for idx in gmin.index} for col in gmin.columns}}
    })

    # 3.6: groupby max
    gmax = gdf.groupby("Group")[["Value", "Score"]].max()
    test_cases.append({
        "name": "groupby_max",
        "category": "groupby",
        "input": gb_input,
        "expected": {"result": {col: {idx: _safe(gmax.loc[idx, col]) for idx in gmax.index} for col in gmax.columns}}
    })

    # 3.7: groupby median
    gmed = gdf.groupby("Group")[["Value", "Score"]].median()
    test_cases.append({
        "name": "groupby_median",
        "category": "groupby",
        "input": gb_input,
        "expected": {"result": {col: {idx: _safe(gmed.loc[idx, col]) for idx in gmed.index} for col in gmed.columns}}
    })

    # 3.8: groupby first
    gfirst = gdf.groupby("Group").first()
    test_cases.append({
        "name": "groupby_first",
        "category": "groupby",
        "input": gb_input,
        "expected": {"result": {col: {idx: _safe(gfirst.loc[idx, col]) for idx in gfirst.index} for col in gfirst.columns}}
    })

    # 3.9: groupby last
    glast = gdf.groupby("Group").last()
    test_cases.append({
        "name": "groupby_last",
        "category": "groupby",
        "input": gb_input,
        "expected": {"result": {col: {idx: _safe(glast.loc[idx, col]) for idx in glast.index} for col in glast.columns}}
    })

    # 3.10: groupby size
    gsize = gdf.groupby("Group").size()
    test_cases.append({
        "name": "groupby_size",
        "category": "groupby",
        "input": gb_input,
        "expected": {"result": {idx: int(gsize.loc[idx]) for idx in gsize.index}}
    })

    # 3.11-3.12: Multi-column groupby
    mdf = pd.DataFrame({"Group": ["A", "A", "B", "B"], "Sub": ["x", "y", "x", "y"], "Val": [1, 2, 3, 4]})
    mgb = mdf.groupby(["Group", "Sub"])["Val"].sum()
    test_cases.append({
        "name": "groupby_multi_sum",
        "category": "groupby",
        "input": {
            "columns": {"Group": ["A", "A", "B", "B"], "Sub": ["x", "y", "x", "y"], "Val": [1, 2, 3, 4]},
            "group_by": ["Group", "Sub"],
            "value_columns": ["Val"]
        },
        "expected": {"result": {f"{idx[0]}|{idx[1]}": _safe(mgb.loc[idx]) for idx in mgb.index}}
    })

    # 3.12: multi groupby mean
    mgm = mdf.groupby(["Group", "Sub"])["Val"].mean()
    test_cases.append({
        "name": "groupby_multi_mean",
        "category": "groupby",
        "input": {
            "columns": {"Group": ["A", "A", "B", "B"], "Sub": ["x", "y", "x", "y"], "Val": [1, 2, 3, 4]},
            "group_by": ["Group", "Sub"],
            "value_columns": ["Val"]
        },
        "expected": {"result": {f"{idx[0]}|{idx[1]}": _safe(mgm.loc[idx]) for idx in mgm.index}}
    })

    # 3.13: groupby var
    gvar = gdf.groupby("Group")[["Value", "Score"]].var()
    test_cases.append({
        "name": "groupby_var",
        "category": "groupby",
        "input": gb_input,
        "expected": {"result": {col: {idx: _safe(gvar.loc[idx, col]) for idx in gvar.index} for col in gvar.columns}}
    })

    # 3.14: groupby with single member group
    test_cases.append({
        "name": "groupby_single_member",
        "category": "groupby",
        "input": gb_input,
        "expected": {
            "group_C_sum_Value": _safe(gdf[gdf["Group"]=="C"]["Value"].sum()),
            "group_C_mean_Value": _safe(gdf[gdf["Group"]=="C"]["Value"].mean()),
            "group_C_count": 1
        }
    })

    # 3.15: groupby ngroups
    test_cases.append({
        "name": "groupby_ngroups",
        "category": "groupby",
        "input": gb_input,
        "expected": {"ngroups": 3}
    })

    # =========================================================================
    # CATEGORY 4: Merge/Join (10 cases)
    # =========================================================================

    left = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    right = pd.DataFrame({"id": [2, 3, 4], "score": [90, 80, 70]})
    merge_input = {
        "left": {"columns": {"id": [1, 2, 3], "name": ["a", "b", "c"]}, "column_order": ["id", "name"]},
        "right": {"columns": {"id": [2, 3, 4], "score": [90, 80, 70]}, "column_order": ["id", "score"]},
        "on": ["id"]
    }

    # 4.1: inner merge
    mi = pd.merge(left, right, on="id", how="inner")
    test_cases.append({
        "name": "merge_inner",
        "category": "merge",
        "input": {**merge_input, "how": "inner"},
        "expected": {"result": _df_to_dict(mi), "shape": list(mi.shape)}
    })

    # 4.2: left merge
    ml = pd.merge(left, right, on="id", how="left")
    test_cases.append({
        "name": "merge_left",
        "category": "merge",
        "input": {**merge_input, "how": "left"},
        "expected": {"result": _df_to_dict(ml), "shape": list(ml.shape)}
    })

    # 4.3: right merge
    mr = pd.merge(left, right, on="id", how="right")
    test_cases.append({
        "name": "merge_right",
        "category": "merge",
        "input": {**merge_input, "how": "right"},
        "expected": {"result": _df_to_dict(mr), "shape": list(mr.shape)}
    })

    # 4.4: outer merge
    mo = pd.merge(left, right, on="id", how="outer")
    test_cases.append({
        "name": "merge_outer",
        "category": "merge",
        "input": {**merge_input, "how": "outer"},
        "expected": {"result": _df_to_dict(mo), "shape": list(mo.shape)}
    })

    # 4.5: duplicate keys inner
    left_dup = pd.DataFrame({"id": [1, 1, 2], "val": ["a", "b", "c"]})
    right_dup = pd.DataFrame({"id": [1, 2, 2], "score": [10, 20, 30]})
    md = pd.merge(left_dup, right_dup, on="id", how="inner")
    test_cases.append({
        "name": "merge_dup_inner",
        "category": "merge",
        "input": {
            "left": {"columns": {"id": [1, 1, 2], "val": ["a", "b", "c"]}, "column_order": ["id", "val"]},
            "right": {"columns": {"id": [1, 2, 2], "score": [10, 20, 30]}, "column_order": ["id", "score"]},
            "on": ["id"],
            "how": "inner"
        },
        "expected": {"result": _df_to_dict(md), "shape": list(md.shape)}
    })

    # 4.6: duplicate keys left
    mdl = pd.merge(left_dup, right_dup, on="id", how="left")
    test_cases.append({
        "name": "merge_dup_left",
        "category": "merge",
        "input": {
            "left": {"columns": {"id": [1, 1, 2], "val": ["a", "b", "c"]}, "column_order": ["id", "val"]},
            "right": {"columns": {"id": [1, 2, 2], "score": [10, 20, 30]}, "column_order": ["id", "score"]},
            "on": ["id"],
            "how": "left"
        },
        "expected": {"result": _df_to_dict(mdl), "shape": list(mdl.shape)}
    })

    # 4.7: duplicate keys outer
    mdo = pd.merge(left_dup, right_dup, on="id", how="outer")
    test_cases.append({
        "name": "merge_dup_outer",
        "category": "merge",
        "input": {
            "left": {"columns": {"id": [1, 1, 2], "val": ["a", "b", "c"]}, "column_order": ["id", "val"]},
            "right": {"columns": {"id": [1, 2, 2], "score": [10, 20, 30]}, "column_order": ["id", "score"]},
            "on": ["id"],
            "how": "outer"
        },
        "expected": {"result": _df_to_dict(mdo), "shape": list(mdo.shape)}
    })

    # 4.8: no overlap inner
    left_no = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
    right_no = pd.DataFrame({"id": [3, 4], "score": [90, 80]})
    mno = pd.merge(left_no, right_no, on="id", how="inner")
    test_cases.append({
        "name": "merge_no_overlap_inner",
        "category": "merge",
        "input": {
            "left": {"columns": {"id": [1, 2], "name": ["a", "b"]}, "column_order": ["id", "name"]},
            "right": {"columns": {"id": [3, 4], "score": [90, 80]}, "column_order": ["id", "score"]},
            "on": ["id"],
            "how": "inner"
        },
        "expected": {"shape": [0, 3]}
    })

    # 4.9: no overlap outer
    mno_o = pd.merge(left_no, right_no, on="id", how="outer")
    test_cases.append({
        "name": "merge_no_overlap_outer",
        "category": "merge",
        "input": {
            "left": {"columns": {"id": [1, 2], "name": ["a", "b"]}, "column_order": ["id", "name"]},
            "right": {"columns": {"id": [3, 4], "score": [90, 80]}, "column_order": ["id", "score"]},
            "on": ["id"],
            "how": "outer"
        },
        "expected": {"result": _df_to_dict(mno_o), "shape": list(mno_o.shape)}
    })

    # 4.10: complete overlap
    left_full = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    right_full = pd.DataFrame({"id": [1, 2, 3], "score": [100, 200, 300]})
    mf = pd.merge(left_full, right_full, on="id", how="inner")
    test_cases.append({
        "name": "merge_complete_overlap",
        "category": "merge",
        "input": {
            "left": {"columns": {"id": [1, 2, 3], "name": ["a", "b", "c"]}, "column_order": ["id", "name"]},
            "right": {"columns": {"id": [1, 2, 3], "score": [100, 200, 300]}, "column_order": ["id", "score"]},
            "on": ["id"],
            "how": "inner"
        },
        "expected": {"result": _df_to_dict(mf), "shape": list(mf.shape)}
    })

    # =========================================================================
    # CATEGORY 5: Concat (5 cases)
    # =========================================================================

    # 5.1: Vertical concat same columns
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
    vc = pd.concat([df1, df2], ignore_index=True)
    test_cases.append({
        "name": "concat_vertical",
        "category": "concat",
        "input": {
            "frames": [
                {"columns": {"A": [1, 2], "B": [3, 4]}},
                {"columns": {"A": [5, 6], "B": [7, 8]}}
            ]
        },
        "expected": {"result": _df_to_dict(vc), "shape": list(vc.shape)}
    })

    # 5.2: Vertical concat three DFs
    df3 = pd.DataFrame({"A": [9, 10], "B": [11, 12]})
    vc3 = pd.concat([df1, df2, df3], ignore_index=True)
    test_cases.append({
        "name": "concat_vertical_three",
        "category": "concat",
        "input": {
            "frames": [
                {"columns": {"A": [1, 2], "B": [3, 4]}},
                {"columns": {"A": [5, 6], "B": [7, 8]}},
                {"columns": {"A": [9, 10], "B": [11, 12]}}
            ]
        },
        "expected": {"result": _df_to_dict(vc3), "shape": list(vc3.shape)}
    })

    # 5.3: Horizontal concat
    hdf1 = pd.DataFrame({"A": [1, 2, 3]})
    hdf2 = pd.DataFrame({"B": [4, 5, 6]})
    hc = pd.concat([hdf1, hdf2], axis=1)
    test_cases.append({
        "name": "concat_horizontal",
        "category": "concat",
        "input": {
            "frames": [
                {"columns": {"A": [1, 2, 3]}},
                {"columns": {"B": [4, 5, 6]}}
            ],
            "axis": 1
        },
        "expected": {"result": _df_to_dict(hc), "shape": list(hc.shape)}
    })

    # 5.4: Horizontal concat three DFs
    hdf3 = pd.DataFrame({"C": [7, 8, 9]})
    hc3 = pd.concat([hdf1, hdf2, hdf3], axis=1)
    test_cases.append({
        "name": "concat_horizontal_three",
        "category": "concat",
        "input": {
            "frames": [
                {"columns": {"A": [1, 2, 3]}},
                {"columns": {"B": [4, 5, 6]}},
                {"columns": {"C": [7, 8, 9]}}
            ],
            "axis": 1
        },
        "expected": {"result": _df_to_dict(hc3), "shape": list(hc3.shape)}
    })

    # 5.5: Vertical concat single row DFs
    s1 = pd.DataFrame({"A": [1], "B": [2]})
    s2 = pd.DataFrame({"A": [3], "B": [4]})
    vs = pd.concat([s1, s2], ignore_index=True)
    test_cases.append({
        "name": "concat_vertical_single_rows",
        "category": "concat",
        "input": {
            "frames": [
                {"columns": {"A": [1], "B": [2]}},
                {"columns": {"A": [3], "B": [4]}}
            ]
        },
        "expected": {"result": _df_to_dict(vs), "shape": list(vs.shape)}
    })

    # =========================================================================
    # CATEGORY 6: Sorting (5 cases)
    # =========================================================================

    sdf = pd.DataFrame({"A": [3, 1, 4, 1, 5], "B": [10, 20, 30, 40, 50]})

    # 6.1: sort ascending
    sa = sdf.sort_values("A").reset_index(drop=True)
    test_cases.append({
        "name": "sort_ascending",
        "category": "sorting",
        "input": {"columns": {"A": [3, 1, 4, 1, 5], "B": [10, 20, 30, 40, 50]}, "by": "A", "ascending": True},
        "expected": {"result": _df_to_dict(sa)}
    })

    # 6.2: sort descending
    sd = sdf.sort_values("A", ascending=False).reset_index(drop=True)
    test_cases.append({
        "name": "sort_descending",
        "category": "sorting",
        "input": {"columns": {"A": [3, 1, 4, 1, 5], "B": [10, 20, 30, 40, 50]}, "by": "A", "ascending": False},
        "expected": {"result": _df_to_dict(sd)}
    })

    # 6.3: sort by second column
    sb = sdf.sort_values("B", ascending=False).reset_index(drop=True)
    test_cases.append({
        "name": "sort_by_B_desc",
        "category": "sorting",
        "input": {"columns": {"A": [3, 1, 4, 1, 5], "B": [10, 20, 30, 40, 50]}, "by": "B", "ascending": False},
        "expected": {"result": _df_to_dict(sb)}
    })

    # 6.4: sort already sorted
    already = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]})
    sa_already = already.sort_values("A").reset_index(drop=True)
    test_cases.append({
        "name": "sort_already_sorted",
        "category": "sorting",
        "input": {"columns": {"A": [1, 2, 3, 4, 5], "B": [5, 4, 3, 2, 1]}, "by": "A", "ascending": True},
        "expected": {"result": _df_to_dict(sa_already)}
    })

    # 6.5: sort with floats
    fdf = pd.DataFrame({"A": [3.14, 1.41, 2.72, 1.62, 0.58], "B": [1, 2, 3, 4, 5]})
    sf = fdf.sort_values("A").reset_index(drop=True)
    test_cases.append({
        "name": "sort_floats",
        "category": "sorting",
        "input": {"columns": {"A": [3.14, 1.41, 2.72, 1.62, 0.58], "B": [1, 2, 3, 4, 5]}, "by": "A", "ascending": True},
        "expected": {"result": _df_to_dict(sf)}
    })

    # =========================================================================
    # CATEGORY 7: Missing Data (8 cases)
    # =========================================================================

    mdf = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [np.nan, 2.0, 3.0], "C": [1.0, 2.0, np.nan]})
    missing_input = {"columns": {"A": [1.0, None, 3.0], "B": [None, 2.0, 3.0], "C": [1.0, 2.0, None]}}

    # 7.1: dropna (any)
    dropped = mdf.dropna().reset_index(drop=True)
    test_cases.append({
        "name": "missing_dropna",
        "category": "missing",
        "input": missing_input,
        "expected": {"shape": list(dropped.shape), "num_rows": len(dropped)}
    })

    # 7.2: dropna how='all'
    dropped_all = mdf.dropna(how="all").reset_index(drop=True)
    test_cases.append({
        "name": "missing_dropna_all",
        "category": "missing",
        "input": missing_input,
        "expected": {"shape": list(dropped_all.shape), "num_rows": len(dropped_all)}
    })

    # 7.3: fillna(0)
    filled = mdf.fillna(0)
    test_cases.append({
        "name": "missing_fillna_zero",
        "category": "missing",
        "input": missing_input,
        "expected": {"result": _df_to_dict(filled)}
    })

    # 7.4: fillna(-1)
    filled_neg = mdf.fillna(-1)
    test_cases.append({
        "name": "missing_fillna_neg1",
        "category": "missing",
        "input": missing_input,
        "expected": {"result": _df_to_dict(filled_neg)}
    })

    # 7.5: isna
    isna = mdf.isna()
    test_cases.append({
        "name": "missing_isna",
        "category": "missing",
        "input": missing_input,
        "expected": {"result": _df_to_dict(isna)}
    })

    # 7.6: count per column
    test_cases.append({
        "name": "missing_count",
        "category": "missing",
        "input": missing_input,
        "expected": {"count": {"A": 2, "B": 2, "C": 2}}
    })

    # 7.7: fillna(99.9)
    filled99 = mdf.fillna(99.9)
    test_cases.append({
        "name": "missing_fillna_99",
        "category": "missing",
        "input": missing_input,
        "expected": {"result": _df_to_dict(filled99)}
    })

    # 7.8: all NaN row
    mdf2 = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [1.0, np.nan, 3.0]})
    dropped2 = mdf2.dropna().reset_index(drop=True)
    test_cases.append({
        "name": "missing_dropna_all_nan_row",
        "category": "missing",
        "input": {"columns": {"A": [1.0, None, 3.0], "B": [1.0, None, 3.0]}},
        "expected": {"shape": list(dropped2.shape), "num_rows": len(dropped2)}
    })

    # =========================================================================
    # CATEGORY 8: Reshape (8 cases)
    # =========================================================================

    # 8.1: Melt
    wide = pd.DataFrame({"id": ["a", "b", "c"], "X": [1, 2, 3], "Y": [4, 5, 6]})
    melted = pd.melt(wide, id_vars=["id"], value_vars=["X", "Y"])
    test_cases.append({
        "name": "reshape_melt",
        "category": "reshape",
        "input": {
            "columns": {"id": ["a", "b", "c"], "X": [1, 2, 3], "Y": [4, 5, 6]},
            "column_order": ["id", "X", "Y"],
            "id_vars": ["id"],
            "value_vars": ["X", "Y"]
        },
        "expected": {"result": _df_to_dict(melted), "shape": list(melted.shape)}
    })

    # 8.2: Melt with more value vars
    wide2 = pd.DataFrame({"id": ["a", "b"], "X": [1, 2], "Y": [3, 4], "Z": [5, 6]})
    melted2 = pd.melt(wide2, id_vars=["id"], value_vars=["X", "Y", "Z"])
    test_cases.append({
        "name": "reshape_melt_three",
        "category": "reshape",
        "input": {
            "columns": {"id": ["a", "b"], "X": [1, 2], "Y": [3, 4], "Z": [5, 6]},
            "column_order": ["id", "X", "Y", "Z"],
            "id_vars": ["id"],
            "value_vars": ["X", "Y", "Z"]
        },
        "expected": {"result": _df_to_dict(melted2), "shape": list(melted2.shape)}
    })

    # 8.3: PivotTable mean
    long = pd.DataFrame({"Row": ["A", "A", "B", "B"], "Col": ["x", "y", "x", "y"], "Val": [1, 2, 3, 4]})
    piv = pd.pivot_table(long, index="Row", columns="Col", values="Val", aggfunc="mean")
    test_cases.append({
        "name": "reshape_pivot_mean",
        "category": "reshape",
        "input": {
            "columns": {"Row": ["A", "A", "B", "B"], "Col": ["x", "y", "x", "y"], "Val": [1, 2, 3, 4]},
            "index": "Row",
            "pivot_columns": "Col",
            "values": "Val",
            "aggfunc": "mean"
        },
        "expected": {"result": _df_to_dict(piv)}
    })

    # 8.4: PivotTable sum
    piv_sum = pd.pivot_table(long, index="Row", columns="Col", values="Val", aggfunc="sum")
    test_cases.append({
        "name": "reshape_pivot_sum",
        "category": "reshape",
        "input": {
            "columns": {"Row": ["A", "A", "B", "B"], "Col": ["x", "y", "x", "y"], "Val": [1, 2, 3, 4]},
            "index": "Row",
            "pivot_columns": "Col",
            "values": "Val",
            "aggfunc": "sum"
        },
        "expected": {"result": _df_to_dict(piv_sum)}
    })

    # 8.5: Crosstab
    ct_df = pd.DataFrame({"R": ["a", "a", "b", "b", "a"], "C": ["x", "y", "x", "y", "x"]})
    ct = pd.crosstab(ct_df["R"], ct_df["C"])
    test_cases.append({
        "name": "reshape_crosstab",
        "category": "reshape",
        "input": {
            "columns": {"R": ["a", "a", "b", "b", "a"], "C": ["x", "y", "x", "y", "x"]},
            "row": "R",
            "col": "C"
        },
        "expected": {"result": {str(c): {str(r): int(ct.loc[r, c]) for r in ct.index} for c in ct.columns}}
    })

    # 8.6: PivotTable count
    piv_count = pd.pivot_table(long, index="Row", columns="Col", values="Val", aggfunc="count")
    test_cases.append({
        "name": "reshape_pivot_count",
        "category": "reshape",
        "input": {
            "columns": {"Row": ["A", "A", "B", "B"], "Col": ["x", "y", "x", "y"], "Val": [1, 2, 3, 4]},
            "index": "Row",
            "pivot_columns": "Col",
            "values": "Val",
            "aggfunc": "count"
        },
        "expected": {"result": _df_to_dict(piv_count)}
    })

    # 8.7: PivotTable min
    piv_min = pd.pivot_table(long, index="Row", columns="Col", values="Val", aggfunc="min")
    test_cases.append({
        "name": "reshape_pivot_min",
        "category": "reshape",
        "input": {
            "columns": {"Row": ["A", "A", "B", "B"], "Col": ["x", "y", "x", "y"], "Val": [1, 2, 3, 4]},
            "index": "Row",
            "pivot_columns": "Col",
            "values": "Val",
            "aggfunc": "min"
        },
        "expected": {"result": _df_to_dict(piv_min)}
    })

    # 8.8: PivotTable max
    piv_max = pd.pivot_table(long, index="Row", columns="Col", values="Val", aggfunc="max")
    test_cases.append({
        "name": "reshape_pivot_max",
        "category": "reshape",
        "input": {
            "columns": {"Row": ["A", "A", "B", "B"], "Col": ["x", "y", "x", "y"], "Val": [1, 2, 3, 4]},
            "index": "Row",
            "pivot_columns": "Col",
            "values": "Val",
            "aggfunc": "max"
        },
        "expected": {"result": _df_to_dict(piv_max)}
    })

    # =========================================================================
    # CATEGORY 9: Statistics (10 cases)
    # =========================================================================

    stat_df = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [5.0, 4.0, 3.0, 2.0, 1.0], "C": [2.0, 4.0, 6.0, 8.0, 10.0]})
    stat_input = {"columns": {"A": [1.0, 2.0, 3.0, 4.0, 5.0], "B": [5.0, 4.0, 3.0, 2.0, 1.0], "C": [2.0, 4.0, 6.0, 8.0, 10.0]}}

    # 9.1: corr
    corr = stat_df.corr()
    test_cases.append({
        "name": "stats_corr",
        "category": "statistics",
        "input": stat_input,
        "expected": {"result": {str(c): {str(r): _safe(corr.loc[r, c]) for r in corr.index} for c in corr.columns}}
    })

    # 9.2: cov
    cov = stat_df.cov()
    test_cases.append({
        "name": "stats_cov",
        "category": "statistics",
        "input": stat_input,
        "expected": {"result": {str(c): {str(r): _safe(cov.loc[r, c]) for r in cov.index} for c in cov.columns}}
    })

    # 9.3: cumsum
    cs = stat_df.cumsum()
    test_cases.append({
        "name": "stats_cumsum",
        "category": "statistics",
        "input": stat_input,
        "expected": {"result": _df_to_dict(cs)}
    })

    # 9.4: diff
    d = stat_df.diff()
    test_cases.append({
        "name": "stats_diff",
        "category": "statistics",
        "input": stat_input,
        "expected": {"result": _df_to_dict(d)}
    })

    # 9.5: pct_change
    pc = stat_df.pct_change()
    test_cases.append({
        "name": "stats_pct_change",
        "category": "statistics",
        "input": stat_input,
        "expected": {"result": _df_to_dict(pc)}
    })

    # 9.6: rank
    rk = stat_df.rank()
    test_cases.append({
        "name": "stats_rank",
        "category": "statistics",
        "input": stat_input,
        "expected": {"result": _df_to_dict(rk)}
    })

    # 9.7: corr diagonal is 1
    test_cases.append({
        "name": "stats_corr_diagonal",
        "category": "statistics",
        "input": stat_input,
        "expected": {"diagonal": [_safe(corr.loc[c, c]) for c in corr.columns]}
    })

    # 9.8: corr symmetry
    test_cases.append({
        "name": "stats_corr_symmetry",
        "category": "statistics",
        "input": stat_input,
        "expected": {
            "AB": _safe(corr.loc["A", "B"]),
            "BA": _safe(corr.loc["B", "A"]),
            "AC": _safe(corr.loc["A", "C"]),
            "CA": _safe(corr.loc["C", "A"])
        }
    })

    # 9.9: rank with ties
    tie_df = pd.DataFrame({"A": [1.0, 2.0, 2.0, 3.0, 3.0]})
    tie_rank = tie_df.rank()
    test_cases.append({
        "name": "stats_rank_ties",
        "category": "statistics",
        "input": {"columns": {"A": [1.0, 2.0, 2.0, 3.0, 3.0]}},
        "expected": {"result": _df_to_dict(tie_rank)}
    })

    # 9.10: cumsum with specific values
    cs_df = pd.DataFrame({"A": [10.0, 20.0, 30.0], "B": [1.0, 1.0, 1.0]})
    cs_result = cs_df.cumsum()
    test_cases.append({
        "name": "stats_cumsum_specific",
        "category": "statistics",
        "input": {"columns": {"A": [10.0, 20.0, 30.0], "B": [1.0, 1.0, 1.0]}},
        "expected": {"result": _df_to_dict(cs_result)}
    })

    # =========================================================================
    # CATEGORY 10: Series Operations (10 cases)
    # =========================================================================

    s = pd.Series([3, 1, 4, 1, 5, 9, 2, 6], name="s")
    series_input = {"values": [3, 1, 4, 1, 5, 9, 2, 6]}

    # 10.1: value_counts
    vc = s.value_counts()
    test_cases.append({
        "name": "series_value_counts",
        "category": "series",
        "input": series_input,
        "expected": {"result": {str(k): int(v) for k, v in vc.items()}}
    })

    # 10.2: unique
    u = sorted([_safe(x) for x in s.unique().tolist()])
    test_cases.append({
        "name": "series_unique",
        "category": "series",
        "input": series_input,
        "expected": {"result_sorted": u, "count": len(u)}
    })

    # 10.3: nunique
    test_cases.append({
        "name": "series_nunique",
        "category": "series",
        "input": series_input,
        "expected": {"result": int(s.nunique())}
    })

    # 10.4: describe
    sd = s.astype(float).describe()
    test_cases.append({
        "name": "series_describe",
        "category": "series",
        "input": {"values": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]},
        "expected": {"result": {k: _safe(v) for k, v in sd.items()}}
    })

    # 10.5: aggregations
    sf = pd.Series([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], name="s")
    test_cases.append({
        "name": "series_aggregations",
        "category": "series",
        "input": {"values": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]},
        "expected": {
            "sum": _safe(sf.sum()),
            "mean": _safe(sf.mean()),
            "std": _safe(sf.std()),
            "var": _safe(sf.var()),
            "min": _safe(sf.min()),
            "max": _safe(sf.max()),
            "median": _safe(sf.median()),
            "count": int(sf.count())
        }
    })

    # 10.6: sort ascending
    ss = sf.sort_values().reset_index(drop=True)
    test_cases.append({
        "name": "series_sort_asc",
        "category": "series",
        "input": {"values": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]},
        "expected": {"result": _series_to_list(ss)}
    })

    # 10.7: clip
    clipped = sf.clip(lower=2, upper=7)
    test_cases.append({
        "name": "series_clip",
        "category": "series",
        "input": {"values": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], "lower": 2.0, "upper": 7.0},
        "expected": {"result": _series_to_list(clipped)}
    })

    # 10.8: abs
    neg_s = pd.Series([-3.0, -1.0, 4.0, -1.0, 5.0, -9.0, 2.0, -6.0], name="s")
    abs_s = neg_s.abs()
    test_cases.append({
        "name": "series_abs",
        "category": "series",
        "input": {"values": [-3.0, -1.0, 4.0, -1.0, 5.0, -9.0, 2.0, -6.0]},
        "expected": {"result": _series_to_list(abs_s)}
    })

    # 10.9: round
    float_s = pd.Series([3.14159, 2.71828, 1.41421, 1.61803], name="s")
    rounded = float_s.round(decimals=1)
    test_cases.append({
        "name": "series_round",
        "category": "series",
        "input": {"values": [3.14159, 2.71828, 1.41421, 1.61803], "decimals": 1},
        "expected": {"result": _series_to_list(rounded)}
    })

    # 10.10: rank
    rank_s = sf.rank()
    test_cases.append({
        "name": "series_rank",
        "category": "series",
        "input": {"values": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]},
        "expected": {"result": _series_to_list(rank_s)}
    })

    # =========================================================================
    # CATEGORY 11: CSV Round-Trip (3 cases)
    # =========================================================================

    # 11.1: basic CSV round-trip
    csv_df = pd.DataFrame({"A": [1, 2, 3], "B": [4.5, 5.5, 6.5]})
    csv_str = csv_df.to_csv(index=False)
    csv_read = pd.read_csv(pd.io.common.StringIO(csv_str))
    test_cases.append({
        "name": "csv_roundtrip_basic",
        "category": "csv",
        "input": {"columns": {"A": [1, 2, 3], "B": [4.5, 5.5, 6.5]}},
        "expected": {
            "csv_string": csv_str,
            "roundtrip_shape": list(csv_read.shape),
            "roundtrip_values": _df_to_dict(csv_read)
        }
    })

    # 11.2: CSV with NaN
    nan_csv_df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": [np.nan, 2.0, 3.0]})
    nan_csv_str = nan_csv_df.to_csv(index=False)
    test_cases.append({
        "name": "csv_roundtrip_nan",
        "category": "csv",
        "input": {"columns": {"A": [1.0, None, 3.0], "B": [None, 2.0, 3.0]}},
        "expected": {
            "csv_string": nan_csv_str,
            "has_nan": True
        }
    })

    # 11.3: CSV with mixed types
    mixed_csv_df = pd.DataFrame({"X": [1, 2, 3], "Y": [1.5, 2.5, 3.5]})
    mixed_csv_str = mixed_csv_df.to_csv(index=False)
    mixed_csv_read = pd.read_csv(pd.io.common.StringIO(mixed_csv_str))
    test_cases.append({
        "name": "csv_roundtrip_mixed",
        "category": "csv",
        "input": {"columns": {"X": [1, 2, 3], "Y": [1.5, 2.5, 3.5]}},
        "expected": {
            "csv_string": mixed_csv_str,
            "roundtrip_shape": list(mixed_csv_read.shape),
            "roundtrip_values": _df_to_dict(mixed_csv_read)
        }
    })

    return test_cases

from __future__ import annotations

import argparse
import math
import re
from difflib import SequenceMatcher
from typing import Iterable

import pandas as pd


DIMENSION_COLUMNS = ["astrio_height", "astrio_width", "astrio_length"]

CANONICAL_ALIASES = {
    "артикул": "sku",
    "sku": "sku",
    "article": "sku",
    "название": "name",
    "name": "name",
    "title": "name",
    "цена": "price",
    "price": "price",
    "цена мин": "price_min",
    "цена max": "price_max",
    "цена макс": "price_max",
    "price min": "price_min",
    "price max": "price_max",
    "brand": "brand",
    "бренд": "brand",
    "description": "description",
    "short_description": "short_description",
    "type": "short_description",
    "astrio_height": "astrio_height",
    "astrio_width": "astrio_width",
    "astrio_length": "astrio_length",
    "color": "color",
    "color_source": "color_source",
    "collections": "collections",
    "parser_backend": "parser_backend",
    "parser_priority": "parser_priority",
}

GENERIC_NAME_TOKENS = {
    "chair",
    "chairs",
    "seating",
    "table",
    "tables",
    "sofa",
    "sofas",
    "storage",
    "furniture",
    "collection",
    "catalogue",
    "catalog",
    "generic",
}

FAMILY_STOPWORDS = {
    "chair",
    "chairs",
    "armchair",
    "armchairs",
    "stool",
    "stools",
    "visitor",
    "executive",
    "task",
    "lounge",
    "barstool",
    "managerial",
    "hospitality",
    "office",
    "with",
    "without",
    "back",
    "base",
    "sled",
    "swivel",
}


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    seen = set(df.columns)
    for col in df.columns:
        key = str(col).strip().lower()
        canonical = CANONICAL_ALIASES.get(key)
        if canonical and canonical not in seen:
            rename_map[col] = canonical
            seen.add(canonical)
    out = df.rename(columns=rename_map).copy()
    if "price" not in out.columns:
        if "price_max" in out.columns:
            out["price"] = out["price_max"]
        elif "price_min" in out.columns:
            out["price"] = out["price_min"]
    for col in ["sku", "name", "brand", "description", "short_description", "color", "color_source", "collections", "price"]:
        if col not in out.columns:
            out[col] = ""
    if "parser_priority" not in out.columns:
        out["parser_priority"] = 0
    for col in DIMENSION_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def brand_from_pdf_name(pdf_name: object) -> str:
    text = str(pdf_name or "").strip()
    if not text:
        return ""
    stem = re.sub(r"\.pdf$", "", text, flags=re.IGNORECASE)
    stem = re.sub(r"\s*\(\d+\)$", "", stem).strip()
    first_chunk = re.split(r"_+", stem, maxsplit=1)[0].strip()
    if not first_chunk:
        first_chunk = re.split(r"\s+", stem, maxsplit=1)[0].strip()
    return re.sub(r"\s+", " ", first_chunk)


def harmonize_brand_per_pdf(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pdf_name" not in out.columns:
        return out
    derived = out["pdf_name"].apply(brand_from_pdf_name)
    out["brand"] = derived.where(derived.ne(""), out["brand"].fillna("").astype(str).str.strip())
    return out


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\s_/|]+", " ", text)
    text = re.sub(r"[^0-9a-zA-Z\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_sku(value: object) -> str:
    text = normalize_text(value).upper()
    text = text.replace(" ", "")
    return text


def normalize_price(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    text = str(value).strip().replace("\u00a0", " ").replace(" ", "").replace(",", ".")
    text = re.sub(r"[^0-9.]+", "", text)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_dimension(value: object) -> float | None:
    number = normalize_price(value)
    if number is None or number <= 0:
        return None
    return round(number, 2)


def tokenize(value: object) -> list[str]:
    text = normalize_text(value)
    return [token for token in text.split() if token]


def best_family_token(name: object, sku: object) -> str:
    for token in tokenize(name):
        if token not in FAMILY_STOPWORDS and len(token) > 1:
            return token
    sku_match = re.match(r"[A-Z]+", normalize_sku(sku))
    if sku_match:
        token = sku_match.group(0).lower()
        if len(token) > 1:
            return token
    return ""


def completeness_score(row: pd.Series) -> int:
    score = 0
    for col in ["sku", "name", "brand", "description", "short_description", "price"]:
        value = row.get(col)
        if value not in ("", None) and not pd.isna(value):
            score += 1
    for col in DIMENSION_COLUMNS:
        value = row.get(col)
        if value not in (None, "") and not pd.isna(value) and float(value) > 0:
            score += 2
    try:
        score += int(float(row.get("parser_priority", 0)))
    except Exception:
        pass
    return score


def name_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(a=left, b=right).ratio()


def relative_close(left: float | None, right: float | None, tolerance: float = 0.05) -> bool:
    if left in (None, 0) or right in (None, 0):
        return False
    denom = max(abs(left), abs(right), 1.0)
    return abs(left - right) / denom <= tolerance


def dimensions_close(left: pd.Series, right: pd.Series, tolerance: float = 0.05) -> bool:
    shared = []
    for col in DIMENSION_COLUMNS:
        lv = left.get(col)
        rv = right.get(col)
        if pd.notna(lv) and pd.notna(rv) and float(lv) > 0 and float(rv) > 0:
            shared.append(relative_close(float(lv), float(rv), tolerance=tolerance))
    if not shared:
        return False
    return sum(shared) >= max(1, len(shared) - 1)


def probable_duplicate(left: pd.Series, right: pd.Series) -> tuple[bool, str]:
    same_sku = bool(left["_norm_sku"]) and left["_norm_sku"] == right["_norm_sku"]
    same_name = bool(left["_norm_name"]) and left["_norm_name"] == right["_norm_name"]
    similar_name = name_similarity(left["_norm_name"], right["_norm_name"]) >= 0.9
    price_close = relative_close(left["price_num"], right["price_num"], tolerance=0.06)
    dims_match = dimensions_close(left, right, tolerance=0.06)

    if same_sku and (same_name or similar_name or dims_match or price_close):
        return True, "same_sku"
    if same_name and (price_close or dims_match):
        return True, "same_name"
    same_brand = bool(left["_brand_key"]) and left["_brand_key"] == right["_brand_key"]
    same_family = bool(left["_family_key"]) and left["_family_key"] == right["_family_key"]
    if same_brand and same_family and similar_name and price_close and dims_match:
        return True, "family_match"
    return False, ""


def add_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = canonicalize_columns(df)
    out = harmonize_brand_per_pdf(out)
    out["sku"] = out["sku"].fillna("").astype(str).str.strip()
    out["name"] = out["name"].fillna("").astype(str).str.strip()
    out["brand"] = out["brand"].fillna("").astype(str).str.strip()
    out["short_description"] = out["short_description"].fillna("").astype(str).str.strip()
    out["collections"] = out["collections"].fillna("").astype(str).str.strip()
    out["price_num"] = out["price"].apply(normalize_price)
    for col in DIMENSION_COLUMNS:
        out[col] = out[col].apply(normalize_dimension)
        source_col = f"{col}_source"
        out[source_col] = out[col].apply(lambda value: "parsed" if pd.notna(value) else "missing")
    out["_norm_sku"] = out["sku"].apply(normalize_sku)
    out["_norm_name"] = out["name"].apply(normalize_text)
    out["_brand_key"] = out["brand"].apply(normalize_text)
    out["_type_key"] = out["short_description"].apply(normalize_text)
    out["_collection_key"] = out["collections"].apply(normalize_text)
    out["_family_key"] = out.apply(lambda row: best_family_token(row["name"], row["sku"]), axis=1)
    out["_completeness_score"] = out.apply(completeness_score, axis=1)
    out["quality_flag"] = "ok"
    out["quality_notes"] = ""
    return out


def deduplicate_records(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    exact_subset = ["_norm_sku", "_norm_name", "price_num"] + DIMENSION_COLUMNS
    scored = df.sort_values(by=["_completeness_score", "_norm_sku", "_norm_name"], ascending=[False, True, True]).copy()
    deduped = scored.drop_duplicates(subset=exact_subset, keep="first").copy()

    kept_rows: list[pd.Series] = []
    consumed: set[int] = set()
    candidate_groups = deduped.groupby(["_brand_key", "_family_key"], dropna=False, sort=False)
    for _, group in candidate_groups:
        indexes = list(group.index)
        for idx in indexes:
            if idx in consumed:
                continue
            anchor = deduped.loc[idx]
            cluster = [idx]
            methods = []
            for other_idx in indexes:
                if other_idx == idx or other_idx in consumed:
                    continue
                other = deduped.loc[other_idx]
                is_dup, method = probable_duplicate(anchor, other)
                if is_dup:
                    cluster.append(other_idx)
                    methods.append(method)
            winner = deduped.loc[cluster].sort_values(
                by=["_completeness_score", "price_num"], ascending=[False, False], na_position="last"
            ).iloc[0].copy()
            winner["_dedupe_cluster_size"] = len(cluster)
            winner["_dedupe_method"] = ",".join(sorted(set(methods))) if methods else "exact"
            kept_rows.append(winner)
            consumed.update(cluster)

    out = pd.DataFrame(kept_rows).reset_index(drop=True)
    out = out.sort_values(
        by=["_completeness_score", "price_num"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)
    same_key_mask = out["_norm_sku"].ne("") & out["_norm_name"].ne("")
    collapsed = pd.concat(
        [
            out.loc[same_key_mask].drop_duplicates(subset=["_norm_sku", "_norm_name"], keep="first"),
            out.loc[~same_key_mask],
        ],
        ignore_index=True,
    )
    out = collapsed.reset_index(drop=True)
    if "_dedupe_cluster_size" not in out.columns:
        out["_dedupe_cluster_size"] = 1
    if "_dedupe_method" not in out.columns:
        out["_dedupe_method"] = "exact"
    return out


def append_quality_note(df: pd.DataFrame, mask: pd.Series, note: str, flag: str) -> None:
    if not mask.any():
        return
    current = df.loc[mask, "quality_notes"].fillna("")
    prefix = current.where(current.eq(""), current + "; ")
    df.loc[mask, "quality_notes"] = prefix + note
    df.loc[mask, "quality_flag"] = flag


def mark_garbage_records(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_dims = out[DIMENSION_COLUMNS].apply(pd.to_numeric, errors="coerce")
    dims_missing = numeric_dims.fillna(0).le(0).all(axis=1)
    price_missing = out["price_num"].fillna(0).le(0)
    name_tokens = out["_norm_name"].apply(lambda value: len(value.split()))
    name_equals_brand = out["_norm_name"].eq(out["_brand_key"]) & out["_norm_name"].ne("")
    generic_name = out["_norm_name"].isin(GENERIC_NAME_TOKENS)
    short_numeric_sku = out["_norm_sku"].str.fullmatch(r"\d{1,3}", na=False)
    garbage_mask = price_missing & dims_missing & (name_equals_brand | generic_name | (name_tokens <= 1) & short_numeric_sku)
    append_quality_note(out, garbage_mask, "garbage_candidate", "garbage_candidate")
    return out


def robust_group_outliers(df: pd.DataFrame, group_cols: Iterable[str], min_group_size: int = 4) -> pd.DataFrame:
    out = df.copy()
    metrics = ["price_num"] + DIMENSION_COLUMNS
    out["outlier_score"] = 0

    for _, group in out.groupby(list(group_cols), dropna=False, sort=False):
        if len(group) < min_group_size:
            continue
        score = pd.Series(0, index=group.index, dtype="int64")
        for metric in metrics:
            series = pd.to_numeric(group[metric], errors="coerce").dropna()
            series = series[series > 0]
            if len(series) < min_group_size:
                continue
            median = series.median()
            mad = (series - median).abs().median()
            if mad == 0:
                continue
            robust_z = 0.6745 * (group[metric] - median).abs() / mad
            score = score.add(robust_z.fillna(0).gt(4.5).astype(int), fill_value=0)
        out.loc[group.index, "outlier_score"] = out.loc[group.index, "outlier_score"].add(score, fill_value=0)

    strong_outlier = out["outlier_score"] >= 2
    append_quality_note(out, strong_outlier, "strong_outlier", "strong_outlier")
    return out


def donor_values(df: pd.DataFrame, row: pd.Series, dim_col: str, key_cols: list[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col in key_cols:
        value = row.get(col, "")
        if value in ("", None) or pd.isna(value):
            return pd.Series(dtype="float64")
        mask &= df[col].eq(value)
    mask &= pd.Series(df.index != row.name, index=df.index)
    numeric_dim = pd.to_numeric(df[dim_col], errors="coerce")
    mask &= numeric_dim.fillna(0).gt(0)
    return numeric_dim.loc[mask].dropna()


def stable_fill_value(values: pd.Series, allow_single: bool = False) -> float | None:
    if values.empty:
        return None
    if len(values) == 1:
        return round(float(values.iloc[0]), 2) if allow_single else None
    median = float(values.median())
    if median <= 0:
        return None
    variation = values.std(ddof=0) / median if median else 0.0
    if variation > 0.18 and len(values) < 4:
        return None
    if variation > 0.25:
        return None
    return round(median, 2)


def impute_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lookup_chain = [
        ("same_sku", ["_norm_sku"], True),
        ("same_name", ["_brand_key", "_norm_name"], True),
        ("family_and_type", ["_brand_key", "_family_key", "_type_key"], False),
        ("family", ["_brand_key", "_family_key"], False),
        ("collection_and_type", ["_brand_key", "_collection_key", "_type_key"], False),
    ]

    for idx, row in out.iterrows():
        if row["quality_flag"] in {"garbage_candidate", "strong_outlier"}:
            continue
        for dim_col in DIMENSION_COLUMNS:
            if pd.notna(row[dim_col]) and float(row[dim_col]) > 0:
                continue
            for source_name, keys, allow_single in lookup_chain:
                candidates = donor_values(out, row, dim_col, keys)
                fill_value = stable_fill_value(candidates, allow_single=allow_single)
                if fill_value is None:
                    continue
                out.at[idx, dim_col] = fill_value
                out.at[idx, f"{dim_col}_source"] = source_name
                current_note = out.at[idx, "quality_notes"]
                suffix = f"imputed_{dim_col}:{source_name}"
                out.at[idx, "quality_notes"] = suffix if not current_note else f"{current_note}; {suffix}"
                if out.at[idx, "quality_flag"] == "ok":
                    out.at[idx, "quality_flag"] = "imputed"
                break
    return out


def mark_conflicting_skus(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for _, group in out.groupby("_norm_sku", dropna=False, sort=False):
        sku_key = group["_norm_sku"].iloc[0]
        if not sku_key or len(group) < 2:
            continue
        distinct_names = group["_norm_name"].replace("", pd.NA).dropna().nunique()
        prices = group["price_num"].dropna()
        has_price_conflict = False
        if len(prices) >= 2:
            price_min = float(prices.min())
            price_max = float(prices.max())
            if price_min > 0 and (price_max - price_min) / price_min > 0.08:
                has_price_conflict = True
        if distinct_names > 1 or has_price_conflict:
            mask = out.index.isin(group.index)
            current = out.loc[mask, "quality_notes"].fillna("")
            prefix = current.where(current.eq(""), current + "; ")
            out.loc[mask, "quality_notes"] = prefix + "sku_conflict_review"
            out.loc[mask, "quality_flag"] = out.loc[mask, "quality_flag"].replace({"ok": "needs_review", "imputed": "needs_review"})
    return out


def finalize_output(df: pd.DataFrame, drop_garbage: bool, drop_outliers: bool) -> pd.DataFrame:
    out = df.copy()
    if drop_garbage:
        out = out[out["quality_flag"] != "garbage_candidate"].copy()
    if drop_outliers:
        out = out[out["quality_flag"] != "strong_outlier"].copy()
    if "price" in out.columns:
        out["price"] = out["price_num"].round(2)
    sort_cols = [col for col in ["brand", "collections", "sku", "name"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(by=sort_cols, na_position="last").reset_index(drop=True)
    return out


def clean_furniture_catalog(
    df: pd.DataFrame,
    *,
    drop_garbage: bool = True,
    drop_outliers: bool = False,
) -> pd.DataFrame:
    working = add_helper_columns(df)
    working = working[~(working["sku"].eq("") & working["name"].eq("") & working["price_num"].isna())].copy()
    working = deduplicate_records(working)
    working = mark_garbage_records(working)
    working = robust_group_outliers(working, group_cols=["_brand_key", "_family_key"])
    working = impute_dimensions(working)
    working = mark_conflicting_skus(working)
    return finalize_output(working, drop_garbage=drop_garbage, drop_outliers=drop_outliers)


def summarize_changes(original: pd.DataFrame, cleaned: pd.DataFrame) -> dict[str, int]:
    summary = {
        "input_rows": len(original),
        "output_rows": len(cleaned),
    }
    for col in DIMENSION_COLUMNS:
        source_col = f"{col}_source"
        if source_col in cleaned.columns:
            summary[f"{col}_imputed"] = int(cleaned[source_col].isin({"same_sku", "same_name", "family_and_type", "family", "collection_and_type"}).sum())
    if "quality_flag" in cleaned.columns:
        flags = cleaned["quality_flag"].value_counts().to_dict()
        for key, value in flags.items():
            summary[f"flag_{key}"] = int(value)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean parsed furniture catalog CSV.")
    parser.add_argument("input_csv", help="Path to parser output CSV")
    parser.add_argument("-o", "--output", default="catalog_product_cleaned.csv", help="Output CSV path")
    parser.add_argument("--keep-outliers", action="store_true", help="Keep strong outlier rows in the output")
    parser.add_argument("--keep-garbage", action="store_true", help="Keep obvious garbage rows in the output")
    args = parser.parse_args()

    raw = pd.read_csv(args.input_csv)
    cleaned = clean_furniture_catalog(
        raw,
        drop_garbage=not args.keep_garbage,
        drop_outliers=not args.keep_outliers,
    )
    cleaned.to_csv(args.output, index=False)

    summary = summarize_changes(raw, cleaned)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

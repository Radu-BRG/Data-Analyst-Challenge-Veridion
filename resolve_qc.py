from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import pandas as pd

# Fuzzy matching
try:
    from rapidfuzz import fuzz

    def name_similarity(a: str, b: str) -> float:
        return fuzz.token_set_ratio(a, b) / 100.0
# If rapidfuzz isn't available (used for the same thing)
except Exception:
    import difflib

    def name_similarity(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()


# Normalization
LEGAL_SUFFIXES = {
    "ltd","limited","plc","inc","inc.","corp","corporation","co","co.","gmbh","ag","sa","s.a.","sas","sarl",
    "bv","b.v.","ab","as","oy","kk","pte","pte.","pty","pty.","llc","lp","l.p.","sp","sp.","zoo",
    "sro","s.r.o.","sr","s.r.","kft","nv","n.v.","private","pvt","pvt.","company","companies",
}

def safe_str(x) -> str:
    return "" if pd.isna(x) else str(x)

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_company_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"[&]", " and ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [t for t in s.split() if t and t not in LEGAL_SUFFIXES]
    return " ".join(tokens)

def split_aliases(s: str) -> list[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [a.strip() for a in s.split("|") if a.strip()]

def extract_domain(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    u = url.strip()
    if not re.match(r"^https?://", u, re.IGNORECASE):
        u = "http://" + u
    try:
        netloc = urlparse(u).netloc.lower()
    except Exception:
        return ""
    netloc = netloc.split("@")[-1].split(":")[0]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


# Scoring
@dataclass(frozen=True)
class Thresholds:
    min_score_auto: float = 0.85
    min_name_auto: float = 0.75
    require_country_code_match: bool = True
    low_max: float = 0.75
    med_max: float = 0.90

def score_candidate(row: pd.Series) -> dict:
    inp_name = normalize_company_name(safe_str(row.get("input_company_name", "")))

    cand_names_raw = (
        [safe_str(row.get("company_name", ""))]
        + split_aliases(safe_str(row.get("company_legal_names", "")))
        + split_aliases(safe_str(row.get("company_commercial_names", "")))
    )
    cand_names = [normalize_company_name(n) for n in cand_names_raw if n]
    sim = max((name_similarity(inp_name, cn) for cn in cand_names), default=0.0) if inp_name else 0.0

    inp_cc = normalize_text(safe_str(row.get("input_main_country_code", "")))
    cand_cc = normalize_text(safe_str(row.get("main_country_code", "")))
    country_match = 1.0 if (inp_cc and cand_cc and inp_cc == cand_cc) else 0.0

    inp_city = normalize_text(safe_str(row.get("input_main_city", "")))
    cand_city = normalize_text(safe_str(row.get("main_city", "")))
    city_match = 1.0 if (inp_city and cand_city and inp_city == cand_city) else 0.0

    inp_pc = normalize_text(safe_str(row.get("input_main_postcode", "")))
    cand_pc = normalize_text(safe_str(row.get("main_postcode", "")))
    postcode_match = 1.0 if (inp_pc and cand_pc and inp_pc == cand_pc) else 0.0

    cand_domain = safe_str(row.get("website_domain", ""))
    if not cand_domain:
        cand_domain = extract_domain(safe_str(row.get("website_url", "")))
    has_website = 1.0 if cand_domain else 0.0

    # Scoring in importance order
    total = (1.00 * sim) + (0.35 * country_match) + (0.15 * city_match) + (0.10 * postcode_match) + (0.05 * has_website)

    # Country mismatch
    if inp_cc and cand_cc and inp_cc != cand_cc:
        total -= 0.50

    return {
        "name_similarity": sim,
        "country_match": country_match,
        "city_match": city_match,
        "postcode_match": postcode_match,
        "has_website": has_website,
        "candidate_domain": cand_domain,
        "score": total,
    }


# Quality check flags
def add_qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["flag_no_website"] = df["candidate_domain"].eq("")
    df["flag_weak_name_match"] = df["name_similarity"] < 0.75

    df["input_cc_norm"] = df["input_main_country_code"].map(normalize_text)
    df["cand_cc_norm"] = df["main_country_code"].map(normalize_text)
    df["flag_country_mismatch"] = df["input_cc_norm"].ne("") & df["cand_cc_norm"].ne("") & df["input_cc_norm"].ne(df["cand_cc_norm"])

    df["input_city_norm"] = df["input_main_city"].map(normalize_text)
    df["cand_city_norm"] = df["main_city"].map(normalize_text)
    df["flag_city_mismatch"] = df["input_city_norm"].ne("") & df["cand_city_norm"].ne("") & df["input_city_norm"].ne(df["cand_city_norm"])

    df["input_pc_norm"] = df["input_main_postcode"].map(normalize_text)
    df["cand_pc_norm"] = df["main_postcode"].map(normalize_text)
    df["flag_postcode_mismatch"] = df["input_pc_norm"].ne("") & df["cand_pc_norm"].ne("") & df["input_pc_norm"].ne(df["cand_pc_norm"])
# Check if unique ID is duplicate, otherwise check if 2 or more instances have the same name, country, website
    if "veridion_id" in df.columns:
        df["flag_candidate_reused_across_inputs"] = df.duplicated(subset=["veridion_id"], keep=False)
    else:
        subset = [c for c in ["company_name", "main_country_code", "website_domain", "website_url"] if c in df.columns]
        df["flag_candidate_reused_across_inputs"] = df.duplicated(subset=subset, keep=False) if subset else False

    return df


# Core
def resolve_best_matches(
    csv_path: str | Path,
    out_dir: str | Path = ".",
    group_key: str = "input_row_key",
    thresholds: Thresholds = Thresholds(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = Path(csv_path)
    out_dir = csv_path.parent

    df = pd.read_csv(csv_path)

    required = {"input_company_name", group_key, "company_name", "main_country_code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    scores = df.apply(score_candidate, axis=1, result_type="expand")
    df = pd.concat([df, scores], axis=1)

    df_qc = add_qc_flags(df)

    sort_cols = ["score", "name_similarity", "country_match", "city_match", "postcode_match", "has_website"]
    df_sorted = df_qc.sort_values(by=[group_key] + sort_cols, ascending=[True, False, False, False, False, False, False])
    best = df_sorted.groupby(group_key, as_index=False).head(1).copy()

    best["confidence"] = pd.cut(
        best["score"],
        bins=[-99, thresholds.low_max, thresholds.med_max, 99],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    auto = (best["score"] >= thresholds.min_score_auto) & (best["name_similarity"] >= thresholds.min_name_auto)
    if thresholds.require_country_code_match:
        auto = auto & (best["country_match"] >= 1.0)

    best["match_status"] = np.where(auto, "MATCHED", "UNMATCHED_REVIEW")

    best["resolved_veridion_id"] = np.where(auto, best.get("veridion_id", ""), "")
    best["resolved_company_name"] = np.where(auto, best.get("company_name", ""), "")
    best["resolved_website_domain"] = np.where(auto, best.get("candidate_domain", ""), "")
    best["resolved_main_country_code"] = np.where(auto, best.get("main_country_code", ""), "")

    # Review
    top2 = df_sorted.groupby(group_key, as_index=False).head(2).copy()
    top2["candidate_rank"] = top2.groupby(group_key).cumcount() + 1
    review_queue = top2[top2[group_key].isin(best.loc[best["match_status"] != "MATCHED", group_key])].copy()
    best.to_csv(out_dir / "resolved_best_matches.csv", index=False)
    df_qc.to_csv(out_dir / "qc_flags.csv", index=False)
    review_queue.to_csv(out_dir / "review_queue_top2.csv", index=False)

    return best, df_qc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pick best ER candidate + allow UNMATCHED + generate QC flags.")
    parser.add_argument("csv_path", help="Path to ER POC CSV (input rows + candidate matches).")
    parser.add_argument("--out", default=".", help="Output directory.")
    parser.add_argument("--group-key", default="input_row_key", help="Column grouping candidates per input.")
    parser.add_argument("--min-score-auto", type=float, default=0.85, help="Auto-match minimum score.")
    parser.add_argument("--min-name-auto", type=float, default=0.75, help="Auto-match minimum name similarity.")
    parser.add_argument("--no-require-country", action="store_true", help="Do not require exact country code match to auto-match.")
    args = parser.parse_args()

    thr = Thresholds(
        min_score_auto=args.min_score_auto,
        min_name_auto=args.min_name_auto,
        require_country_code_match=not args.no_require_country,
    )

    best_df, qc_df = resolve_best_matches(args.csv_path, args.out, args.group_key, thr)
    print(f"Saved best matches to: {Path(args.out) / 'resolved_best_matches.csv'}")
    print(f"Saved QC flags to: {Path(args.out) / 'qc_flags.csv'}")
    print(f"Saved review queue to: {Path(args.out) / 'review_queue_top2.csv'}")
    print(f"Inputs resolved: {len(best_df)}")
    print(f"Candidate rows QC: {len(qc_df)}")

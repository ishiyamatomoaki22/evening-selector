import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Juggler Evening Selector", layout="wide")
st.title("ジャグラー 夕方続行 判定ツール（自分専用）")
st.caption("CSVをアップロードして、夕方の“続行候補台”を抽出します。")

# ---- 確率表記を float(分母) に統一 ----
def parse_rate(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "")
    if s == "":
        return np.nan
    m = re.match(r"^1\s*/\s*([0-9]+(?:\.[0-9]+)?)$", s)
    if m:
        return float(m.group(1))
    try:
        return float(s)
    except ValueError:
        return np.nan

# ---- CSV列名のゆれを吸収（必要なら増やせる） ----
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "台番": "unit_number",
        "台番号": "unit_number",
        "総回転": "total_start",
        "累計スタート": "total_start",
        "BB回数": "bb_count",
        "RB回数": "rb_count",
        "合成確率": "gassan_rate",
        "BB確率": "bb_rate",
        "RB確率": "rb_rate",
        "店舗": "shop",
        "機種": "machine",
        "日付": "date",
    }
    return df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

# ---- rate列が無い/空なら計算で補完 ----
def compute_rates(df: pd.DataFrame) -> pd.DataFrame:
    # 数値化
    for c in ["total_start", "bb_count", "rb_count"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 存在しない列は作る
    for c in ["bb_rate", "rb_rate", "gassan_rate"]:
        if c not in df.columns:
            df[c] = np.nan

    # 文字列 "1/xxx" -> xxx
    df["bb_rate"] = df["bb_rate"].map(parse_rate)
    df["rb_rate"] = df["rb_rate"].map(parse_rate)
    df["gassan_rate"] = df["gassan_rate"].map(parse_rate)

    # 欠損を計算で補完
    bb_mask = df["bb_rate"].isna() & df["bb_count"].gt(0)
    rb_mask = df["rb_rate"].isna() & df["rb_count"].gt(0)
    gs_mask = df["gassan_rate"].isna() & (df["bb_count"] + df["rb_count"]).gt(0)

    df.loc[bb_mask, "bb_rate"] = df.loc[bb_mask, "total_start"] / df.loc[bb_mask, "bb_count"]
    df.loc[rb_mask, "rb_rate"] = df.loc[rb_mask, "total_start"] / df.loc[rb_mask, "rb_count"]
    df.loc[gs_mask, "gassan_rate"] = df.loc[gs_mask, "total_start"] / (df.loc[gs_mask, "bb_count"] + df.loc[gs_mask, "rb_count"])
    return df

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

# ================= UI =================
uploaded = st.file_uploader("CSVをアップロード", type=["csv"])

left, right = st.columns([1, 2], vertical_alignment="top")

with left:
    st.subheader("夕方判定ルール（スライダーで調整）")
    min_games = st.slider("最低 総回転（total_start）", 0, 10000, 3000, 100)
    max_rb = st.slider("REG確率（rb_rate）上限（小さいほど良い）", 150.0, 600.0, 270.0, 1.0)
    max_gassan = st.slider("合算（gassan_rate）上限（小さいほど良い）", 80.0, 300.0, 150.0, 1.0)

    st.divider()
    st.subheader("表示")
    top_n = st.number_input("上位N件表示", 1, 200, 30, 1)
    sort_key = st.selectbox("並び順", ["REG優先", "合算優先", "総回転優先"])

with right:
    if not uploaded:
        st.info("CSVをアップロードすると、続行候補台を抽出して表示します。")
        st.stop()

    df = pd.read_csv(uploaded)
    df = normalize_columns(df)

    # 必須列チェック
    required = ["unit_number", "total_start", "bb_count", "rb_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"必須列が足りません: {missing}\n\n最低限: unit_number, total_start, bb_count, rb_count が必要です。")
        st.stop()

    df = compute_rates(df)

    # 候補抽出（夕方専用）
    cand = df[
        (df["total_start"] >= min_games) &
        (df["rb_rate"].notna()) &
        (df["gassan_rate"].notna()) &
        (df["rb_rate"] <= max_rb) &
        (df["gassan_rate"] <= max_gassan)
    ].copy()

    # スコア（見やすさ用：REG最重視）
    cand["score"] = (
        (max_rb / cand["rb_rate"]) * 70 +
        (cand["total_start"] / max(min_games, 1)) * 20 +
        (max_gassan / cand["gassan_rate"]) * 10
    )

    # ソート
    if sort_key == "REG優先":
        cand = cand.sort_values(["rb_rate", "total_start"], ascending=[True, False])
    elif sort_key == "合算優先":
        cand = cand.sort_values(["gassan_rate", "rb_rate"], ascending=[True, True])
    else:
        cand = cand.sort_values(["total_start", "rb_rate"], ascending=[False, True])

    st.subheader("続行候補（本日データのみ）")

    if cand.empty:
        st.warning("条件に合う台がありません。閾値を緩めるか、回転数が増えてから再判定してください。")
        st.stop()

    # 表示列
    show_cols = [c for c in [
        "date", "shop", "machine",
        "unit_number", "total_start",
        "bb_count", "rb_count",
        "bb_rate", "rb_rate", "gassan_rate",
        "score"
    ] if c in cand.columns]

    view = cand[show_cols].head(int(top_n)).copy()
    for c in ["bb_rate", "rb_rate", "gassan_rate", "score"]:
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors="coerce").round(1)

    st.dataframe(view, use_container_width=True, hide_index=True)

    st.download_button(
        "抽出結果をCSVでダウンロード",
        data=to_csv_bytes(cand[show_cols]),
        file_name="evening_candidates.csv",
        mime="text/csv",
    )

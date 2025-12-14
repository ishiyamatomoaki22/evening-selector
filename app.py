import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date

# ========= Config =========
HEADER = [
    "date","shop","machine",
    "unit_number","start_games","total_start","bb_count","rb_count","art_count","max_medals",
    "bb_rate","rb_rate","art_rate","gassan_rate","prev_day_end"
]

SHOP_PRESETS = ["武蔵境", "吉祥寺", "三鷹", "国分寺", "新宿", "渋谷"]
MACHINE_PRESETS = ["マイジャグラーV", "アイムジャグラーEX", "ファンキージャグラー2", "ゴーゴージャグラー3"]

st.set_page_config(page_title="ジャグラー夕方セレクター", layout="wide")
st.title("ジャグラー 夕方続行セレクター（変換→選定まで一発）")
st.caption("入力（CSV or 生テキスト）→ ヘッダー統一 → 夕方判定 → 候補台を出力")

# ========= Helpers =========
def parse_rate_token(tok: str) -> float:
    """ '1/186.3' -> 186.3 , '186.3' -> 186.3 """
    if pd.isna(tok):
        return np.nan
    s = str(tok).strip().replace(",", "")
    if s == "":
        return np.nan
    m = re.match(r"^1\s*/\s*([0-9]+(?:\.[0-9]+)?)$", s)
    if m:
        return float(m.group(1))
    try:
        return float(s)
    except ValueError:
        return np.nan

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # 列名の空白除去
    df.columns = df.columns.astype(str).str.strip()

    rename_map = {
        "台番": "unit_number",
        "台番号": "unit_number",
        "総回転": "total_start",
        "累計スタート": "total_start",
        "BB回数": "bb_count",
        "RB回数": "rb_count",
        "ART回数": "art_count",
        "最大持ち玉": "max_medals",
        "BB確率": "bb_rate",
        "RB確率": "rb_rate",
        "ART確率": "art_rate",
        "合成確率": "gassan_rate",
        "前日最終": "prev_day_end",
        "店舗": "shop",
        "機種": "machine",
        "日付": "date",
    }
    return df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

def compute_rates_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    # まず numeric 列の整形（カンマ除去も）
    for c in ["unit_number","start_games","total_start","bb_count","rb_count","art_count","max_medals","prev_day_end"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")

    # rate列が無ければ作る
    for c in ["bb_rate","rb_rate","art_rate","gassan_rate"]:
        if c not in df.columns:
            df[c] = np.nan

    # 文字列の確率を分母に統一
    df["bb_rate"] = df["bb_rate"].map(parse_rate_token)
    df["rb_rate"] = df["rb_rate"].map(parse_rate_token)
    df["art_rate"] = df["art_rate"].map(parse_rate_token)
    df["gassan_rate"] = df["gassan_rate"].map(parse_rate_token)

    # 欠損を total_start / 回数 で補完（必要な場合）
    if "total_start" in df.columns:
        bb_mask = df["bb_rate"].isna() & df["bb_count"].gt(0)
        rb_mask = df["rb_rate"].isna() & df["rb_count"].gt(0)
        gs_mask = df["gassan_rate"].isna() & (df["bb_count"].add(df["rb_count"]).gt(0))

        df.loc[bb_mask, "bb_rate"] = df.loc[bb_mask, "total_start"] / df.loc[bb_mask, "bb_count"]
        df.loc[rb_mask, "rb_rate"] = df.loc[rb_mask, "total_start"] / df.loc[rb_mask, "rb_count"]
        df.loc[gs_mask, "gassan_rate"] = df.loc[gs_mask, "total_start"] / (df.loc[gs_mask, "bb_count"] + df.loc[gs_mask, "rb_count"])

    return df

def clean_to_12_parts(line: str):
    """
    先頭にアイコン等のゴミが混ざっても、
    数値/確率(1/xxx)だけ抽出して12列に整形する。
    """
    line = line.strip()
    if not line:
        return None

    parts = re.split(r"\s+", line)

    # 数値 or 1/数値 だけ残す
    def is_data_token(tok: str) -> bool:
        tok = tok.strip().replace(",", "")
        return bool(re.match(r"^(?:\d+(?:\.\d+)?|1/\d+(?:\.\d+)?)$", tok))

    data_parts = [p.replace(",", "") for p in parts if is_data_token(p)]

    # 先頭に余計なものが入った場合は data_parts が13個以上になることがある
    if len(data_parts) > 12:
        data_parts = data_parts[-12:]  # ★最後の12個を採用（先頭ゴミ対策）

    if len(data_parts) != 12:
        return None  # 呼び出し側でエラー処理

    return data_parts


def parse_raw12(text: str, date_str: str, shop: str, machine: str) -> pd.DataFrame:
    rows = []
    for line_no, line in enumerate((text or "").splitlines(), start=1):
        parts12 = clean_to_12_parts(line)
        if parts12 is None:
            # 空行はスキップ、そうでなければ原因が分かるようにエラー
            if line.strip() == "":
                continue
            raise ValueError(f"{line_no}行目：12列に整形できませんでした: {line}")

        unit_number, start_games, total_start, bb_count, rb_count, art_count, max_medals, bb_rate, rb_rate, art_rate, gassan_rate, prev_day_end = parts12

        rows.append({
            "date": date_str, "shop": shop, "machine": machine,
            "unit_number": unit_number, "start_games": start_games, "total_start": total_start,
            "bb_count": bb_count, "rb_count": rb_count, "art_count": art_count, "max_medals": max_medals,
            "bb_rate": bb_rate, "rb_rate": rb_rate, "art_rate": art_rate, "gassan_rate": gassan_rate,
            "prev_day_end": prev_day_end
        })

    df = pd.DataFrame(rows, columns=HEADER)
    df = compute_rates_if_needed(df)
    return df

def ensure_meta_columns(df: pd.DataFrame, date_str: str, shop: str, machine: str) -> pd.DataFrame:
    if "date" not in df.columns:
        df["date"] = date_str
    if "shop" not in df.columns:
        df["shop"] = shop
    if "machine" not in df.columns:
        df["machine"] = machine
    # 空欄補完
    df["date"] = df["date"].fillna(date_str)
    df["shop"] = df["shop"].fillna(shop)
    df["machine"] = df["machine"].fillna(machine)
    return df

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

# ========= Sidebar: meta & thresholds =========
with st.sidebar:
    st.header("補完情報（date / shop / machine）")
    d = st.date_input("日付", value=date.today())
    date_str = d.strftime("%Y-%m-%d")

    shop_mode = st.radio("店名", ["選択", "手入力"], horizontal=True)
    if shop_mode == "選択":
        shop = st.selectbox("shop", SHOP_PRESETS, index=0)
    else:
        shop = st.text_input("shop", value="武蔵境")

    machine_mode = st.radio("機種", ["選択", "手入力"], horizontal=True)
    if machine_mode == "選択":
        machine = st.selectbox("machine", MACHINE_PRESETS, index=0)
    else:
        machine = st.text_input("machine", value="マイジャグラーV")

    st.divider()
    st.header("夕方判定（スライダー）")
    min_games = st.slider("最低 総回転（total_start）", 0, 10000, 3000, 100)
    max_rb = st.slider("REG上限（rb_rate）", 150.0, 600.0, 270.0, 1.0)
    max_gassan = st.slider("合算上限（gassan_rate）", 80.0, 350.0, 180.0, 1.0)

    top_n = st.number_input("上位N件表示", 1, 200, 30, 1)

# ========= Main UI =========
tab1, tab2 = st.tabs(["入力 → 変換", "夕方候補（選定結果）"])

# セッションに保存（タブを跨いで使う）
if "df_unified" not in st.session_state:
    st.session_state["df_unified"] = None

with tab1:
    st.subheader("入力方法を選んでください")
    input_mode = st.radio("入力", ["CSVアップロード", "生データ貼り付け（12列）"], horizontal=True)

    if input_mode == "CSVアップロード":
        uploaded = st.file_uploader("CSVをアップロード", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            df = normalize_columns(df)
            df = ensure_meta_columns(df, date_str, shop, machine)
            df = compute_rates_if_needed(df)

            # 必須列チェック
            required = ["unit_number","total_start","bb_count","rb_count","rb_rate","gassan_rate"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"必要な列が足りません: {missing}")
            else:
                # 列順をHEADERに寄せる（無い列はそのまま）
                for c in HEADER:
                    if c not in df.columns:
                        df[c] = np.nan
                df = df[HEADER]
                st.session_state["df_unified"] = df
                st.success(f"CSVを読み込み＆統一しました：{len(df)}行")
                st.dataframe(df.head(30), use_container_width=True, hide_index=True)
                st.download_button("統一済みCSVをダウンロード", data=to_csv_bytes(df), file_name="unified.csv", mime="text/csv")

    else:
        sample = "478 45 3539 19 11 0 2481 1/186.3 1/321.7 0.0 118.0 449"
        raw_text = st.text_area("台データオンラインの行を貼り付け（複数行OK）", value=sample, height=220)
        if st.button("変換して統一する", type="primary"):
            df = parse_raw12(raw_text, date_str, shop, machine)
            st.session_state["df_unified"] = df
            st.success(f"貼り付けデータを統一しました：{len(df)}行")
            st.dataframe(df.head(30), use_container_width=True, hide_index=True)
            st.download_button("統一済みCSVをダウンロード", data=to_csv_bytes(df), file_name="unified.csv", mime="text/csv")

with tab2:
    st.subheader("夕方候補（本日データのみ）")
    df = st.session_state.get("df_unified")

    if df is None or len(df) == 0:
        st.info("先に「入力 → 変換」タブでデータを読み込んでください。")
        st.stop()

    # =========================
    # 安定化：比較・計算用の数値列を作る
    # =========================
    df = df.copy()
    df["total_start_num"] = pd.to_numeric(df["total_start"], errors="coerce")
    df["rb_rate_num"] = pd.to_numeric(df["rb_rate"], errors="coerce")
    df["gassan_rate_num"] = pd.to_numeric(df["gassan_rate"], errors="coerce")

    # =========================
    # 夕方候補抽出（num列で判定）
    # =========================
    cand = df[
        (df["total_start_num"] >= min_games) &
        (df["rb_rate_num"] <= max_rb) &
        (df["gassan_rate_num"] <= max_gassan)
    ].copy()

    if cand.empty:
        st.warning("条件に合う台がありません。閾値を緩めるか、回転数が増えてから再判定してください。")
        # デバッグしたいときは以下を一時的に表示すると便利です
        # st.write(df[["unit_number","total_start","rb_rate","gassan_rate","total_start_num","rb_rate_num","gassan_rate_num"]].head(30))
        st.stop()

    # =========================
    # スコア（REG最重視：num列で計算）
    # =========================
    cand["score"] = (
        (max_rb / cand["rb_rate_num"]) * 70 +
        (cand["total_start_num"] / max(min_games, 1)) * 20 +
        (max_gassan / cand["gassan_rate_num"]) * 10
    )

    # 並び順：REG優先 → 総回転
    cand = cand.sort_values(["rb_rate_num", "total_start_num"], ascending=[True, False])

    # =========================
    # 表示用（見せたい列だけ）
    # =========================
    show = cand[[
        "date","shop","machine",
        "unit_number","total_start","bb_count","rb_count",
        "bb_rate","rb_rate","gassan_rate","score"
    ]].copy()

    # 表示を整える（小数1桁）
    for c in ["bb_rate","rb_rate","gassan_rate","score"]:
        show[c] = pd.to_numeric(show[c], errors="coerce").round(1)

    st.dataframe(show.head(int(top_n)), use_container_width=True, hide_index=True)

    st.download_button(
        "候補台をCSVでダウンロード",
        data=to_csv_bytes(show),
        file_name="evening_candidates.csv",
        mime="text/csv",
    )

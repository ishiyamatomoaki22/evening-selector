import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date, datetime
from zoneinfo import ZoneInfo

# ========= Config =========
HEADER = [
    "date","shop","machine",
    "unit_number","start_games","total_start","bb_count","rb_count","art_count","max_medals",
    "bb_rate","rb_rate","art_rate","gassan_rate","prev_day_end"
]

SHOP_PRESETS = ["武蔵境", "吉祥寺", "三鷹", "国分寺", "新宿", "渋谷"]
MACHINE_PRESETS = ["マイジャグラーV", "ゴーゴージャグラー3", "ハッピージャグラーVIII", "ファンキージャグラー2KT", "ミスタージャグラー", "ジャグラーガールズSS", "ネオアイムジャグラーEX", "ウルトラミラクルジャグラー"]

st.set_page_config(page_title="ジャグラー夕方セレクター", layout="wide")
st.title("ジャグラー 夕方続行セレクター（変換→選定まで一発）")
st.caption("入力（CSV or 生テキスト）→ ヘッダー統一 → 夕方判定 → 候補台を出力")

# 機種ごとのおすすめ設定（夕方向け・目安）
RECOMMENDED = {
    "マイジャグラーV":         {"min_games": 3000, "max_rb": 270.0, "max_gassan": 180.0},
    "ゴーゴージャグラー3":     {"min_games": 3000, "max_rb": 280.0, "max_gassan": 185.0},
    "ハッピージャグラーVIII":  {"min_games": 3500, "max_rb": 260.0, "max_gassan": 175.0},
    "ファンキージャグラー2KT": {"min_games": 3000, "max_rb": 300.0, "max_gassan": 190.0},
    "ミスタージャグラー":      {"min_games": 2800, "max_rb": 300.0, "max_gassan": 190.0},
    "ジャグラーガールズSS":    {"min_games": 2500, "max_rb": 260.0, "max_gassan": 175.0},
    "ネオアイムジャグラーEX":  {"min_games": 2500, "max_rb": 330.0, "max_gassan": 200.0},
    "ウルトラミラクルジャグラー":{"min_games": 3500, "max_rb": 300.0, "max_gassan": 195.0},
}

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

def make_filename(machine: str, suffix: str, date_str: str) -> str:
    """
    YYYY-MM-dd_HH-mm-ss_機種名_suffix.csv
    suffix例: evening / candidates
    """
    time_part = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%H-%M-%S")

    safe_machine = (
        str(machine)
        .replace(" ", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "-")
    )

    return f"{date_str}_{time_part}_{safe_machine}_{suffix}.csv"

# ======== Play Log (append to uploaded CSV) ========
PLAYLOG_HEADER = [
    "created_at",          # 記録作成日時（自動）
    "date","shop","machine","unit_number",
    "start_time","end_time",
    "invest_medals","payout_medals","profit_medals",
    "play_games",
    "stop_reason","memo"
]

def append_row_to_uploaded_csv(uploaded_bytes: bytes, new_row: dict) -> bytes:
    """
    既存CSV(アップロード)を読み込み → 1行追記 → CSV bytes を返す
    """
    df = pd.read_csv(io.BytesIO(uploaded_bytes))

    # ヘッダー不足でも壊れないように補完
    for c in PLAYLOG_HEADER:
        if c not in df.columns:
            df[c] = np.nan
    df = df[PLAYLOG_HEADER]

    # 追記行
    df2 = pd.DataFrame([new_row], columns=PLAYLOG_HEADER)
    df_out = pd.concat([df, df2], ignore_index=True)

    return df_out.to_csv(index=False).encode("utf-8-sig")

def make_safe_filename_part(s: str) -> str:
    return (
        str(s)
        .replace(" ", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "-")
    )

def make_log_filename(date_str: str) -> str:
    """
    YYYY-MM-dd_HH-mm-ss_playlog.csv
    """
    time_part = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%H-%M-%S")
    return f"{date_str}_{time_part}_playlog.csv"

# ========= Island Master =========
ISLAND_HEADER = ["unit_number","island_id","side","pos","edge_type","is_end"]

def load_island_master(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame(columns=ISLAND_HEADER)

    df = pd.read_csv(uploaded)
    df.columns = df.columns.astype(str).str.strip()

    need = set(ISLAND_HEADER)
    if not need.issubset(df.columns):
        st.warning("島マスタの列が不足しています。島情報なしで続行します。")
        return pd.DataFrame(columns=ISLAND_HEADER)

    df = df[ISLAND_HEADER].copy()
    df["unit_number"] = pd.to_numeric(df["unit_number"], errors="coerce")
    df = df[df["unit_number"].notna()].copy()
    df["unit_number"] = df["unit_number"].astype(int)

    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
    df["is_end"] = pd.to_numeric(df["is_end"], errors="coerce").fillna(0).astype(int)
    return df

def join_island(df: pd.DataFrame, island_df: pd.DataFrame) -> pd.DataFrame:
    if island_df is None or island_df.empty:
        return df
    out = df.copy()
    out["unit_number"] = pd.to_numeric(out["unit_number"], errors="coerce")
    out = out[out["unit_number"].notna()].copy()
    out["unit_number"] = out["unit_number"].astype(int)
    return out.merge(island_df, on="unit_number", how="left")



# ========= Sidebar: meta & thresholds =========
# 初期化（スライダー値をsession_stateで持つ）
if "min_games" not in st.session_state:
    st.session_state["min_games"] = 3000
if "max_rb" not in st.session_state:
    st.session_state["max_rb"] = 270.0
if "max_gassan" not in st.session_state:
    st.session_state["max_gassan"] = 180.0

def apply_recommended(machine_name: str):
    rec = RECOMMENDED.get(machine_name)
    if not rec:
        return
    st.session_state["min_games"] = int(rec["min_games"])
    st.session_state["max_rb"] = float(rec["max_rb"])
    st.session_state["max_gassan"] = float(rec["max_gassan"])

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

    # 機種：選択時はプリセット候補。選択されたらおすすめ値を自動セット
    if machine_mode == "選択":
        machine = st.selectbox(
            "machine",
            MACHINE_PRESETS,
            index=0,
            on_change=lambda: apply_recommended(st.session_state["machine_select"]),
            key="machine_select",
        )
        # selectboxの返り値(machine)とkey("machine_select")は同値になります
        machine = st.session_state["machine_select"]
    else:
        machine = st.text_input("machine", value="マイジャグラーV")

    # ---- 補足的におすすめ値を表示（メインではなく控えめに）----
    rec = RECOMMENDED.get(machine)
    with st.expander("おすすめ設定値（補足）", expanded=False):
        if rec:
            st.caption("※ 夕方の続行候補を“厳しめに抽出”するための目安です。店の傾向で調整してください。")
            st.write(f"- 最低総回転: **{rec['min_games']}**")
            st.write(f"- REG上限: **{rec['max_rb']}**")
            st.write(f"- 合算上限: **{rec['max_gassan']}**")
            if st.button("おすすめ値をスライダーに反映", use_container_width=True):
                apply_recommended(machine)
                st.rerun()
        else:
            st.caption("この機種はプリセット未登録です。手動でスライダーを調整してください。")

    st.divider()
    st.header("夕方判定（スライダー）")

    # ★ keyを付けて、session_stateの値を直接使う
    min_games = st.slider(
        "最低 総回転（total_start）",
        0, 10000,
        value=int(st.session_state["min_games"]),
        step=100,
        key="min_games",
    )
    max_rb = st.slider(
        "REG上限（rb_rate）",
        150.0, 600.0,
        value=float(st.session_state["max_rb"]),
        step=1.0,
        key="max_rb",
    )
    max_gassan = st.slider(
        "合算上限（gassan_rate）",
        80.0, 350.0,
        value=float(st.session_state["max_gassan"]),
        step=1.0,
        key="max_gassan",
    )

    top_n = st.number_input("上位N件表示", 1, 200, 30, 1)

# ========= Main UI =========
st.divider()
st.subheader("任意：島マスタアップロード（並び判定に使用）")
island_file = st.file_uploader(
    "島マスタCSV（island.csv）",
    type=["csv"],
    key="island_csv_evening"
)
island_df = load_island_master(island_file)

tab1, tab2, tab3 = st.tabs([
    "入力 → 変換（統一CSV作成）",
    "夕方候補（統一CSVを選択して判定）",
    "実戦ログ（CSVに追記して更新版DL）"
])

with tab1:
    st.subheader("① 入力 → 変換（統一済みCSVを作成してダウンロード）")
    input_mode = st.radio("入力", ["CSVアップロード", "生データ貼り付け（12列）"], horizontal=True)

    df_unified = None  # このタブ内だけで扱う（session_stateは使わない）

    if input_mode == "CSVアップロード":
        uploaded = st.file_uploader("元CSVをアップロード（ヘッダーあり想定）", type=["csv"], key="tab1_csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            df = normalize_columns(df)
            df = ensure_meta_columns(df, date_str, shop, machine)
            df = compute_rates_if_needed(df)

            # HEADERに寄せる（足りない列は作る）
            for c in HEADER:
                if c not in df.columns:
                    df[c] = np.nan
            df_unified = df[HEADER]

    else:
        sample = "478 45 3539 19 11 0 2481 1/186.3 1/321.7 0.0 118.0 449"
        raw_text = st.text_area("台データオンラインの行を貼り付け（複数行OK）", value=sample, height=220, key="tab1_raw")
        if st.button("変換して統一CSVを作る", type="primary", key="tab1_convert"):
            df_unified = parse_raw12(raw_text, date_str, shop, machine)

    if df_unified is None:
        st.info("入力を行うと、ここに統一済みデータが表示され、CSVダウンロードできます。")
    else:
        st.success(f"統一済みデータを作成しました：{len(df_unified)}行")
        st.dataframe(df_unified.head(30), use_container_width=True, hide_index=True)

        filename = make_filename(machine, "original", date_str)

        st.download_button(
            "統一済みCSVをダウンロード",
            data=to_csv_bytes(df_unified),
            file_name=filename,
            mime="text/csv",
            key="tab1_dl_unified"
        )

        st.caption("次に「夕方候補」タブで、この unified.csv をアップロードして判定します。")


with tab2:
    st.subheader("② 夕方候補（統一済みCSVをアップロードして判定）")

    unified_file = st.file_uploader("統一済みCSV（unified.csv）をアップロード", type=["csv"], key="tab2_unified")
    if not unified_file:
        st.info("タブ1でダウンロードした unified.csv をここで選択してください。")
        st.stop()

    # 統一済みCSVを読み込み
    df = pd.read_csv(unified_file)
    df = normalize_columns(df)  # 念のため
    df = compute_rates_if_needed(df)  # 念のため
    for c in HEADER:
        if c not in df.columns:
            df[c] = np.nan
    df = df[HEADER].copy()
    df = join_island(df, island_df)


    # 安定化：判定用の数値列を作る
    df["total_start_num"] = pd.to_numeric(df["total_start"], errors="coerce")
    df["rb_rate_num"] = pd.to_numeric(df["rb_rate"], errors="coerce")
    df["gassan_rate_num"] = pd.to_numeric(df["gassan_rate"], errors="coerce")

    # 夕方候補抽出
    cand = df[
        (df["total_start_num"] >= min_games) &
        (df["rb_rate_num"] <= max_rb) &
        (df["gassan_rate_num"] <= max_gassan)
    ].copy()

    # ===== 並びボーナス（run_bonus）=====
    cand["pos_num"] = pd.to_numeric(cand.get("pos", np.nan), errors="coerce")
    cand["run_bonus"] = 0

    if "island_id" in cand.columns and cand["island_id"].notna().any():
        key_cols = ["island_id", "side"]
        pos_map = (
            cand.dropna(subset=["pos_num"])
            .groupby(key_cols)["pos_num"]
            .apply(lambda s: set(s.astype(int)))
            .to_dict()
        )

        def _run_bonus(row):
            if pd.isna(row["pos_num"]):
                return 0
            k = (row["island_id"], row["side"])
            if k not in pos_map:
                return 0
            p = int(row["pos_num"])
            s = pos_map[k]
            return 1 if ((p - 1 in s) or (p + 1 in s)) else 0

        cand["run_bonus"] = cand.apply(_run_bonus, axis=1)


    if cand.empty:
        st.warning("条件に合う台がありません。閾値を緩めるか、回転数が増えてから再判定してください。")
        st.stop()

    # スコア（REG最重視）
    cand["score"] = (
        (max_rb / cand["rb_rate_num"]) * 70 +
        (cand["total_start_num"] / max(min_games, 1)) * 20 +
        (max_gassan / cand["gassan_rate_num"]) * 10
    )

    cand["score"] = cand["score"] + (cand["run_bonus"] * 1.5)

    cand = cand.sort_values(["rb_rate_num", "total_start_num"], ascending=[True, False])

    show = cand[[
        "date","shop","machine",
        "unit_number","total_start","bb_count","rb_count",
        "bb_rate","rb_rate","gassan_rate",
        "run_bonus","score"
    ]].copy()


    for c in ["bb_rate","rb_rate","gassan_rate","score"]:
        show[c] = pd.to_numeric(show[c], errors="coerce").round(1)

    st.dataframe(show.head(int(top_n)), use_container_width=True, hide_index=True)

    filename = make_filename(machine, "candidates", date_str)

    st.download_button(
        "候補台をCSVでダウンロード",
        data=to_csv_bytes(show),
        file_name=filename,
        mime="text/csv",
        key="tab2_dl_candidates"
    )

with tab3:
    st.subheader("③ 実戦ログ（ローカルCSVに追記 → 更新版をダウンロード）")
    st.caption("※ Streamlit Cloudではローカルファイルを直接書き換えできないため、追記した“更新版CSV”を生成してダウンロードします。")

    # 1) 追記対象のCSVをアップロード
    uploaded_log = st.file_uploader(
        "追記したいログCSVを選択（既存のplay_log.csvなど）",
        type=["csv"],
        key="tab3_log_upload"
    )

    st.divider()

    # 2) 入力フォーム（必要最低限＋任意）
    with st.form("playlog_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            log_date = st.text_input("date（YYYY-MM-DD）", value=date_str)
            log_shop = st.text_input("shop", value=shop)
        with col2:
            log_machine = st.text_input("machine", value=machine)
            unit_number = st.number_input("unit_number（台番号）", min_value=0, step=1, value=0)
        with col3:
            play_games = st.number_input("play_games（自分が回したG数）", min_value=0, step=10, value=0)

        col4, col5, col6 = st.columns(3)
        with col4:
            start_time = st.text_input("start_time（例 18:10）", value="")
        with col5:
            end_time = st.text_input("end_time（例 20:45）", value="")
        with col6:
            stop_reason = st.selectbox(
                "stop_reason（ヤメ理由）",
                ["", "閉店", "REG悪化", "合算悪化", "資金切れ", "他に移動", "その他"]
            )

        col7, col8, col9 = st.columns(3)
        with col7:
            invest = st.number_input("invest_medals（投資枚）", min_value=0, step=50, value=0)
        with col8:
            payout = st.number_input("payout_medals（回収枚）", min_value=0, step=50, value=0)
        with col9:
            profit = int(payout - invest)
            st.metric("profit_medals（収支枚）", profit)

        memo = st.text_area("memo（任意）", value="", height=100)

        submit = st.form_submit_button("この内容で追記用データを作成", type="primary")

    # 3) 追記 → 更新版CSVダウンロード
    if submit:
        if uploaded_log is None:
            st.error("先に「追記したいログCSV」を選択してください。")
            st.stop()

        new_row = {
            "created_at": datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d %H:%M:%S"),
            "date": log_date,
            "shop": log_shop,
            "machine": log_machine,
            "unit_number": int(unit_number),
            "start_time": start_time,
            "end_time": end_time,
            "invest_medals": int(invest),
            "payout_medals": int(payout),
            "profit_medals": int(profit),
            "play_games": int(play_games),
            "stop_reason": stop_reason,
            "memo": memo,
        }

        out_bytes = append_row_to_uploaded_csv(uploaded_log.getvalue(), new_row)
        out_name = make_log_filename(log_date)

        st.success("追記済みの更新版CSVを作成しました。下のボタンからダウンロードしてください。")
        st.download_button(
            "追記済みログCSVをダウンロード（更新版）",
            data=out_bytes,
            file_name=out_name,
            mime="text/csv",
            key="tab3_log_download"
        )

        # 参考：追記後のプレビュー
        st.divider()
        st.markdown("#### 追記後プレビュー（末尾5行）")
        preview_df = pd.read_csv(io.BytesIO(out_bytes))
        st.dataframe(preview_df.tail(5), use_container_width=True, hide_index=True)

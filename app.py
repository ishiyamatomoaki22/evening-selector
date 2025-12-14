import re
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date

HEADER = [
    "date","shop","machine",
    "unit_number","start_games","total_start","bb_count","rb_count","art_count","max_medals",
    "bb_rate","rb_rate","art_rate","gassan_rate","prev_day_end"
]

# あなた用の候補（好きに増やしてOK）
SHOP_PRESETS = ["武蔵境", "吉祥寺", "三鷹", "国分寺", "新宿", "渋谷"]
MACHINE_PRESETS = [
    "マイジャグラーV",
    "アイムジャグラーEX",
    "ファンキージャグラー2",
    "ゴーゴージャグラー3",
]

st.set_page_config(page_title="台データ → CSV整形ツール", layout="wide")
st.title("台データオンライン（コピペ）→ CSV整形ツール")
st.caption("12列固定の行を貼り付けて、日付・店名・機種を補完し、CSVヘッダー形式で出力します。")

def parse_rate_token(tok: str) -> float:
    """ '1/186.3' -> 186.3 , '186.3' -> 186.3 """
    if tok is None:
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

def parse_lines(text: str, date_str: str, shop: str, machine: str) -> pd.DataFrame:
    rows = []
    for line_no, line in enumerate((text or "").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s+", line)
        if len(parts) != 12:
            raise ValueError(f"{line_no}行目：列数が合いません（期待12 / 実際{len(parts)}）: {line}")

        unit_number, start_games, total_start, bb_count, rb_count, art_count, max_medals, bb_rate, rb_rate, art_rate, gassan_rate, prev_day_end = parts

        row = {
            "date": date_str,
            "shop": shop,
            "machine": machine,
            "unit_number": int(unit_number),
            "start_games": float(start_games),
            "total_start": float(total_start),
            "bb_count": float(bb_count),
            "rb_count": float(rb_count),
            "art_count": float(art_count),
            "max_medals": float(max_medals),
            "bb_rate": parse_rate_token(bb_rate),
            "rb_rate": parse_rate_token(rb_rate),
            "art_rate": parse_rate_token(art_rate),
            "gassan_rate": parse_rate_token(gassan_rate),
            "prev_day_end": float(prev_day_end),
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=HEADER)
    return df

def validate_meta(shop: str, machine: str):
    if not shop.strip():
        st.error("店名（shop）が空です。入力または選択してください。")
        return False
    if not machine.strip():
        st.error("機種（machine）が空です。入力または選択してください。")
        return False
    return True

# ================= UI =================
left, right = st.columns([1, 2], vertical_alignment="top")

with left:
    st.subheader("補完する情報（UIから選択 or 入力）")

    # 日付：カレンダー
    d = st.date_input("日付（date）", value=date.today())
    date_str = d.strftime("%Y-%m-%d")

    st.markdown("### 店名（shop）")
    use_custom_shop = st.checkbox("店名を手入力する", value=False)
    if use_custom_shop:
        shop = st.text_input("店名を入力", value="武蔵境")
    else:
        shop = st.selectbox("店名を選択", options=SHOP_PRESETS, index=0)

    st.markdown("### 機種（machine）")
    use_custom_machine = st.checkbox("機種を手入力する", value=False)
    if use_custom_machine:
        machine = st.text_input("機種を入力", value="マイジャグラーV")
    else:
        machine = st.selectbox("機種を選択", options=MACHINE_PRESETS, index=0)

    st.divider()

    st.subheader("貼り付け（取得データ：12列固定）")
    sample = "478 45 3539 19 11 0 2481 1/186.3 1/321.7 0.0 118.0 449"
    raw_text = st.text_area("空白区切りの行を複数貼ってOK", value=sample, height=260)

    col_a, col_b = st.columns(2)
    with col_a:
        convert = st.button("CSVに変換", type="primary")
    with col_b:
        clear = st.button("貼り付け欄をクリア")

    if clear:
        st.session_state["raw_text_clear"] = True

with right:
    # クリアボタン対応（簡易）
    if st.session_state.get("raw_text_clear"):
        st.session_state["raw_text_clear"] = False
        st.info("貼り付け欄のクリアは、テキストエリアを手動で空にしてください（Streamlit仕様）")

    st.subheader("変換結果")
    st.write("出力ヘッダー：")
    st.code(",".join(HEADER))

    if convert:
        if not validate_meta(shop, machine):
            st.stop()

        try:
            df = parse_lines(raw_text, date_str, shop, machine)
            st.success(f"変換OK：{len(df)}行")

            st.dataframe(df, use_container_width=True, hide_index=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "CSVをダウンロード",
                data=csv_bytes,
                file_name=f"{shop}_{machine}_{date_str}.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(str(e))
            st.info("よくある原因：コピペ時に列が欠けている / 余計な文字が混ざっている / 空白区切りが崩れている")

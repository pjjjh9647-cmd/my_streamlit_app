# -*- coding: utf-8 -*-
# 파일명: streamlit_gunwi_results_viewer.py
import re
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="군위 품종별 회귀결과 뷰어", layout="wide")
st.title("군위 품종별 회귀결과 뷰어")

# ------------------------------------------------------------
# 0) 파서 유틸 (OLS_요약.txt → 레코드)
# ------------------------------------------------------------
# 품종 블록: "=== 품종: 홍로 ===" ... (여러 타깃 블록) ...
OLS_BLOCK_RE = re.compile(
    r"===\s*품종:\s*(?P<cultivar>.+?)\s*===\s*(?P<body>.*?)(?=(?:\n===\s*품종:|\Z))",
    flags=re.DOTALL
)

# 타깃 블록: "--- 타깃: 과중 ---" ~ "모델식: ..."
TARGET_BLOCK_RE = re.compile(
    r"---\s*타깃:\s*(?P<target>.+?)\s*---\s*"
    r"n=(?P<n>\d+),\s*R²=(?P<R2>[0-9\.\-nan]+),\s*adj\.R²=(?P<R2adj>[0-9\.\-nan]+),\s*"
    r"F=(?P<F>[0-9\.\-nan]+),\s*p\(F\)=(?P<pF>[0-9eE\.\-nan]+)\s*"
    r"계수\(유의성\):\s*(?P<params>.+?)\s*"
    r"p-values:\s*(?P<pvals>.+?)\s*"
    r"모델식:\s*(?P<formula>.+?)\s*$",
    flags=re.DOTALL | re.MULTILINE
)

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _parse_series_block(txt: str) -> pd.DataFrame:
    """
    statsmodels Series.to_string() 같은 형식:
      const        12.34
      x_m05        -0.12
    → feature / value 로 변환
    """
    lines = [ln.strip() for ln in str(txt).strip().splitlines() if ln.strip()]
    idx, val = [], []
    for ln in lines:
        m = re.match(r"(.+?)\s+([\-0-9\.eE]+)$", ln)
        if m:
            idx.append(m.group(1).strip())
            val.append(_to_float(m.group(2)))
    return pd.DataFrame({"feature": idx, "value": val})

def parse_ols_text(text: str):
    """OLS_요약.txt 전체를 파싱하여 [레코드 dict] 리스트 반환"""
    out_records = []
    for m in OLS_BLOCK_RE.finditer(text):
        cultivar = m.group("cultivar").strip()
        body = m.group("body")
        for t in TARGET_BLOCK_RE.finditer(body):
            rec = {
                "cultivar": cultivar,
                "target":   t.group("target").strip(),
                "n":        int(t.group("n")),
                "R2":       _to_float(t.group("R2")),
                "R2adj":    _to_float(t.group("R2adj")),
                "F":        _to_float(t.group("F")),
                "pF":       t.group("pF").strip(),
                "formula":  t.group("formula").strip(),
                "params_df": _parse_series_block(t.group("params")),
                "pvals_df":  _parse_series_block(t.group("pvals")),
            }
            out_records.append(rec)
    return out_records

# ------------------------------------------------------------
# 1) OUTDIR 선택 UI
# ------------------------------------------------------------
with st.sidebar:
    st.subheader("결과 폴더 선택")
    # 여기를 본인 OUTDIR 로 기본값 세팅
    default_out = r"C:\Users\User\Desktop\mba\환경데이터\_ANALYSIS_OUT"
    outdir_str = st.text_input("OUTDIR 경로", default_out)
    run = st.button("불러오기")

if not run:
    st.info("좌측에서 OUTDIR 경로를 확인한 뒤 [불러오기]를 누르세요.")
    st.stop()

OUTDIR = Path(outdir_str)
cv_root = OUTDIR / "_BY_CULTIVAR"
if not cv_root.exists():
    st.error(f"폴더가 없습니다: {cv_root}")
    st.stop()

# ------------------------------------------------------------
# 2) 품종 폴더들 스캔
# ------------------------------------------------------------
packs = []
for sub in sorted(cv_root.iterdir()):
    if not sub.is_dir():
        continue
    ols_path = sub / "OLS_요약.txt"
    if not ols_path.exists():
        continue
    text = ols_path.read_text(encoding="utf-8", errors="ignore")
    records = parse_ols_text(text)
    if not records:
        continue
    pack = {
        "cultivar": sub.name,
        "ols_records": records,
        "sel_df": (pd.read_csv(sub / "선택변수_요약.csv", encoding="utf-8")
                   if (sub / "선택변수_요약.csv").exists() else pd.DataFrame()),
        "corr_tg_df": (pd.read_csv(sub / "상관_타깃vs기상.csv", encoding="utf-8")
                       if (sub / "상관_타깃vs기상.csv").exists() else pd.DataFrame()),
        "corr_all_df": (pd.read_csv(sub / "상관_전체_테이블.csv", encoding="utf-8")
                        if (sub / "상관_전체_테이블.csv").exists() else pd.DataFrame()),
    }
    packs.append(pack)

if not packs:
    st.warning("표시할 품종 폴더(OLS_요약.txt 포함)가 없습니다.")
    st.stop()

# ------------------------------------------------------------
# 3) 렌더링
# ------------------------------------------------------------
cultivars = [p["cultivar"] for p in packs]
tab_cvs = st.tabs(cultivars)

def show_param_table(params_df: pd.DataFrame, pvals_df: pd.DataFrame):
    df = params_df.merge(pvals_df, on="feature", how="outer", suffixes=("_coef","_p"))
    df = df.rename(columns={"value_coef":"coef","value_p":"p_value"})
    order = {"const": -1}
    df["order"] = df["feature"].map(order).fillna(1e6)
    df = df.sort_values(["order","feature"]).drop(columns="order")
    st.dataframe(df, use_container_width=True)

for pack, tab in zip(packs, tab_cvs):
    with tab:
        st.markdown(f"### 품종: **{pack['cultivar']}**")

        if not pack["sel_df"].empty:
            st.markdown("#### 선택변수 요약")
            st.dataframe(pack["sel_df"], use_container_width=True)

        if not pack["corr_tg_df"].empty:
            st.markdown("#### 타깃-기상 상관 (상위 |r|)")
            cdf = pack["corr_tg_df"].copy()
            if "Unnamed: 0" in cdf.columns:
                cdf = cdf.rename(columns={"Unnamed: 0": "target"}).set_index("target")
            show = (cdf.stack()
                      .rename("corr")
                      .reset_index()
                      .rename(columns={"level_1":"feature"})
                      .sort_values("corr", key=lambda s: s.abs(), ascending=False)
                      .head(50))
            st.dataframe(show, use_container_width=True)

        st.markdown("#### OLS 결과")
        records = pack["ols_records"]
        targets = [r["target"] for r in records]
        ttabs = st.tabs(targets)

        for rec, ttab in zip(records, ttabs):
            with ttab:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("n", rec["n"])
                c2.metric("R²", None if np.isnan(rec["R2"]) else round(rec["R2"], 3))
                c3.metric("adj.R²", None if np.isnan(rec["R2adj"]) else round(rec["R2adj"], 3))
                c4.metric("F", None if np.isnan(rec["F"]) else round(rec["F"], 3))
                c5.metric("p(F)", rec["pF"])

                st.markdown("##### 회귀식 (복붙용)")
                st.code(rec["formula"], language="text")

                st.markdown("##### 계수 / p-value")
                show_param_table(rec["params_df"], rec["pvals_df"])

                st.caption("※ y_true/y_pred 그래프가 필요하면 병합 CSV를 추가로 읽어 예측값을 계산하는 로직을 붙일 수 있습니다.")

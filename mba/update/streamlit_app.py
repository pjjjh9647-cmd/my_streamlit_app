# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine, text
from datetime import datetime

from pipeline_core import (
    COL_REGION, build_monthly_features, _norm_cultivar_name,
    load_latest_model
)

# --------------------------
# 설정
# --------------------------
REGISTRY_ROOT = Path(r"C:\Users\User\Desktop\orchard_models_registry")
USE_DB = True
DB_URL = "mssql+pyodbc://USER:PASS@SERVER/DBNAME?driver=ODBC+Driver+17+for+SQL+Server"
SQL_ENV_RECENT = """
SELECT 지역명, 일자, 평균기온, 최고기온, 최저기온, 습도, 강우량, 일사량, 결로시간, 평균풍속, 최대풍속
FROM environment_daily
WHERE 일자 BETWEEN DATEADD(month, -12, GETDATE()) AND GETDATE()
"""

st.set_page_config(page_title="과수 생육·품질 예측 대시보드", layout="wide")
st.title("과수 생육·품질 예측 대시보드")

tab_pred, tab_train = st.tabs(["예측 보기", "수동 재학습"])

# --------------------------
# 유틸
# --------------------------
def list_models():
    rows = []
    if not REGISTRY_ROOT.exists():
        return pd.DataFrame(columns=["cultivar","target","version_path"])
    for cv in REGISTRY_ROOT.iterdir():
        if not cv.is_dir():
            continue
        for tgt in cv.iterdir():
            if not tgt.is_dir():
                continue
            vers = sorted([p for p in tgt.iterdir() if p.is_dir() and p.name.startswith("v_")])
            if vers:
                rows.append({"cultivar": cv.name, "target": tgt.name, "version_path": str(vers[-1])})
    return pd.DataFrame(rows)

def get_latest_model(cultivar, target):
    model_dir = REGISTRY_ROOT / cultivar / target
    return load_latest_model(model_dir)

# --------------------------
# 예측 탭
# --------------------------
with tab_pred:
    st.subheader("최신 등록 모델")
    models_df = list_models()
    if models_df.empty:
        st.info("등록된 모델이 없습니다. [수동 재학습] 탭에서 먼저 학습하세요.")
    else:
        st.dataframe(models_df, use_container_width=True)

    st.markdown("예측 옵션을 선택하세요.")
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        sel_cv = st.selectbox("품종(레지스트리 기준)", sorted(models_df["cultivar"].unique()))
    with col2:
        sel_tgt = st.selectbox("타깃", sorted(models_df[models_df["cultivar"]==sel_cv]["target"].unique()))
    with col3:
        pred_mode = st.radio("입력 방식", ["DB 최근 12개월 집계 사용", "CSV 업로드(월별 특징 완성본)"], horizontal=True)

    model, meta = get_latest_model(sel_cv, sel_tgt)
    if model is None:
        st.warning("선택한 모델이 없습니다.")
    else:
        st.write("모델 메타:", meta)

        if pred_mode == "DB 최근 12개월 집계 사용" and USE_DB:
            eng = create_engine(DB_URL)
            env = pd.read_sql(text(SQL_ENV_RECENT), eng)
            env["일자"] = pd.to_datetime(env["일자"], errors="coerce")
            env["year"]  = env["일자"].dt.year
            env["month"] = env["일자"].dt.month

            # meta의 env_cols 기준으로 집계
            env_cols = meta.get("env_cols", [])
            monthly = build_monthly_features(env, env_cols)
            # 최신 연도만 예시로
            latest_years = sorted(monthly["year"].dropna().unique())[-3:]  # 최근 3개년
            pred_in = monthly[monthly["year"].isin(latest_years)].copy()

            # 모델에 필요한 특징 컬럼만
            feats = meta.get("features", [])
            X = pred_in[feats].astype(float).fillna(pred_in[feats].mean())
            yhat = model.predict(sm.add_constant(X))  # statsmodels OLS
            out = pred_in[[COL_REGION,"year"]].copy()
            out["prediction"] = yhat
            st.subheader("최근 연도별 예측")
            st.dataframe(out.sort_values(["지역명","year"]), use_container_width=True)

        elif pred_mode == "CSV 업로드(월별 특징 완성본)":
            st.caption("컬럼 예: 지역명, year, <특징들(meta['features'])>")
            up = st.file_uploader("월별 특징 CSV 업로드", type=["csv"])
            if up is not None:
                df = pd.read_csv(up)
                feats = meta.get("features", [])
                missing = [c for c in feats if c not in df.columns]
                if missing:
                    st.error(f"필수 특징 컬럼 없음: {missing}")
                else:
                    X = df[feats].astype(float).fillna(df[feats].mean())
                    yhat = model.predict(sm.add_constant(X))
                    res = df[[COL_REGION,"year"]].copy() if COL_REGION in df.columns and "year" in df.columns else pd.DataFrame(index=df.index)
                    res["prediction"] = yhat
                    st.dataframe(res, use_container_width=True)

# --------------------------
# 수동 재학습 탭
# --------------------------
with tab_train:
    st.subheader("재학습 실행")
    st.write("학습 스크립트를 실행하여 새 데이터를 반영한 모델을 등록합니다. 실행 후 레지스트리 최신 버전이 갱신됩니다.")
    if st.button("재학습 실행"):
        # 외부 스크립트를 서브프로세스로 호출 (윈도우 예시)
        import subprocess, sys
        trainer = str((Path(__file__).parent / "train_and_register.py").resolve())
        try:
            out = subprocess.run([sys.executable, trainer], capture_output=True, text=True, timeout=1800)
            st.code(out.stdout or "(no stdout)")
            if out.stderr:
                st.error(out.stderr)
            st.success("재학습 프로세스가 종료되었습니다.")
        except Exception as e:
            st.error(f"실행 오류: {e}")

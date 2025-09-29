# -*- coding: utf-8 -*-
# 파일명: streamlit_regression_viewer.py

import io
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from importlib.machinery import SourceFileLoader
from datetime import datetime

# ------------------------------------------------------------
# 1) 사용자 환경 설정
#    - 아래 MODULE_PATH를 본인 파일 경로로 변경
#      예) r"C:\newvariety\환경-과실데이터\회귀식(홍로4-8, 후지4-10).py"
# ------------------------------------------------------------
MODULE_PATH = r"C:\newvariety\환경-과실데이터\회귀식(홍로4-8, 후지4-10).py"

st.set_page_config(page_title="회귀식 요약/시각화 뷰어", layout="wide")
st.title("회귀식 요약·시각화 뷰어")

with st.sidebar:
    st.subheader("모듈·데이터 경로")
    module_path = st.text_input("회귀식 모듈 경로", MODULE_PATH)
    env_path = st.text_input("기상데이터 엑셀 경로", r"C:\Users\User\Desktop\mba\환경데이터\기상데이터_통합.xlsx")
    fruit_path = st.text_input("과실데이터 엑셀 경로", r"C:\Users\User\Desktop\mba\과실데이터\과실데이터_통합.xlsx")
    run_btn = st.button("모델 실행")

# ------------------------------------------------------------
# 2) 외부 모듈 로드
#    - 파일명이 특수문자/공백이 있어도 SourceFileLoader로 안전 로드
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_regression_module(path: str):
    loader = SourceFileLoader("regmod", path)
    mod = loader.load_module()
    return mod

def _is_num(s):
    return pd.api.types.is_numeric_dtype(s)

def plot_actual_pred(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Actual vs Predicted"):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.6)
    minv = float(np.nanmin([y_true.min(), y_pred.min()]))
    maxv = float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([minv, maxv], [minv, maxv])
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    st.pyplot(fig)

def plot_residuals(resid: np.ndarray, title: str = "Residuals Histogram"):
    fig, ax = plt.subplots()
    ax.hist(resid, bins=20)
    ax.set_title(title)
    st.pyplot(fig)

def make_download_excel(mod, results, include_merged=True):
    # results: 리스트[(sheet_prefix, model, eq, merged, corr_s, summary, features, lasso_coef)]
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for sheet_prefix, model, eq, merged, corr_s, summary, features, lasso_coef in results:
            # 모듈의 export_to_sheets를 그대로 사용
            mod.INCLUDE_MERGED = include_merged
            mod.export_to_sheets(writer, sheet_prefix, model, eq, merged, corr_s, summary, features, lasso_coef)
        # 서식
        wb = writer.book
        bold = wb.add_format({"bold": True})
        for sh in writer.sheets.values():
            sh.set_row(0, None, bold)
            sh.freeze_panes(1, 1)
    bio.seek(0)
    return bio

# ------------------------------------------------------------
# 3) 실행 로직
#    - 모듈의 run_for_target()을 사용하되,
#      입력 데이터는 이 앱에서 직접 읽어 전달
# ------------------------------------------------------------
if run_btn:
    try:
        with st.spinner("모듈 로드 중..."):
            mod = load_regression_module(module_path)

        # 필수 상수
        COL_REGION = mod.COL_REGION
        COL_DATE   = mod.COL_DATE

        with st.spinner("데이터 로드 중..."):
            env = pd.read_excel(env_path)
            fruit = pd.read_excel(fruit_path)

        # 날짜 파싱 및 연/월 생성
        if COL_DATE not in env.columns:
            st.error(f"기상데이터에 {COL_DATE} 컬럼이 없습니다")
            st.stop()
        env = env.copy()
        env[COL_DATE] = pd.to_datetime(env[COL_DATE], errors="coerce")
        if env[COL_DATE].isna().all():
            st.error("기상 일자 파싱 실패")
            st.stop()
        env["year"] = env[COL_DATE].dt.year
        env["month"] = env[COL_DATE].dt.month

        # 환경 후보 컬럼은 모듈의 CANDIDATE_COLS 중 실제 존재하는 것만
        env_cols = [c for c in mod.CANDIDATE_COLS if c in env.columns]
        if not env_cols:
            st.error("사용 가능한 기상 변수(CANDIDATE_COLS)가 없습니다")
            st.stop()

        # 타깃 후보: 과실 데이터 내 숫자형 컬럼에서 KEY_COLS 제외
        num_cols = fruit.select_dtypes(include='number').columns.tolist()
        targets_all = [c for c in num_cols if c not in mod.KEY_COLS]
        if not targets_all:
            st.error("과실 타깃 후보(숫자형)가 없습니다")
            st.stop()

        # 품종 처리
        if "품종" in fruit.columns and mod.GROUP_BY_CULTIVAR:
            cultivars = fruit["품종"].dropna().astype(str).map(mod._norm_cultivar_name).unique().tolist()
        else:
            cultivars = [None]

        # 사용자 선택 UI
        st.subheader("실행 옵션")
        cols = st.columns(2)
        with cols[0]:
            sel_cultivars = st.multiselect("품종 선택", options=[c if c is not None else "(통합)" for c in cultivars],
                                           default=[c if c is not None else "(통합)" for c in cultivars])
        with cols[1]:
            sel_targets = st.multiselect("타깃 지표 선택", options=targets_all, default=targets_all)

        # 선택 반영
        def _cv_norm(label):
            return None if label == "(통합)" else label
        sel_cultivars = [_cv_norm(c) for c in sel_cultivars]

        results = []
        tabs = []
        tab_titles = []

        # 실행
        for cv in sel_cultivars:
            fsub = fruit if cv is None else fruit[fruit["품종"].astype(str).map(mod._norm_cultivar_name) == cv]
            # 월 범위 라벨
            if cv is not None and cv in mod.CULTIVAR_MONTH_WINDOWS:
                mm = mod.CULTIVAR_MONTH_WINDOWS[cv]
                mm_str = f"(m{min(mm):02d}-m{max(mm):02d})"
            else:
                mm_str = "(m01-m12)"

            for tgt in sel_targets:
                try:
                    model, eq, merged, corr_s, summary, features, lasso_coef = mod.run_for_target(
                        env, fsub, env_cols, tgt, cultivar_name=cv
                    )
                    sheet_prefix = f"{tgt}{'' if cv is None else '_'+cv}{mm_str}"
                    results.append((sheet_prefix, model, eq, merged, corr_s, summary, features, lasso_coef))
                    tab_titles.append(sheet_prefix)
                except Exception as e:
                    st.warning(f"[SKIP] {tgt}{'' if cv is None else '_'+cv} 실패: {e}")

        if not results:
            st.error("생성된 결과가 없습니다")
            st.stop()

        # 탭 구성
        tabs = st.tabs(tab_titles)
        for (sheet_prefix, model, eq, merged, corr_s, summary, features, lasso_coef), tab in zip(results, tabs):
            with tab:
                st.markdown(f"#### {sheet_prefix}")
                # 요약
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("샘플수", summary["n_samples"])
                c2.metric("R²(in-sample)", summary["r2_in_sample"])
                c3.metric("RMSE(in-sample)", summary["rmse_in_sample"])
                c4.metric("Lasso α", round(summary["lasso_alpha"], 6))

                # 회귀식
                st.markdown("##### 회귀식")
                st.code(eq, language="text")

                # 계수표
                st.markdown("##### 계수표")
                params = model.params
                pvals = model.pvalues
                conf = model.conf_int()
                coef_df = pd.DataFrame({
                    "feature": params.index,
                    "coef": params.values,
                    "p_value": pvals.values,
                    "ci_low": conf[0].values,
                    "ci_high": conf[1].values
                })
                # 선택 피처 우선 정렬
                order = {"const": -1} | {f:i for i,f in enumerate(features)}
                coef_df["order"] = coef_df["feature"].map(order).fillna(1e6)
                coef_df = coef_df.sort_values(["order","feature"]).drop(columns="order")
                st.dataframe(coef_df, use_container_width=True)

                # 상관계수 상위
                st.markdown("##### 상관계수 순위")
                corr_df = (corr_s.to_frame("corr").reset_index()
                           .rename(columns={"index":"feature"})
                           .sort_values("corr", key=lambda s: s.abs(), ascending=False))
                st.dataframe(corr_df.head(50), use_container_width=True)

                # 예측 성능 그래프
                st.markdown("##### 성능 그래프")
                y_true = model.model.endog
                y_pred = model.predict(model.model.exog)
                resid = y_true - y_pred
                plot_actual_pred(y_true, y_pred, title=f"{sheet_prefix}  Actual vs Predicted")
                plot_residuals(resid, title=f"{sheet_prefix}  Residuals")

                # 병합표 일부 미리보기
                with st.expander("병합 데이터 미리보기"):
                    st.dataframe(merged.head(30), use_container_width=True)

        # 엑셀 다운로드
        st.markdown("---")
        st.subheader("엑셀로 내보내기")
        incl_merged = st.checkbox("Merged 시트 포함", value=True)
        if st.button("엑셀 파일 생성"):
            with st.spinner("엑셀 생성 중..."):
                bio = make_download_excel(mod, results, include_merged=incl_merged)
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                label="엑셀 다운로드",
                data=bio,
                file_name=f"통합회귀_요약_{ts}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except FileNotFoundError as e:
        st.error(f"파일을 찾을 수 없습니다: {e}")
    except Exception as e:
        st.exception(e)

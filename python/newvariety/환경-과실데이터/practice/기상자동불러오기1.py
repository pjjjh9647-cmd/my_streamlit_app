import streamlit as st
import pandas as pd
import requests
from io import StringIO
import numpy as np

# ===== 환경 변수 =====
ENV_COLS = ["평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]

st.title("🍎 사과 과실 특성 예측기 (기상데이터 자동 불러오기)")

# ===== 사용자 입력 =====
region = st.selectbox("지역 선택", ["거창", "영주", "충주", "장수"])  # 필요시 확장
year = st.number_input("연도", value=2024, min_value=2010, max_value=2030)
month = st.selectbox("월", list(range(1,13)))

if st.button("기상데이터 불러오기"):
    # ===== 실제 데이터 요청 =====
    url = "https://fruit.nihhs.go.kr/apple/aws/awsSearch.do"
    payload = {
        "schYear": year,
        "schMonth": f"{month:02d}",
        "schArea": region,
    }
    resp = requests.post(url, data=payload)

    if resp.status_code == 200:
        try:
            # 표 HTML 파싱
            tables = pd.read_html(resp.text)
            df = tables[0]
            st.success("데이터 불러오기 성공!")
            st.dataframe(df.head())

            # ===== 월별 집계 =====
            # (사이트 구조에 따라 컬럼명 매칭 필요)
            df["일자"] = pd.to_datetime(df["일자"], errors="coerce")
            df["연도"] = df["일자"].dt.year
            df["월"] = df["일자"].dt.month

            agg_map = {
                "평균기온":"mean","최고기온":"mean","최저기온":"mean","습도":"mean",
                "강우량":"sum","일사량":"sum","결로시간":"sum","평균풍속":"mean","최대풍속":"mean"
            }
            env_m = df.groupby(["연도","월"], as_index=False).agg(agg_map)

            st.subheader("월별 요약")
            st.dataframe(env_m)

            # ===== OLS 회귀식 기반 예측 =====
            coef_df = pd.read_csv(r"C:\Users\User\Desktop\분석결과_월단위\02_회귀분석_OLS\회귀계수.csv")
            targets = coef_df["타깃"].unique()

            st.subheader("예측 결과 (OLS)")
            for target in targets:
                coef_sub = coef_df[(coef_df["타깃"]==target) & (coef_df["월"]==month)]
                if coef_sub.empty:
                    continue

                coefs = {row["변수"]: row["계수"] for _, row in coef_sub.iterrows()}
                intercept = coefs.get("const", 0.0)

                pred = intercept
                for col in ENV_COLS:
                    if col in coefs and col in env_m.columns:
                        pred += coefs[col].mean() * env_m[col].values[0]

                st.write(f"- {target}: {pred:.2f}")

        except Exception as e:
            st.error(f"파싱 오류: {e}")
    else:
        st.error(f"데이터 요청 실패 (status {resp.status_code})")

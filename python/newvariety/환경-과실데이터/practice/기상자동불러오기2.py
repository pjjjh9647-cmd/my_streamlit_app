# -*- coding: utf-8 -*-
# 파일명 예: 기상자동불러오기_app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# =========================
# Streamlit 기본 설정
# =========================
st.set_page_config(page_title="🍎 사과 기상 통계 수집기", layout="wide")
st.title("🍎 사과 기상 통계 수집기 (NIHHS awsStat)")

# =========================
# 지역 코드 매핑 (사이트 select의 value와 동일)
# =========================
AREA_CODE = {
    "경기화성": "324",
    "경북영주": "331",
    "경북청송": "332",
    "대구군위": "333",
    "경남거창": "334",
    "전북장수": "335",
    "경기포천": "337",
    "충북충주": "338",
}

# =========================
# 유틸 함수
# =========================
def _ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _agg_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    일별/순별 데이터를 월별로 집계.
    - 평균: 평균기온/최고기온/최저기온/습도/평균풍속/최대풍속
    - 합계: 강우량/일사량/결로시간
    """
    if not {"연도", "월"}.issubset(df.columns):
        return df.copy()

    agg_map = {
        "평균기온":"mean","최고기온":"mean","최저기온":"mean","습도":"mean",
        "강우량":"sum","일사량":"sum","결로시간":"sum",
        "평균풍속":"mean","최대풍속":"mean"
    }
    use_cols = {k:v for k,v in agg_map.items() if k in df.columns}
    out = df.groupby(["연도","월"], as_index=False).agg(use_cols)
    # 정렬
    out = out.sort_values(["연도","월"]).reset_index(drop=True)
    return out

# =========================
# 원본 AJAX 호출 → JSON → DataFrame 변환
# =========================
@st.cache_data(show_spinner=False, ttl=300)
def fetch_aws_stat(region_name: str, stat_gb_code: str, s_ym: str, e_ym: str) -> dict:
    """
    region_name: 화면 표시용 지역명 (areaName으로 전송)
    stat_gb_code: "A"(기간), "B"(일별), "C"(순별), "D"(월별)
    s_ym, e_ym: "YYYY-MM"
    """
    session = requests.Session()
    # 쿠키/세션 확보용 페이지 GET
    session.get("https://fruit.nihhs.go.kr/apple/aws/awsStat.do", timeout=20)

    # 폼 구성 (사이트 JS serialize와 동일 키)
    form = {
        "pageIndex": "1",
        "searchNum": "0",
        "areaName": region_name,
        "areaCode": AREA_CODE.get(region_name, region_name),  # 혹시 코드 직접 입력해도 동작
        "statmethod": stat_gb_code,
    }

    if stat_gb_code == "A":  # 기간 (일자 범위)
        form["wetherDtBgn"] = f"{s_ym}-01"
        form["wetherDtEnd"] = f"{e_ym}-30"
    elif stat_gb_code in ("B", "C"):  # 일별/순별은 단일 '통계월'
        form["wetherDtM"] = s_ym
    elif stat_gb_code == "D":  # 월별 (시작/종료 월)
        form["wetherDtBgn2"] = s_ym
        form["wetherDtEnd2"] = e_ym
        # 서버 JS가 내부에서 Bgn/End도 함께 참조하므로 같이 넣어준다
        form["wetherDtBgn"] = s_ym
        form["wetherDtEnd"]  = e_ym

    resp = session.post(
        "https://fruit.nihhs.go.kr/apple/aws/awsStatList.do",
        data=form, timeout=30,
        headers={"Referer": "https://fruit.nihhs.go.kr/apple/aws/awsStat.do"}
    )
    resp.raise_for_status()

    # JSON 아닌 경우를 대비
    try:
        return resp.json()
    except Exception:
        return {"error": "JSON 파싱 실패", "raw": resp.text[:5000]}

def json_to_dataframe(payload: dict) -> pd.DataFrame:
    """
    사이트에서 내려주는 JSON 구조를 표로 변환.
    statmethod:
      - A: 기간(일자범위) → statsDt, dalyWetherXXXX
      - B: 일별 → wetherDt, wetherXXXX
      - C: 순별(상/중/하순) → wetherDt=0/1/2, wetherXXXX
      - D: 월별 → wetherDt=YYYY-MM, wetherXXXX
    """
    if not payload or "result" not in payload:
        return pd.DataFrame()

    res = payload.get("result", [])
    if len(res) == 0:
        return pd.DataFrame()

    method = payload.get("mainAwsVO", {}).get("statmethod")

    if method == "A":
        df = pd.DataFrame(res)[[
            "statsDt","dalyWetherAvrgTp","dalyWetherMxmmTp","dalyWetherMummTp",
            "dalyWetherAvrgHd","dalyWetherTtalRainqy","dalyWetherMxmmSolradqy",
            "dalyWetherMxmmCondenstime","dalyWetherAvrgWs","dalyWetherMxmmWs"
        ]].rename(columns={
            "statsDt":"일자","dalyWetherAvrgTp":"평균기온","dalyWetherMxmmTp":"최고기온",
            "dalyWetherMummTp":"최저기온","dalyWetherAvrgHd":"습도","dalyWetherTtalRainqy":"강우량",
            "dalyWetherMxmmSolradqy":"일사량","dalyWetherMxmmCondenstime":"결로시간",
            "dalyWetherAvrgWs":"평균풍속","dalyWetherMxmmWs":"최대풍속"
        })
    elif method == "B":
        df = pd.DataFrame(res)[[
            "wetherDt","wetherAvgTp","wetherMaxTp","wetherMinTp",
            "WetherAvgHd","wetherMaxRainqy","wetherMaxSolradqy",
            "wetherAvgWs","wetherMaxWs"
        ]].rename(columns={
            "wetherDt":"일자","wetherAvgTp":"평균기온","wetherMaxTp":"최고기온",
            "wetherMinTp":"최저기온","WetherAvgHd":"습도","wetherMaxRainqy":"강우량",
            "wetherMaxSolradqy":"일사량","wetherAvgWs":"평균풍속","wetherMaxWs":"최대풍속"
        })
    elif method == "C":
        df = pd.DataFrame(res)[[
            "wetherDt","wetherAvgTp","wetherMaxTp","wetherMinTp",
            "WetherAvgHd","wetherMaxRainqy","wetherMaxSolradqy",
            "wetherAvgWs","wetherMaxWs"
        ]].rename(columns={
            "wetherDt":"순","wetherAvgTp":"평균기온","wetherMaxTp":"최고기온",
            "wetherMinTp":"최저기온","WetherAvgHd":"습도","wetherMaxRainqy":"강우량",
            "wetherMaxSolradqy":"일사량","wetherAvgWs":"평균풍속","wetherMaxWs":"최대풍속"
        })
    else:  # "D" 월별
        df = pd.DataFrame(res)[[
            "wetherDt","wetherAvgTp","wetherMaxTp","wetherMinTp",
            "WetherAvgHd","wetherMaxRainqy","wetherMaxSolradqy",
            "wetherMaxCondenstime","wetherAvgWs","wetherMaxWs"
        ]].rename(columns={
            "wetherDt":"월","wetherAvgTp":"평균기온","wetherMaxTp":"최고기온",
            "wetherMinTp":"최저기온","WetherAvgHd":"습도","wetherMaxRainqy":"강우량",
            "wetherMaxSolradqy":"일사량","wetherMaxCondenstime":"결로시간",
            "wetherAvgWs":"평균풍속","wetherMaxWs":"최대풍속"
        })

    df = _ensure_numeric(df, ["평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"])

    # 날짜/월 파싱
    if "일자" in df.columns:
        df["일자"] = pd.to_datetime(df["일자"], errors="coerce")
        df["연도"] = df["일자"].dt.year
        df["월"] = df["일자"].dt.month
    elif "월" in df.columns:
        # 'YYYY-MM' → 연/월
        df["연도"] = df["월"].astype(str).str[:4].astype(int)
        df["월"] = df["월"].astype(str).str[5:7].astype(int)

    return df

# =========================
# 사이드바 입력
# =========================
with st.sidebar:
    st.header("조회 조건")
    region = st.selectbox("지역 선택", list(AREA_CODE.keys()), index=1)
    stat_label = st.selectbox(
        "통계 구분",
        ["기간(일자범위)", "일별", "순별(상·중·하순)", "월별"], index=3
    )
    # 코드 매핑
    stat_map = {
        "기간(일자범위)": "A",
        "일별": "B",
        "순별(상·중·하순)": "C",
        "월별": "D",
    }
    stat_gb = stat_map[stat_label]

    col1, col2 = st.columns(2)
    with col1:
        s_year = st.number_input("시작 연도", min_value=2010, max_value=2100, value=2024, step=1)
        s_month = st.number_input("시작 월", min_value=1, max_value=12, value=1, step=1)
    with col2:
        e_year = st.number_input("종료 연도", min_value=2010, max_value=2100, value=2025, step=1)
        e_month = st.number_input("종료 월", min_value=1, max_value=12, value=8, step=1)

    s_ym = f"{int(s_year):04d}-{int(s_month):02d}"
    e_ym = f"{int(e_year):04d}-{int(e_month):02d}"

    run = st.button("🔎 조회")

# =========================
# 본문: 조회 실행
# =========================
if run:
    with st.spinner("서버에서 데이터를 가져오는 중..."):
        payload = fetch_aws_stat(region, stat_gb, s_ym, e_ym)

    # 오류 처리
    if not payload or ("result" in payload and len(payload["result"]) == 0):
        st.error("결과가 없습니다. (조건을 바꾸거나 기간을 1년 이내로 설정해 보세요)")
        st.stop()
    if "error" in payload:
        st.error("JSON 파싱 실패 (사이트 응답이 HTML일 수 있음).")
        st.code(payload.get("raw","")[:2000])
        st.stop()

    # JSON → 표
    df = json_to_dataframe(payload)
    if df.empty:
        st.warning("표로 변환할 데이터가 없습니다.")
        st.stop()

    st.subheader("원자료 (서버 응답)")
    st.dataframe(df, use_container_width=True)

    # 월별 집계 (A/B/C 는 월별로 요약해서 쓰는 경우가 많음)
    if stat_gb in ("A","B","C"):
        env_m = _agg_to_monthly(df)
        st.subheader("월별 요약(집계)")
        st.dataframe(env_m, use_container_width=True)

        # 다운로드
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ 원자료 CSV 다운로드", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{region}_{stat_gb}_{s_ym}_{e_ym}_raw.csv",
                               mime="text/csv")
        with c2:
            st.download_button("⬇️ 월별요약 CSV 다운로드", env_m.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{region}_{stat_gb}_{s_ym}_{e_ym}_monthly.csv",
                               mime="text/csv")
    else:
        # 월별(D)은 그대로 사용
        st.info("월별(D) 응답이므로 그대로 사용하면 됩니다.")
        st.download_button("⬇️ 월별 응답 CSV 다운로드", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{region}_{stat_gb}_{s_ym}_{e_ym}_monthly.csv",
                           mime="text/csv")

# 첫 로딩 안내
if not run:
    st.info("좌측 사이드바에서 조건을 선택하고 **🔎 조회**를 눌러 데이터를 가져오세요.")

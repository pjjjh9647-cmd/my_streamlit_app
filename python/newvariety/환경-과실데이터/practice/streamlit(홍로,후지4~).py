# -*- coding: utf-8 -*-
# 파일명: 기상자동불러오기3.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# -------------------------------------------------
# 기본 UI
# -------------------------------------------------
st.set_page_config(page_title="🍎 사과 기상 통계 + 회귀식 예측", layout="wide")
st.title("🍎 사과 기상 통계 수집 + 회귀식 예측기")

# -------------------------------------------------
# 지역 코드 (사이트의 select value와 동일)
# -------------------------------------------------
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

# -------------------------------------------------
# 회귀식 하드코딩: 품종별
#   중간점 '·'은 실행 시 자동으로 '*'로 치환
# -------------------------------------------------
EQUATIONS_BY_CULTIVAR = {
    "홍로": {
        "과중": "과중 = 667.243 -0.597465·일사량_sum_m06 -0.376337·일사량_sum_m04 +0.0756442·일사량_sum_m08 +24.5342·평균풍속_mean_m06 +5.05212·최저기온_mean_m06",
        "종경": "종경 = 124.046 -0.0439843·일사량_sum_m04 -0.0673823·일사량_sum_m06 +0.397906·최저기온_mean_m07 +0.559339·평균풍속_mean_m06 +7.43416·평균풍속_mean_m05 +0.00854761·일사량_sum_m08 -0.15116·최저기온_mean_m06 +1.05474·최저기온_mean_m05 -2.574·최대풍속_mean_m04",
        "횡경": "횡경 = 141.358 -0.0531011·일사량_sum_m06 -0.0459844·일사량_sum_m04 -0.00297393·일사량_sum_m08",
        "L":   "L = -68.3528 +2.1686e-05·강우량_sum_m07 +0.653086·습도_mean_m07 +2.72693·최저기온_mean_m06 +1.14857·평균기온_mean_m08 +0.00918464·강우량_sum_m06 -1.52125·최저기온_mean_m05 -6.00147·평균풍속_mean_m06 +3.14509·최대풍속_mean_m06 +4.16545·평균풍속_mean_m07",
        "a":   "a = 226.087 -1.71711·습도_mean_m07 +0.00830904·강우량_sum_m07 -4.51272·최저기온_mean_m06 +1.51756·최저기온_mean_m05 -1.94971·평균기온_mean_m08 -0.040713·일사량_sum_m07 -0.0185248·강우량_sum_m06 +0.00975096·강우량_sum_m08 +1.93026·최고기온_mean_m07 -0.192988·평균풍속_mean_m06",
        "b":   "b = -23.7933 +0.381237·습도_mean_m07 +0.0052134·강우량_sum_m06 +0.0606139·습도_mean_m05 +1.14817·최저기온_mean_m06 +0.682908·평균풍속_mean_m07 -0.523353·최저기온_mean_m05 -0.350293·최고기온_mean_m05 -0.00264168·강우량_sum_m08 -1.23413·평균풍속_mean_m08 +0.652516·최대풍속_mean_m06",
        "경도": "경도 = 54.6658 +0.2039·습도_mean_m04 +0.0144323·일사량_sum_m08 -0.194462·습도_mean_m08 -0.0140798·일사량_sum_m04 +0.00218425·결로시간_mean_m04 +0.364872·평균기온_mean_m08",
        "당도": "당도 = 1.14467 -0.425354·최대풍속_mean_m06 -1.03279·평균풍속_mean_m07 -0.0754722·평균풍속_mean_m08 +0.781233·평균풍속_mean_m04 -0.0277847·최고기온_mean_m08 -0.0127413·습도_mean_m05 +0.0022906·결로시간_mean_m05 +0.259103·최고기온_mean_m06 +0.0923847·습도_mean_m04 +0.410232·최대풍속_mean_m04 -0.00215038·강우량_sum_m06",
        "산도": "산도 = 0.262689 +0.0555189·최대풍속_mean_m06 +0.0451885·평균풍속_mean_m08 -0.0549304·평균풍속_mean_m04 -0.00534754·최고기온_mean_m06 -0.0236952·평균풍속_mean_m07 -0.00264247·습도_mean_m04 +0.00413186·습도_mean_m08 -0.00334177·평균기온_mean_m07 +0.000124634·강우량_sum_m06 -0.001393·습도_mean_m05",
    },
    "후지": {
        "과중": "과중 = 399.484 -0.229845·일사량_sum_m04 +6.76485·최저기온_mean_m05 -17.8404·최대풍속_mean_m07",
        "종경": "종경 = 183.553 -0.0439139·일사량_sum_m04 -0.626045·최저기온_mean_m04 -0.0145561·일사량_sum_m10 +0.955631·최저기온_mean_m05 -0.0373121·평균기온_mean_m10 -0.656449·습도_mean_m08 -0.0131851·습도_mean_m06 +1.16239·평균기온_mean_m07 -1.16892·최저기온_mean_m09 -0.420997·습도_mean_m09",
        "횡경": "횡경 = 79.6808 -1.82161·최대풍속_mean_m07 +0.796471·평균기온_mean_m05",
        "L":   "L = 75.7441 -0.134414·습도_mean_m08 -0.400198·평균기온_mean_m10 +0.0159958·강우량_sum_m10 +0.0240315·일사량_sum_m08 +0.812706·최저기온_mean_m04 -0.0812023·습도_mean_m04 -0.206954·습도_mean_m06 +0.116267·습도_mean_m10 -0.0069829·일사량_sum_m04 -2.452·평균풍속_mean_m04 -0.0989298·평균기온_mean_m08 -0.384125·평균기온_mean_m07",
        "a":   "a = 4.0922 -0.0203282·강우량_sum_m10 -2.85393·최대풍속_mean_m05 -0.00910066·일사량_sum_m08 +0.0392451·평균기온_mean_m06 +0.307386·습도_mean_m08 +0.000860505·강우량_sum_m08 -0.688696·최대풍속_mean_m08 -0.00144964·일사량_sum_m05 +0.758309·최대풍속_mean_m07",
        "b":   "b = 12.841 -0.0308293·습도_mean_m08 +0.0145751·일사량_sum_m08 -0.00322966·강우량_sum_m08 +0.035395·평균기온_mean_m10 +0.0796701·습도_mean_m10 -0.000543608·일사량_sum_m06 +0.00337387·강우량_sum_m10 -0.00372859·일사량_sum_m04 -0.14243·습도_mean_m06 +0.0641568·평균기온_mean_m08 +0.14721·최저기온_mean_m07 +0.53868·평균풍속_mean_m04",
        "경도": "경도 = 8.39888 +0.0529999·일사량_sum_m09 +6.94666·평균풍속_mean_m07 +3.98929·최대풍속_mean_m08 -0.0451264·일사량_sum_m08 +0.406065·습도_mean_m04 -3.47701·평균풍속_mean_m06 -0.00806023·결로시간_mean_m10 +0.0392251·일사량_sum_m06 +0.00773583·결로시간_mean_m09 -2.0759·최대풍속_mean_m10 +0.000289527·결로시간_mean_m06 -2.8229·최대풍속_mean_m05 +0.000106158·결로시간_mean_m05 +0.270037·최고기온_mean_m09",
        "당도": "당도 = 10.492 +0.00486017·일사량_sum_m06 +0.00146432·결로시간_mean_m10 +0.00262004·결로시간_mean_m07 -0.00156465·결로시간_mean_m09 -1.20735·평균풍속_mean_m10 -0.000317261·결로시간_mean_m05",
        "산도": "산도 = 0.766184 -0.0175941·평균기온_mean_m10 -0.00379855·습도_mean_m04 -0.00807644·최고기온_mean_m04 -4.6679e-05·강우량_sum_m08 +0.00318949·최고기온_mean_m08 -8.77968e-05·강우량_sum_m10 -0.00456198·평균풍속_mean_m04 +0.00411344·최고기온_mean_m07",
    },
}

# -------------------------------------------------
# 유틸
# -------------------------------------------------
def _ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# -------------------------------------------------
# 사이트 AJAX → JSON
# -------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def fetch_aws_stat(region_name: str, stat_gb_code: str, s_ym: str, e_ym: str) -> dict:
    """
    stat_gb_code: "A"(기간), "B"(일별), "C"(순별), "D"(월별)
    s_ym/e_ym: "YYYY-MM"
    """
    session = requests.Session()
    session.get("https://fruit.nihhs.go.kr/apple/aws/awsStat.do", timeout=20)

    form = {
        "pageIndex": "1",
        "searchNum": "0",
        "areaName": region_name,
        "areaCode": AREA_CODE.get(region_name, region_name),
        "statmethod": stat_gb_code,
    }
    if stat_gb_code == "A":
        form["wetherDtBgn"] = f"{s_ym}-01"
        form["wetherDtEnd"] = f"{e_ym}-30"
    elif stat_gb_code in ("B", "C"):
        form["wetherDtM"] = s_ym
    elif stat_gb_code == "D":
        form["wetherDtBgn2"] = s_ym
        form["wetherDtEnd2"] = e_ym
        form["wetherDtBgn"] = s_ym
        form["wetherDtEnd"] = e_ym

    resp = session.post(
        "https://fruit.nihhs.go.kr/apple/aws/awsStatList.do",
        data=form, timeout=30,
        headers={"Referer": "https://fruit.nihhs.go.kr/apple/aws/awsStat.do"}
    )
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {"error": "JSON 파싱 실패", "raw": resp.text[:4000]}

# -------------------------------------------------
# JSON → 표
# -------------------------------------------------
def json_to_dataframe(payload: dict) -> pd.DataFrame:
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
    else:  # "D"
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

    df = _ensure_numeric(
        df, ["평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]
    )

    if "일자" in df.columns:
        df["일자"] = pd.to_datetime(df["일자"], errors="coerce")
        df["연도"] = df["일자"].dt.year
        df["월"] = df["일자"].dt.month
    elif "월" in df.columns:
        df["연도"] = df["월"].astype(str).str[:4].astype(int)
        df["월"] = df["월"].astype(str).str[5:7].astype(int)

    return df

# -------------------------------------------------
# 일/순 → 월별 집계
# -------------------------------------------------
def agg_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if not {"연도","월"}.issubset(df.columns):
        return df.copy()

    agg_map = {
        "평균기온":"mean","최고기온":"mean","최저기온":"mean","습도":"mean",
        "강우량":"sum","일사량":"sum","결로시간":"sum",
        "평균풍속":"mean","최대풍속":"mean"
    }
    use_cols = {k:v for k,v in agg_map.items() if k in df.columns}
    out = df.groupby(["연도","월"], as_index=False).agg(use_cols)
    return out.sort_values(["연도","월"]).reset_index(drop=True)

# -------------------------------------------------
# 월별 가로 확장 피처 (_mean_mMM / _sum_mMM)
# -------------------------------------------------
def build_wide_month_feats(env_m: pd.DataFrame) -> pd.DataFrame:
    if not {"연도","월"}.issubset(env_m.columns):
        raise ValueError("env_m에 '연도','월' 컬럼이 필요합니다.")
    num_cols = [c for c in env_m.columns if c not in ("연도","월") and pd.api.types.is_numeric_dtype(env_m[c])]

    mean_agg = env_m.groupby(["연도","월"], as_index=False)[num_cols].mean()
    sum_agg  = env_m.groupby(["연도","월"], as_index=False)[num_cols].sum()

    wide_mean = None
    for m in range(1, 13):
        sub = mean_agg[mean_agg["월"] == m].drop(columns=["월"]).copy()
        sub = sub.rename(columns={c: f"{c}_mean_m{m:02d}" for c in num_cols})
        wide_mean = sub if wide_mean is None else pd.merge(wide_mean, sub, on="연도", how="outer")

    wide_sum = None
    for m in range(1, 13):
        sub = sum_agg[sum_agg["월"] == m].drop(columns=["월"]).copy()
        sub = sub.rename(columns={c: f"{c}_sum_m{m:02d}" for c in num_cols})
        wide_sum = sub if wide_sum is None else pd.merge(wide_sum, sub, on="연도", how="outer")

    wide = pd.merge(wide_mean, wide_sum, on="연도", how="outer").fillna(0)
    return wide

# -------------------------------------------------
# 회귀식 적용 (한 연도 row에 대해)
# -------------------------------------------------
def apply_equation_row(row: pd.Series, eq_str: str) -> float:
    rhs = eq_str.split("=", 1)[1].strip().replace("·", "*")
    cols = sorted(row.index.tolist(), key=len, reverse=True)
    expr = rhs
    for c in cols:
        expr = expr.replace(c, f"row[{repr(c)}]")
    return float(eval(expr, {"__builtins__": {}}, {"row": row, "np": np}))

# -------------------------------------------------
# 미확보 월 채우기: 최근 3년 평균 or 전체 평균
# -------------------------------------------------
def fill_missing_months_with_climatology(env_m: pd.DataFrame, target_year: int, mode: str = "last3") -> pd.DataFrame:
    req_cols = {"연도", "월"}
    if not req_cols.issubset(env_m.columns):
        raise ValueError("env_m에는 '연도','월' 컬럼이 있어야 합니다.")

    cur = env_m[env_m["연도"] == target_year].copy()
    hist = env_m[env_m["연도"] < target_year].copy()
    if hist.empty:
        return cur

    if mode == "last3":
        last_years = sorted(hist["연도"].unique())[-3:]
        hist = hist[hist["연도"].isin(last_years)]

    num_cols = [c for c in env_m.columns if c not in ("연도","월") and pd.api.types.is_numeric_dtype(env_m[c])]
    climo = hist.groupby("월", as_index=False)[num_cols].mean()

    have_months = set(cur["월"].tolist())
    all_months = set(range(1, 13))
    missing = sorted(list(all_months - have_months))

    if missing:
        fill_rows = climo[climo["월"].isin(missing)].copy()
        fill_rows.insert(0, "연도", target_year)
        cur = pd.concat([cur, fill_rows], ignore_index=True, axis=0)

    cur = cur.sort_values(["연도","월"]).reset_index(drop=True)
    for c in num_cols:
        cur[c] = pd.to_numeric(cur[c], errors="coerce")
    return cur

# -------------------------------------------------
# 사이드바
# -------------------------------------------------
with st.sidebar:
    st.header("조회 조건")
    region = st.selectbox("지역", list(AREA_CODE.keys()), index=1)
    stat_label = st.selectbox("통계 구분", ["기간(일자범위)","일별","순별(상·중·하순)","월별"], index=3)
    stat_map = {"기간(일자범위)":"A","일별":"B","순별(상·중·하순)":"C","월별":"D"}
    stat_gb = stat_map[stat_label]

    col1, col2 = st.columns(2)
    with col1:
        s_year = st.number_input("시작 연도", 2010, 2100, 2024, 1)
        s_month = st.number_input("시작 월", 1, 12, 1, 1)
    with col2:
        e_year = st.number_input("종료 연도", 2010, 2100, 2025, 1)
        e_month = st.number_input("종료 월", 1, 12, 8, 1)

    s_ym = f"{int(s_year):04d}-{int(s_month):02d}"
    e_ym = f"{int(e_year):04d}-{int(e_month):02d}"

    cultivar = st.selectbox("품종", ["홍로", "후지"], index=0)

    fill_strategy = st.selectbox(
        "미확보 월 채움 방법",
        ["채우지 않음(예측 불가)", "최근 3년 월평균으로 채우기", "전체 과거 월평균으로 채우기"],
        index=1
    )

    run = st.button("🔎 조회 & 예측")

# -------------------------------------------------
# 실행
# -------------------------------------------------
if run:
    with st.spinner("서버에서 기상 데이터를 가져오는 중..."):
        payload = fetch_aws_stat(region, stat_gb, s_ym, e_ym)

    if "error" in payload:
        st.error("응답이 JSON 형식이 아닙니다.")
        st.code(payload.get("raw","")[:1000])
        st.stop()

    df = json_to_dataframe(payload)
    if df.empty:
        st.error("결과가 없습니다. 기간(특히 '기간'은 1년 이내)과 지역을 조정하세요.")
        st.stop()

    st.subheader("원자료")
    st.dataframe(df, use_container_width=True)

    # 월별 집계
    if stat_gb in ("A","B","C"):
        env_m = agg_to_monthly(df)
        st.subheader("월별 요약(집계)")
    else:
        env_m = df.copy()
        if "연도" not in env_m.columns and "월" in env_m.columns:
            env_m["연도"] = env_m["월"].astype(str).str[:4].astype(int)
            env_m["월"] = env_m["월"].astype(str).str[5:7].astype(int)
        st.subheader("월별 응답")

    st.dataframe(env_m, use_container_width=True)

    # 다운로드
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ 원자료 CSV",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{region}_{stat_gb}_{s_ym}_{e_ym}_raw.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "⬇️ 월별요약 CSV",
            env_m.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{region}_{stat_gb}_{s_ym}_{e_ym}_monthly.csv",
            mime="text/csv",
        )

    # 예측 연도 선택
    years = sorted(env_m["연도"].dropna().astype(int).unique())
    if not years:
        st.warning("예측 가능한 연도가 없습니다.")
        st.stop()

    sel_year = st.selectbox("예측 연도 선택", years, index=len(years)-1)

    # 선택 연도의 미확보 월 채우기
    env_m_filled = env_m.copy()
    if fill_strategy != "채우지 않음(예측 불가)":
        mode = "last3" if "최근 3년" in fill_strategy else "all"
        filled_rowset = fill_missing_months_with_climatology(env_m, sel_year, mode=mode)
        env_m_filled = pd.concat(
            [env_m_filled[env_m_filled["연도"] != sel_year], filled_rowset],
            ignore_index=True
        ).sort_values(["연도","월"])

        st.info(f"{sel_year}년의 비어 있는 월을 '{fill_strategy}'로 채웠습니다.")
        st.dataframe(env_m_filled[env_m_filled["연도"] == sel_year], use_container_width=True)

    # 가로 확장 피처
    try:
        wide = build_wide_month_feats(env_m_filled)
    except Exception as e:
        st.error(f"월별 피처 생성 실패: {e}")
        st.stop()

    st.subheader("월별 가로 확장 피처 (선택 연도)")
    st.dataframe(wide[wide["연도"] == sel_year], use_container_width=True)

    # 선택 연도 행
    row = wide[wide["연도"] == sel_year].iloc[0]

    # 품종별 회귀식 적용
    EQUATIONS = EQUATIONS_BY_CULTIVAR.get(cultivar, {})
    preds = {}
    for tgt, formula in EQUATIONS.items():
        try:
            preds[tgt] = apply_equation_row(row, formula)
        except Exception as e:
            preds[tgt] = f"에러: {e}"

    st.subheader(f"회귀식 예측 결과  품종: {cultivar}  연도: {sel_year}")
    st.write(preds)

else:
    st.info("좌측에서 조건을 고르고 품종을 선택한 뒤 조회 & 예측 버튼을 눌러주세요.")

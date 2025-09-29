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
# 회귀식 하드코딩: 품종별 [RF Surrogate (Top3)]
#   주의: 변수명은 아래 compute_surrogate_features가 생성하는 이름과 일치해야 합니다.
# -------------------------------------------------
EQUATIONS_BY_CULTIVAR = {
    "홍로": {
        "과중": "과중 = 9.464941*tmin_min + -7.956509*rad_mean + -5.788883*BIO3 + 499.639944",
        "종경": "종경 = -1.830825*BIO3 + 0.100745*BIO6 + -0.021367*BIO19 + 140.473176",
        "횡경": "횡경 = -1.413365*rad_mean + 0.877456*tmin_min + 0.029047*BIO4 + 75.328444",
        "L":   "L = 1.56844*rad_mean + 0.136542*BIO15 + 0.011198*BIO13 + 9.639695",
        "a":   "a = -0.812848*rad_mean + -0.214583*BIO15 + -0.012265*BIO13 + 53.620448",
        "b":   "b = 0.49007*rad_mean + 0.384621*tmin_min + 0.002595*prcp_sum + 4.937532",
        "경도": "경도 = 0.920394*BIO7 + 0.037746*BIO19 + 0.010592*BIO17 + 20.121689",
        "당도": "당도 = 0.371647*BIO7 + -0.164394*rad_mean + 0.002821*BIO3 + 1.434033",
        "산도": "산도 = -0.009582*BIO7 + -0.007435*BIO5 + -0.007435*tmax_max + 1.091684",
    },
    "후지": {
        "과중": "과중 = 4.182353*tmin_min + 1.990877*BIO1 + -1.53413*BIO3 + 281.827865",
        "종경": "종경 = 1.012748*tmean_mean + 0.555473*BIO8 + -0.460091*BIO3 + 58.496371",
        "횡경": "횡경 = -1.067049*rad_mean + 0.19149*BIO3 + 0.010416*BIO4 + 88.350053",
        "L":   "L = 0.761287*tmax_max + 0.761287*BIO5 + -0.084436*BIO15 + 6.401431",
        "a":   "a = -0.987092*BIO6 + 0.390801*BIO8 + 0.109661*BIO15 + -12.053255",
        "b":   "b = 0.563414*rad_mean + 0.386501*tmax_max + 0.386501*BIO5 + -17.13521",
        "경도": "경도 = 1.427411*rad_mean + 0.046635*BIO17 + 0.010503*BIO19 + 34.0088",
        "당도": "당도 = 0.138938*BIO7 + 0.059249*BIO3 + 0.000237*BIO4 + 5.799479",
        "산도": "산도 = -0.010091*BIO5 + -0.00721*BIO1 + -0.001546*rad_mean + 0.769385",
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
# 연간 파생지표 + BIO 지표 계산
#   입력: 특정 연도의 월별 데이터(12행 권장, 열: 월/평균기온/최고기온/최저기온/강우량/일사량/...)
#   출력: dict (수식에 들어갈 피처들)
# -------------------------------------------------
def _rolling3_with_wrap(arr):
    # 길이 12 가정, 3개월 이동합/평균을 월 경계 넘어 순환 계산
    n = len(arr)
    ext = np.concatenate([arr, arr[:2]])
    sums = np.array([ext[i:i+3].sum() for i in range(n)])
    means = sums / 3.0
    return sums, means

def compute_surrogate_features(env_m_year: pd.DataFrame) -> dict:
    # 필요한 컬럼 확인
    needed = ["월","평균기온","최고기온","최저기온","강우량","일사량"]
    for c in needed:
        if c not in env_m_year.columns:
            raise ValueError(f"선택 연도 데이터에 '{c}' 컬럼이 필요합니다.")

    df = env_m_year.sort_values("월").reset_index(drop=True).copy()

    # 배열 준비
    tmean = df["평균기온"].to_numpy(float)
    tmax  = df["최고기온"].to_numpy(float)
    tmin  = df["최저기온"].to_numpy(float)
    prcp  = df["강우량"].to_numpy(float)
    rad   = df["일사량"].to_numpy(float)

    # 기본 파생
    tmin_min   = np.nanmin(tmin)
    tmax_max   = np.nanmax(tmax)
    tmean_mean = np.nanmean(tmean)
    # rad_mean: 월 일사량(rad)은 월합으로 집계되어 있음 → 월합들의 평균(=연평균 월 일사량)
    rad_mean   = np.nanmean(rad)
    prcp_sum   = np.nansum(prcp)

    # BIO 지표
    # BIO1: 연평균기온
    BIO1 = float(tmean_mean)

    # BIO5/6/7
    BIO5 = float(np.nanmax(tmax))   # 가장 따뜻한 달의 최고기온(월 평균 Tmax 최대)
    BIO6 = float(np.nanmin(tmin))   # 가장 추운 달의 최저기온(월 평균 Tmin 최소)
    BIO7 = float(BIO5 - BIO6)       # 연평 온도 범위

    # BIO2(평균 일교차) 근사: 각 월(Tmax - Tmin)의 평균
    diurnal = tmax - tmin
    BIO2 = float(np.nanmean(diurnal))

    # BIO3: 등온성 = (BIO2 / BIO7) * 100
    BIO3 = float((BIO2 / BIO7) * 100) if BIO7 not in (0, np.nan) else np.nan

    # BIO4: 온도 계절성 = 월평균기온 표준편차 * 100
    BIO4 = float(np.nanstd(tmean, ddof=0) * 100.0)

    # 분기 관련(3개월 창, 순환)
    prcp_3sum, tmean_3mean = _rolling3_with_wrap(prcp), _rolling3_with_wrap(tmean)
    prcp_3sum_vals = prcp_3sum[0]
    tmean_3mean_vals = tmean_3mean[1]

    # BIO8: 가장 습한 분기의 평균기온
    wettest_q_idx = int(np.nanargmax(prcp_3sum_vals))
    BIO8 = float(tmean_3mean_vals[wettest_q_idx])

    # BIO13: 가장 습한 달의 강수량
    BIO13 = float(np.nanmax(prcp))

    # BIO15: 강수 계절성 = 월강수량 변동계수(CV, %) = 표준편차/평균 *100
    prcp_mean = float(np.nanmean(prcp))
    prcp_std  = float(np.nanstd(prcp, ddof=0))
    BIO15 = float((prcp_std / prcp_mean) * 100.0) if prcp_mean not in (0, np.nan) else np.nan

    # BIO17: 가장 건조한 분기의 강수량 합
    driest_q_idx = int(np.nanargmin(prcp_3sum_vals))
    BIO17 = float(prcp_3sum_vals[driest_q_idx])

    # BIO19: 가장 추운 분기의 강수량 합
    coldest_q_idx = int(np.nanargmin(tmean_3mean_vals))
    BIO19 = float(prcp_3sum_vals[coldest_q_idx])

    feats = {
        "tmin_min": float(tmin_min),
        "tmax_max": float(tmax_max),
        "tmean_mean": float(tmean_mean),
        "rad_mean": float(rad_mean),
        "prcp_sum": float(prcp_sum),
        "BIO1": BIO1, "BIO3": BIO3, "BIO4": BIO4, "BIO5": BIO5, "BIO6": BIO6, "BIO7": BIO7,
        "BIO8": BIO8, "BIO13": BIO13, "BIO15": BIO15, "BIO17": BIO17, "BIO19": BIO19,
    }
    return feats

# -------------------------------------------------
# 수식 적용
# -------------------------------------------------
def apply_equation_series(series: pd.Series, eq_str: str) -> float:
    """
    수식을 그대로 평가하되, 변수는 locals로 전달한다.
    예: "과중 = 1.2*tmin_min + 0.3*BIO3 + 10" 에서
    tmin_min, BIO3 등은 series에 들어 있는 값을 locals로 공급.
    """
    rhs = eq_str.split("=", 1)[1].strip().replace("·", "*")

    # locals로 쓸 네임스페이스 구성 (float 변환, NaN은 그대로 유지)
    ns = {k: (float(v) if pd.notna(v) else float("nan")) for k, v in series.items()}

    try:
        val = eval(rhs, {"__builtins__": {}}, ns)  # 안전 모드
        return float(val)
    except Exception as e:
        raise RuntimeError(f"식 평가 실패: {e}\n식: {rhs}\n변수: {list(ns.keys())}")


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

    # 선택 연도 데이터만 추출
    env_year = env_m_filled[env_m_filled["연도"] == sel_year].copy()

    # 파생지표 계산
    try:
        feats_dict = compute_surrogate_features(env_year)
    except Exception as e:
        st.error(f"파생지표 계산 실패: {e}")
        st.stop()

    feats_series = pd.Series(feats_dict)
    st.subheader("연간 파생지표 / BIO 지표")
    st.dataframe(pd.DataFrame([feats_dict]), use_container_width=True)

    # 품종별 회귀식 적용
EQUATIONS = EQUATIONS_BY_CULTIVAR.get(cultivar, {})
preds = {}
for tgt, formula in EQUATIONS.items():
    try:
        v = apply_equation_series(feats_series, formula)
        preds[tgt] = None if (pd.isna(v) or np.isinf(v)) else float(v)
    except Exception as e:
        preds[tgt] = f"에러: {e}"


    st.subheader(f"회귀식 예측 결과  품종: {cultivar}  연도: {sel_year}")
    st.write(preds)

else:
    st.info("좌측에서 조건을 고르고 품종을 선택한 뒤 조회 & 예측 버튼을 눌러주세요.")

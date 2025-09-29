# -*- coding: utf-8 -*-
# 파일명: 기상자동불러오기3.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import io
import matplotlib.pyplot as plt
from typing import Optional
import re

import matplotlib
import matplotlib.font_manager as fm
import os

def _set_korean_font():
    bundled_candidates = [
        "fonts/NanumGothic.ttf",
        "fonts/NotoSansKR-Regular.otf",
        "NanumGothic.ttf",
        "NotoSansKR-Regular.otf",
    ]
    for p in bundled_candidates:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                family = fm.FontProperties(fname=p).get_name()
                matplotlib.rcParams["font.family"] = family
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                pass
    preferred = ["Malgun Gothic", "AppleGothic", "NanumGothic",
                 "Noto Sans CJK KR", "Noto Sans KR", "NanumBarunGothic"]
    sys_fonts = set(f.name for f in fm.fontManager.ttflist)
    for name in preferred:
        if name in sys_fonts:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
    matplotlib.rcParams["axes.unicode_minus"] = False
    st.warning("한글 폰트를 찾지 못했습니다. fonts/ 폴더에 나눔고딕(NanumGothic.ttf) 등을 넣어주세요.")

_set_korean_font()

# ---------------------------
# 공통 유틸: 품질표 컬럼 정규화/패턴매칭
# ---------------------------
def normalize_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    """NBSP/줄바꿈/괄호 공백/멀티헤더 등을 통일"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip()
                      for tup in df.columns.values]
    def _clean(s: str) -> str:
        s = str(s)
        s = s.replace("\xa0", " ")              # NBSP -> space
        s = s.replace("\n", " ").replace("\r", " ")
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"\s*\(\s*", " (", s)        # "( " -> " ("
        s = re.sub(r"\s*\)\s*", ")", s)         # " )" -> ")"
        return s
    out = df.copy()
    out.columns = [_clean(c) for c in out.columns]
    # 자주 보이는 변형 별칭도 한 번 더 정돈(선택)
    alias = {
        "경도 평균(N/ø11mm)": "경도평균(N/ø11mm)",
        "경도 평균 (N/ø11mm)": "경도평균(N/ø11mm)",
        "착색 (Hunter L)": "착색(Hunter L)",
        "착색 (Hunter a)": "착색(Hunter a)",
        "착색 (Hunter b)": "착색(Hunter b)",
    }
    out = out.rename(columns={k: v for k, v in alias.items() if k in out.columns})
    return out

def get_first_col_by_pattern(df: pd.DataFrame, pattern: str) -> Optional[str]:
    """정규식 패턴으로 컬럼명 1개 찾기(대소문자 무시)"""
    pat = re.compile(pattern, flags=re.IGNORECASE)
    for c in df.columns:
        if pat.search(str(c)):
            return c
    return None

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

# 과실품질 페이지의 지역 표기와 매핑
REGION_NAME_MAP = {
    "경기화성": "화성",
    "경북영주": "영주",
    "경북청송": "청송",
    "대구군위": "군위",
    "경남거창": "거창",
    "전북장수": "장수",
    "경기포천": "포천",
    "충북충주": "충주",
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

def cultivar_window(cultivar: str):
    if cultivar == "홍로":
        return 4, 8
    return 4, 10  # 후지 기본

def get_today_ym():
    now = datetime.now()
    return now.year, now.month

# -------------------------------------------------
# 사이트 AJAX → JSON (AWS 통계)
# -------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def fetch_aws_stat(region_name: str, stat_gb_code: str, s_ym: str, e_ym: str) -> dict:
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
    raw = pd.DataFrame(res)

    with st.expander("🔧 API 원시 컬럼 보기", expanded=False):
        st.write(sorted(list(raw.columns)))

    rename_map = {
        "statsDt": "일자", "wetherDt": "일자",
        "dalyWetherAvrgTp": "평균기온", "wetherAvgTp": "평균기온",
        "dalyWetherMxmmTp": "최고기온", "wetherMaxTp": "최고기온",
        "dalyWetherMummTp": "최저기온", "wetherMinTp": "최저기온",
        "dalyWetherAvrgHd": "습도",     "WetherAvgHd": "습도",
        "dalyWetherTtalRainqy": "강우량", "wetherMaxRainqy": "강우량",
        "dalyWetherMxmmSolradqy": "일사량",
        "wetherMaxSolradqy": "일사량",
        "wetherSumSolradqy": "일사량",
        "dalyWetherMxmmCondenstime": "결로시간",
        "wetherMaxCondenstime": "결로시간",
        "wetherSumCondenstime": "결로시간",
        "dalyWetherAvrgWs": "평균풍속", "wetherAvgWs": "평균풍속",
        "dalyWetherMxmmWs": "최대풍속", "wetherMaxWs": "최대풍속",
        "wetherDtMonth": "월", "wetherDt": "월",
    }
    raw = raw.rename(columns=rename_map)

    if method == "A":
        want = ["일자","평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]
    elif method == "B":
        want = ["일자","평균기온","최고기온","최저기온","습도","강우량","일사량","평균풍속","최대풍속"]
    elif method == "C":
        want = ["순","평균기온","최고기온","최저기온","습도","강우량","일사량","평균풍속","최대풍속"] if "순" in raw.columns \
               else ["일자","평균기온","최고기온","최저기온","습도","강우량","일사량","평균풍속","최대풍속"]
    else:  # "D"
        want = ["월","평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"] \
               if "월" in raw.columns else \
               ["일자","평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]

    use_cols = [c for c in want if c in raw.columns]
    if not use_cols:
        return pd.DataFrame()
    df = raw[use_cols].copy()

    numcands = ["평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]
    df = _ensure_numeric(df, [c for c in numcands if c in df.columns])

    if "일자" in df.columns and df["일자"].dtype != "int64":
        df["일자"] = pd.to_datetime(df["일자"], errors="coerce")
        if "연도" not in df.columns:
            df["연도"] = df["일자"].dt.year
        if "월" not in df.columns:
            df["월"] = df["일자"].dt.month
    elif "월" in df.columns:
        df["연도"] = pd.to_numeric(df["월"].astype(str).str[:4], errors="coerce")
        df["월"] = pd.to_numeric(df["월"].astype(str).str[-2:], errors="coerce")

    if {"연도","월"}.issubset(df.columns):
        df = df.sort_values(["연도","월"]).reset_index(drop=True)
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
# 미확보 월 채우기
# -------------------------------------------------
def fill_missing_or_future_with_climatology(env_m: pd.DataFrame, target_year: int, cultivar: str, mode: str = "last3") -> pd.DataFrame:
    need_cols = {"연도","월"}
    if not need_cols.issubset(env_m.columns):
        raise ValueError("env_m에는 '연도','월' 컬럼이 있어야 합니다.")
    s_mon, e_mon = cultivar_window(cultivar)
    cur_year, cur_mon = get_today_ym()

    cur = env_m[env_m["연도"] == target_year].copy()
    hist = env_m[env_m["연도"] < target_year].copy()
    if hist.empty:
        return cur
    if mode == "last3":
        last_years = sorted(hist["연도"].unique())[-3:]
        hist = hist[hist["연도"].isin(last_years)]

    num_cols = [c for c in env_m.columns if c not in ("연도","월") and pd.api.types.is_numeric_dtype(env_m[c])]
    climo = hist.groupby("월", as_index=False)[num_cols].mean()

    months_window = list(range(s_mon, e_mon+1))
    have = set(cur["월"].tolist())
    future_cut = cur_mon if target_year == cur_year else 12
    future_months = [m for m in months_window if (target_year == cur_year and m > future_cut) or (target_year > cur_year)]
    missing_months = [m for m in months_window if m not in have]

    to_fill = sorted(set(future_months) | set(missing_months))
    if to_fill:
        fill_rows = climo[climo["월"].isin(to_fill)].copy()
        if fill_rows.empty:
            climo_all = env_m[env_m["연도"] < target_year].groupby("월", as_index=False)[num_cols].mean()
            fill_rows = climo_all[climo_all["월"].isin(to_fill)].copy()
        fill_rows.insert(0, "연도", target_year)
        cur = pd.concat([cur, fill_rows], ignore_index=True, axis=0)

    cur = cur[(cur["월"] >= s_mon) & (cur["월"] <= e_mon)]
    cur = cur.sort_values(["연도","월"]).reset_index(drop=True)
    for c in num_cols:
        cur[c] = pd.to_numeric(cur[c], errors="coerce")
    return cur

# -------------------------------------------------
# 전년도 과실품질 테이블
# -------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_quality_tables(selyear: int, lastyear: int, cultivar: str) -> dict:
    code = "apple01" if cultivar == "후지" else "apple02"
    url = "https://fruit.nihhs.go.kr/apple/qlityInfo_frutQlity.do"
    params = {
        "frtgrdCode": "apple",
        "selyear": str(selyear),
        "lastYear": str(lastyear),
        "searchGubun": code,
        "pageIndex": "1",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    try:
        import io as _io
        tables = pd.read_html(_io.StringIO(r.text))
    except Exception as e:
        st.warning(f"전년도 과실품질 표 파싱에 실패했습니다: {e}\n"
                   "lxml/html5lib 설치 여부를 확인해 주세요. (pip install lxml html5lib)")
        return {}

    cleaned = []
    for t in tables:
        t2 = normalize_quality_columns(t)
        if set(["지역","수확일자"]).issubset(set(t2.columns)):
            cleaned.append(t2)

    result = {}
    if len(cleaned) >= 1:
        result["this"] = cleaned[0]
    if len(cleaned) >= 2:
        result["last"] = cleaned[1]
    return result

def pick_region_row(qdf: pd.DataFrame, region_disp_name: str) -> Optional[pd.Series]:
    if qdf is None or qdf.empty:
        return None
    sub = qdf[qdf["지역"].astype(str).str.strip() == region_disp_name]
    if sub.empty:
        return None
    sub = sub.copy()
    sub["수확일자"] = pd.to_datetime(sub["수확일자"], errors="coerce")
    sub = sub.sort_values("수확일자", ascending=False)
    return sub.iloc[0]  # ✅ 여기서 끝. (죽은 코드 제거)

# -------------------------------------------------
# 사이드바
# -------------------------------------------------
with st.sidebar:
    st.header("조회 조건")
    region = st.selectbox("지역", list(AREA_CODE.keys()), index=1)
    cultivar = st.selectbox("품종", ["홍로", "후지"], index=0)
    fill_strategy = st.selectbox("예상 날씨 방법", ["최근 3년 월평균", "전체 과거 월평균"], index=0)
    run = st.button("🔎 자동조회 & 예측")

# -------------------------------------------------
# 실행
# -------------------------------------------------
if run:
    try:
        cur_year, cur_month = get_today_ym()
        s_mon, e_mon = cultivar_window(cultivar)
        s_ym = f"{cur_year:04d}-{s_mon:02d}"
        e_ym_real = f"{cur_year:04d}-{min(e_mon, cur_month):02d}"

        with st.spinner("기상 데이터를 불러오고 있습니다..."):
            payload = fetch_aws_stat(region, "D", s_ym, e_ym_real)

        if "error" in payload:
            st.error("응답이 JSON 형식이 아닙니다.")
            st.code(payload.get("raw","")[:1000])
            st.stop()

        df = json_to_dataframe(payload)
        st.subheader("{region_disp} 기상 데이터")
        if df.empty:
            st.warning("올해 기간 내 실측 데이터가 없습니다. 과거 기후평년만으로 채워 예측합니다.")
        else:
            st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error("예상치 못한 오류가 발생했습니다. 아래 예외를 확인해 주세요.")
        st.exception(e)

    env_m = df.copy()
    if not env_m.empty and "연도" not in env_m.columns and "월" in env_m.columns:
        env_m["연도"] = env_m["월"].astype(str).str[:4].astype(int)
        env_m["월"] = env_m["월"].astype(str).str[5:7].astype(int)

    past_payload = fetch_aws_stat(region, "D", f"{max(cur_year-15,2010):04d}-01", f"{cur_year-1:04d}-12")
    past_df = json_to_dataframe(past_payload)
    env_all = pd.concat([env_m, past_df], ignore_index=True) if not past_df.empty else env_m

    mode = "last3" if "최근 3년" in fill_strategy else "all"
    filled_this_year = fill_missing_or_future_with_climatology(env_all, cur_year, cultivar, mode=mode)

    st.subheader("{region_disp} 예상 기상데이터")
    st.dataframe(filled_this_year, use_container_width=True)

    try:
        env_for_wide = pd.concat([env_all[env_all["연도"] != cur_year], filled_this_year], ignore_index=True)
        wide = build_wide_month_feats(env_for_wide)
    except Exception as e:
        st.error(f"월별 피처 생성 실패: {e}")
        st.stop()

    if cur_year not in set(wide["연도"].astype(int).tolist()):
        st.error("예측 연도 행을 구성하지 못했습니다.")
        st.stop()

    row = wide[wide["연도"] == cur_year].iloc[0]

    EQUATIONS = EQUATIONS_BY_CULTIVAR.get(cultivar, {})
    preds = {}
    for tgt, formula in EQUATIONS.items():
        try:
            preds[tgt] = apply_equation_row(row, formula)
        except Exception as e:
            preds[tgt] = f"에러: {e}"

    st.subheader(f"{cur_year} {cultivar} 과실품질예측")
    pred_df = pd.DataFrame([preds]).T.reset_index()
    pred_df.columns = ["항목", "예측값(올해)"]
    st.dataframe(pred_df, use_container_width=True)

    # -----------------------------
    # 전년도 과실품질 표 불러와서 지역 행 추출
    # -----------------------------
    with st.spinner("전년도 과실품질 데이터를 불러오는 중..."):
        qdict = fetch_quality_tables(cur_year, cur_year-1, cultivar)

    region_disp = REGION_NAME_MAP.get(region, region)
    last_row = None
    if qdict and "last" in qdict and qdict["last"] is not None and not qdict["last"].empty:
        q_last = normalize_quality_columns(qdict["last"])
        with st.expander("전년도 테이블 실제 컬럼명(정규화 후)"):
            st.write(list(q_last.columns))
        last_row = pick_region_row(q_last, region_disp)

    if last_row is not None:
        # 패턴 기반 매칭으로 컬럼 찾기
        patterns = {
            "과중": r"^과중",
            "종경": r"^종경",
            "횡경": r"^횡경",
            "경도": r"경도\s*평균|경도평균",       # 경도평균(N/ø11mm) 다양한 표기
            "당도": r"^당도",
            "산도": r"^산도",
            "L":   r"Hunter\s*L\b",
            "a":   r"Hunter\s*a\b",
            "b":   r"Hunter\s*b\b",
        }

        rows = []
        for k, pat in patterns.items():
            col = get_first_col_by_pattern(last_row.to_frame().T, pat)
            last_val = None
            if col is not None:
                try:
                    last_val = float(str(last_row[col]).replace(",", "").strip())
                except Exception:
                    last_val = None
            pred_val = preds.get(k, None)
            rows.append([k, pred_val, last_val])

        compare_df = pd.DataFrame(rows, columns=["항목","예측값(올해)","전년도 실제값"])
        st.subheader(f"{cur_year} 예측 vs 전년도 과실품질")
        st.dataframe(compare_df, use_container_width=True)

        plot_df = compare_df.dropna(subset=["예측값(올해)","전년도 실제값"]).copy()
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(plot_df))
            w = 0.35
            ax.bar(x - w/2, plot_df["예측값(올해)"].values, width=w, label="예측(올해)")
            ax.bar(x + w/2, plot_df["전년도 실제값"].values, width=w, label="전년도")
            ax.set_xticks(x)
            ax.set_xticklabels(plot_df["항목"].tolist(), rotation=0)
            ax.set_title(f"{region_disp} · {cultivar}  올해 예측 vs 전년도")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("그래프로 비교할 수 있는 공통 항목이 없습니다.")
    else:
        st.warning("전년도 과실품질에서 해당 지역 행을 찾지 못했습니다. 품종/지역 조합을 바꿔보세요.")

    # 원자료 다운로드
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "⬇️ 올해 실측 월별 CSV",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{region}_{cultivar}_{cur_year}_monthly_measured.csv",
            mime="text/csv",
            disabled=df.empty
        )
    with c2:
        st.download_button(
            "⬇️ 올해 예측에 사용한 월별 CSV",
            filled_this_year.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{region}_{cultivar}_{cur_year}_monthly_used.csv",
            mime="text/csv"
        )
    with c3:
        st.download_button(
            "⬇️ 회귀식 예측 결과 CSV",
            pred_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{region}_{cultivar}_{cur_year}_predictions.csv",
            mime="text/csv"
        )
else:
    st.info("좌측에서 지역과 품종을 고른 뒤 자동조회 & 예측 버튼을 눌러주세요. 올해 남은 월은 ‘예상 날씨(최근 3년 또는 전체 과거 평균)’로 채워 예측합니다.")

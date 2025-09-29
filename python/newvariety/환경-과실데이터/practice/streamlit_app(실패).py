# -*- coding: utf-8 -*-
# 파일: streamlit_app.py
import os, re, json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm

# =========================
# 기본 세팅
# =========================
st.set_page_config(page_title="🍎 사과 과실 품질 예측 (API 자동 재학습)", layout="wide")

EQUATIONS_PATH = Path("equations.json")  # 재학습한 회귀식 영구 저장 위치

AREA_CODE = {
    "경기화성": "324", "경북영주": "331", "경북청송": "332", "대구군위": "333",
    "경남거창": "334", "전북장수": "335", "경기포천": "337", "충북충주": "338",
}
REGION_NAME_MAP = {
    "경기화성": "화성", "경북영주": "영주", "경북청송": "청송", "대구군위": "군위",
    "경남거창": "거창", "전북장수": "장수", "경기포천": "포천", "충북충주": "충주",
}

# 최초 기동시 기본식(없으면 예측은 재학습 후 가능)
DEFAULT_EQUATIONS = {
    "홍로": {},  # 초기에는 비워두고 버튼으로 학습 권장(원하시면 여기 기본식 넣으셔도 됩니다)
    "후지": {},
}

# =========================
# 한글 폰트(가능할 때만)
# =========================
import matplotlib
import matplotlib.font_manager as fm

def _set_korean_font():
    preferred = ["Malgun Gothic", "AppleGothic", "NanumGothic",
                 "Noto Sans CJK KR", "Noto Sans KR", "NanumBarunGothic"]
    sys_fonts = set(f.name for f in fm.fontManager.ttflist)
    for name in preferred:
        if name in sys_fonts:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
    matplotlib.rcParams["axes.unicode_minus"] = False

_set_korean_font()
plt.rcParams["font.size"] = 6

def fetch_aws_stat_fallback_html(region_name: str, s_ym: str, e_ym: str) -> pd.DataFrame:
    """
    JSON 응답이 실패할 때: 월별 통계를 HTML에서 직접 파싱하여 DataFrame으로 반환.
    - statmethod=D(월별)
    - 필요한 컬럼: 월, 평균기온/최고기온/최저기온/습도/강우량/일사량/결로시간/평균풍속/최대풍속
    """
    import io as _io
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://fruit.nihhs.go.kr/apple/aws/awsStat.do",
    }
    sess = requests.Session()
    # 첫 페이지로 쿠키/세션 취득
    sess.get("https://fruit.nihhs.go.kr/apple/aws/awsStat.do", headers=headers, timeout=20)

    params = {
        "pageIndex": "1",
        "searchNum": "0",
        "areaName": region_name,
        "areaCode": AREA_CODE.get(region_name, region_name),
        "statmethod": "D",
        "wetherDtBgn2": s_ym,
        "wetherDtEnd2": e_ym,
        "wetherDtBgn": s_ym,
        "wetherDtEnd": e_ym,
    }
    # 리스트 페이지(HTML) 요청
    r = sess.post("https://fruit.nihhs.go.kr/apple/aws/awsStatList.do",
                  data=params, headers=headers, timeout=30)
    r.raise_for_status()

    # 페이지 내 테이블을 전부 읽고, 월/지표 컬럼을 포함한 표를 선택
    tables = pd.read_html(_io.StringIO(r.text))
    wanted_cols = {"월","평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"}
    cand = None
    for t in tables:
        cols = set(map(str, t.columns))
        if "월" in cols and len(wanted_cols & cols) >= 5:
            cand = t.copy()
            break
    if cand is None:
        # 디버그: 일부 페이지는 리스트가 아니라 상세 그리드로 반환될 수 있어 원문 일부 출력
        st.error("[환경/HTML] 월별 표를 찾지 못했습니다. 응답 일부를 표시합니다.")
        st.code(r.text[:800])
        return pd.DataFrame()

    # 컬럼 정리 & 숫자화
    cand = cand.rename(columns=lambda x: str(x).strip())
    cand["연도"] = pd.to_numeric(cand["월"].astype(str).str[:4], errors="coerce")
    cand["월"]   = pd.to_numeric(cand["월"].astype(str).str[-2:], errors="coerce")
    numcands = ["평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]
    for c in numcands:
        if c in cand.columns:
            cand[c] = pd.to_numeric(cand[c], errors="coerce")
    cand = cand.dropna(subset=["연도","월"])
    cand = cand.sort_values(["연도","월"]).reset_index(drop=True)
    return cand


# =========================
# 공통 유틸
# =========================
def _clean_str(s: str) -> str:
    s = str(s)
    s = s.replace("\xa0", " ").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*\(\s*", " (", s)
    s = re.sub(r"\s*\)\s*", ")", s)
    return s

def normalize_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip()
                       for tup in out.columns.values]
    out.columns = [_clean_str(c) for c in out.columns]
    alias = {
        "경도 평균(N/ø11mm)": "경도평균(N/ø11mm)",
        "경도 평균 (N/ø11mm)": "경도평균(N/ø11mm)",
        "착색 (Hunter L)": "착색(Hunter L)",
        "착색 (Hunter a)": "착색(Hunter a)",
        "착색 (Hunter b)": "착색(Hunter b)",
    }
    out = out.rename(columns={k: v for k, v in alias.items() if k in out.columns})
    if "지역" in out.columns:
        out["지역"] = out["지역"].map(_clean_str)
    if "수확일자" in out.columns:
        out["수확일자"] = pd.to_datetime(out["수확일자"], errors="coerce")
    return out

def get_first_col_by_pattern(df: pd.DataFrame, pattern: str) -> Optional[str]:
    pat = re.compile(pattern, flags=re.IGNORECASE)
    for c in df.columns:
        if pat.search(str(c)):
            return c
    return None

def cultivar_window(cultivar: str):
    return (4, 8) if cultivar == "홍로" else (4, 10)

def get_today_ym():
    now = datetime.now()
    return now.year, now.month

def _ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# =========================
# API: 환경데이터(AWS)
# =========================
@st.cache_data(show_spinner=False, ttl=300)
@st.cache_data(show_spinner=False, ttl=300)
def fetch_aws_stat(region_name: str, stat_gb_code: str, s_ym: str, e_ym: str) -> dict:
    """
    과수생육품질관리시스템 AWS 월별 통계(JSON/AJAX) 호출.
    - 헤더 강화(ajax/ua/referer), 세션 쿠키 선취득
    - JSON/JSONP/HTML 응답 모두 대비
    - 실패 시 진단 로그 반환
    """
    import time, json, re
    BASE = "https://fruit.nihhs.go.kr/apple/aws"
    list_url = f"{BASE}/awsStatList.do"
    referer  = f"{BASE}/awsStat.do"

    # 0) 파라미터 sanity
    s_ym = str(s_ym)
    e_ym = str(e_ym)
    if not re.match(r"^\d{4}-\d{2}$", s_ym) or not re.match(r"^\d{4}-\d{2}$", e_ym):
        return {"error": f"잘못된 날짜형식: s_ym={s_ym}, e_ym={e_ym}"}

    # 1) 세션과 헤더 준비
    session = requests.Session()
    base_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": "https://fruit.nihhs.go.kr",
        "Referer": referer,
        "X-Requested-With": "XMLHttpRequest",
        "Connection": "keep-alive",
    }

    try:
        # 2) 쿠키 선취득
        session.get(referer, headers=base_headers, timeout=20)
    except Exception as e:
        return {"error": f"사전 접속 실패: {e}"}

    # 3) 폼 데이터 구성 (월별: D 기준)
    form = {
        "pageIndex": "1",
        "searchNum": "0",
        "areaName": region_name,
        "areaCode": AREA_CODE.get(region_name, region_name),
        "statmethod": stat_gb_code,  # "D" = 월별
    }
    if stat_gb_code == "A":
        form["wetherDtBgn"] = f"{s_ym}-01"
        form["wetherDtEnd"] = f"{e_ym}-30"
    elif stat_gb_code in ("B", "C"):
        form["wetherDtM"] = s_ym
    elif stat_gb_code == "D":
        # 일부 환경에서 _Bgn/_End 만 있어도 동작하는 경우가 있어 둘 다 전달
        form["wetherDtBgn2"] = s_ym
        form["wetherDtEnd2"] = e_ym
        form["wetherDtBgn"] = s_ym
        form["wetherDtEnd"] = e_ym

    # 4) 리트라이 루프 (최대 3회)
    last_text = ""
    for attempt in range(3):
        try:
            resp = session.post(list_url, data=form, headers=base_headers, timeout=30)
            last_text = resp.text or ""
            ctype = resp.headers.get("Content-Type", "")

            # 4-1) JSON 본문
            if "application/json" in ctype or last_text.strip().startswith("{"):
                try:
                    return resp.json()
                except Exception:
                    # JSON 텍스트가 깨진 경우
                    try:
                        return json.loads(last_text)
                    except Exception:
                        pass

            # 4-2) JSONP 가능성: callback(...) 래핑 제거
            m = re.match(r"^[^(]+\((\s*{.*}\s*)\)\s*;?\s*$", last_text, re.S)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass

            # 4-3) HTML로 떨어진 경우: 디버그 정보 반환 (상위에서 보여줌)
            # 보통은 세션/헤더 문제이거나 서버 일시 오류
            time.sleep(0.8)  # 짧게 대기 후 재시도
        except Exception as e:
            last_text = f"[attempt {attempt+1}] 요청 예외: {e}"
            time.sleep(0.8)
            continue

    # 5) 최종 실패: 앞부분만 돌려 진단
    snippet = re.sub(r"\s+", " ", last_text)[:800]
    return {"error": "JSON 파싱 실패", "raw": snippet, "debug": {"form": form}}


def json_to_dataframe(payload: dict) -> pd.DataFrame:
    if not payload or "result" not in payload:
        return pd.DataFrame()
    res = payload.get("result", [])
    if len(res) == 0:
        return pd.DataFrame()
    method = payload.get("mainAwsVO", {}).get("statmethod")
    raw = pd.DataFrame(res)

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

    if method == "D":
        want = ["월","평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"] \
               if "월" in raw.columns else \
               ["일자","평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]
    else:
        want = ["일자","평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]

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

def agg_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if not {"연도","월"}.issubset(df.columns):
        return df.copy()
    agg_map = {
        "평균기온":"mean","최고기온":"mean","최저기온":"mean","습도":"mean",
        "강우량":"sum","일사량":"sum","결로시간":"sum","평균풍속":"mean","최대풍속":"mean"
    }
    use_cols = {k:v for k,v in agg_map.items() if k in df.columns}
    out = df.groupby(["연도","월"], as_index=False).agg(use_cols)
    return out.sort_values(["연도","월"]).reset_index(drop=True)

def build_wide_month_feats(env_m: pd.DataFrame) -> pd.DataFrame:
    if not {"연도","월"}.issubset(env_m.columns):
        raise ValueError("env_m에 '연도','월' 컬럼 필요")
    num_cols = [c for c in env_m.columns if c not in ("연도","월") and pd.api.types.is_numeric_dtype(env_m[c])]
    mean_agg = env_m.groupby(["연도","월"], as_index=False)[num_cols].mean()
    sum_agg  = env_m.groupby(["연도","월"], as_index=False)[num_cols].sum()
    wide_mean = None
    for m in range(1, 12+1):
        sub = mean_agg[mean_agg["월"] == m].drop(columns=["월"]).copy()
        sub = sub.rename(columns={c: f"{c}_mean_m{m:02d}" for c in num_cols})
        wide_mean = sub if wide_mean is None else pd.merge(wide_mean, sub, on="연도", how="outer")
    wide_sum = None
    for m in range(1, 12+1):
        sub = sum_agg[sum_agg["월"] == m].drop(columns=["월"]).copy()
        sub = sub.rename(columns={c: f"{c}_sum_m{m:02d}" for c in num_cols})
        wide_sum = sub if wide_sum is None else pd.merge(wide_sum, sub, on="연도", how="outer")
    wide = pd.merge(wide_mean, wide_sum, on="연도", how="outer").fillna(0)
    wide = wide.rename(columns={"연도":"year"})
    return wide

def apply_equation_row(row: pd.Series, eq_str: str) -> float:
    rhs = eq_str.split("=", 1)[1].strip().replace("·", "*")
    cols = sorted(row.index.tolist(), key=len, reverse=True)
    expr = rhs
    for c in cols:
        expr = expr.replace(c, f"row[{repr(c)}]")
    return float(eval(expr, {"__builtins__": {}}, {"row": row, "np": np}))

# =========================
# 과실 품질(라벨) 수집
# =========================
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
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://fruit.nihhs.go.kr/apple/qlityInfo_frutQlity.do",
        "Accept": "text/html,application/xhtml+xml",
    }
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()

    # 진단: 응답 길이/연도 출력
    st.info(f"[품질표 GET] year={selyear}, last={lastyear}, cultivar={cultivar}, bytes={len(r.text)}")

    try:
        import io as _io
        tables = pd.read_html(_io.StringIO(r.text), flavor=["lxml","bs4","html5lib"])
    except Exception as e:
        st.warning(f"전년도 과실품질 표 파싱 실패: {e}")
        return {}

    cleaned = []
    for t in tables:
        t2 = normalize_quality_columns(t)
        if set(["지역","수확일자"]).issubset(t2.columns):
            cleaned.append(t2)

    result = {}
    if len(cleaned) >= 1:
        result["this"] = cleaned[0]
    if len(cleaned) >= 2:
        result["last"] = cleaned[1]
    return result


def pick_region_row(qdf: pd.DataFrame, region_disp_name: str) -> Optional[pd.Series]:
    if qdf is None or qdf.empty or "지역" not in qdf.columns:
        return None
    tmp = qdf.copy()
    tmp["지역"] = tmp["지역"].map(_clean_str)
    if "수확일자" in tmp.columns:
        tmp["수확일자"] = pd.to_datetime(tmp["수확일자"], errors="coerce")
    sub = tmp[tmp["지역"] == _clean_str(region_disp_name)]
    if sub.empty: return None
    sub = sub.sort_values("수확일자", ascending=False, na_position="last")
    return sub.iloc[0]

def collect_fruit_labels_from_api(cultivar: str, region_disp: str, start_year: int = 2012, end_year: int = None) -> pd.DataFrame:
    if end_year is None:
        end_year = datetime.now().year - 1
    rows = []
    for y in range(start_year+1, end_year+1):
        try:
            qdict = fetch_quality_tables(y, y-1, cultivar)
            if not qdict or "last" not in qdict or qdict["last"] is None or qdict["last"].empty:
                st.info(f"[라벨] {y}: last 표 없음")
                continue
            q_last = normalize_quality_columns(qdict["last"])
            r = pick_region_row(q_last, region_disp)
            if r is None:
                st.info(f"[라벨] {y}: 지역 '{region_disp}' 행 없음")
                continue
            def take(pat):
                c = get_first_col_by_pattern(r.to_frame().T, pat)
                return None if c is None else pd.to_numeric(r[c], errors="coerce")
            rows.append({
                "year": y,
                "과중": take(r"^과중"),
                "종경": take(r"^종경"),
                "횡경": take(r"^횡경"),
                "경도": take(r"(경도\s*평균|경도평균|N\s*/?\s*ø?\s*11\s*mm)"),
                "당도": take(r"^당도(\s*\((°|˚)?\s*Brix\))?"),
                "산도": take(r"^산도(\s*\(%\))?"),
                "L":    take(r"착색.*Hunter\s*L\b"),
                "a":    take(r"착색.*Hunter\s*a\b"),
                "b":    take(r"착색.*Hunter\s*b\b"),
            })
        except Exception as e:
            st.info(f"[라벨] {y}: 예외 {e}")
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("[라벨] 수집 결과 0행. 연도/지역/품종을 바꾸거나 시작 연도를 더 과거로 설정하세요.")
        return df
    targ = [c for c in ["과중","종경","횡경","경도","당도","산도","L","a","b"] if c in df.columns]
    df = df.dropna(how="all", subset=targ)
    st.success(f"[라벨] 수집 완료: {len(df)}행 (예: 상단 3행)")
    st.dataframe(df.head(3))
    return df


def collect_env_features_from_api(region: str, start_year: int = 2012, end_year: int = None) -> pd.DataFrame:
    if end_year is None:
        end_year = datetime.now().year
    payload = fetch_aws_stat(region, "D", f"{start_year:04d}-01", f"{end_year:04d}-12")

    # 1) JSON 경로
    if "error" not in payload:
        df = json_to_dataframe(payload)
    else:
        # 2) HTML 우회 경로
        st.warning(f"[환경] JSON 실패 → HTML 파싱 우회 시도: {payload.get('error')}")
        raw = payload.get("raw", "")
        if raw:
            st.caption("서버 응답 스니펫(참고용)")
            st.code(raw)
        df = fetch_aws_stat_fallback_html(region, f"{start_year:04d}-01", f"{end_year:04d}-12")

    if df.empty:
        st.error("[환경] 월별 원본이 비어 있습니다.")
        return pd.DataFrame()

    # 월별 집계 후 wide 변환
    env_m = agg_to_monthly(df) if {"연도","월"}.issubset(df.columns) else df
    if env_m.empty:
        st.error("[환경] 월별 집계 결과 0행")
        return pd.DataFrame()

    wide = build_wide_month_feats(env_m)
    if wide.empty:
        st.error("[환경] wide 피처 0행")
        return pd.DataFrame()

    st.success(f"[환경] wide 피처 수집 완료: {len(wide)}행, 컬럼 {len(wide.columns)}개 (예: 상단 3행)")
    st.dataframe(wide.head(3))
    return wide



# =========================
# 간단 자동 재학습(상관 Top-20 + OLS)
# =========================
def fit_auto_equations_from_api(cultivar: str, region: str, start_year: int = 2012):
    region_disp = REGION_NAME_MAP.get(region, region)

    st.info(f"[학습] 수집 시작: 품종={cultivar}, 지역={region}({region_disp}), 시작연도={start_year}")
    labels = collect_fruit_labels_from_api(cultivar, region_disp, start_year=start_year)
    env_wide = collect_env_features_from_api(region, start_year=start_year)

    if labels.empty and env_wide.empty:
        raise RuntimeError("라벨과 환경 피처 모두 비어 있습니다.")
    if labels.empty:
        raise RuntimeError("라벨(전년도 과실표)이 비어 있습니다.")
    if env_wide.empty:
        raise RuntimeError("환경 wide 피처가 비어 있습니다.")

    data = pd.merge(labels, env_wide, on="year", how="inner").dropna()
    st.info(f"[학습] 병합 후 데이터: {len(data)}행, 컬럼 {len(data.columns)}개")
    st.dataframe(data.head(3))

    if len(data) < 8:
        raise RuntimeError("학습 표본이 너무 적습니다(8행 미만). 시작 연도를 더 과거로 설정해 보세요.")

    # 설명변수 풀
    num_cols = [c for c in data.columns if c != "year"]
    env_cols = [c for c in num_cols if re.search("기온|습도|강우|일사|결로|풍속", c)]

    targets = [c for c in ["과중","종경","횡경","경도","당도","산도","L","a","b"] if c in data.columns]
    if not targets:
        raise RuntimeError("타깃 컬럼(과중/경도/당도 등)이 한 개도 없습니다.")

    new_eq = {}
    info_rows = []

    for tgt in targets:
        corr = data[env_cols + [tgt]].corr(numeric_only=True)[tgt].dropna().abs().sort_values(ascending=False)
        if corr.empty:
            st.warning(f"[학습] {tgt}: 유효한 상관 없음 → 스킵")
            continue
        feats = corr.head(20).index.tolist()
        X = data[feats].astype(float)
        y = data[tgt].astype(float)
        Xc = sm.add_constant(X)
        model = sm.OLS(y, Xc).fit()

        parts = [f"{model.params['const']:.6g}"] + [f"{model.params[f]:+.6g}·{f}" for f in feats]
        eq = f"{tgt} = " + " ".join(parts)
        new_eq[tgt] = eq
        info_rows.append([tgt, len(feats), round(model.rsquared, 3), len(data)])

    if not new_eq:
        raise RuntimeError("모든 타깃에서 식 생성 실패(상관/표본 부족).")

    info_df = pd.DataFrame(info_rows, columns=["target","n_feats","R2_in_sample","n_samples"])
    return new_eq, info_df, data


# =========================
# 회귀식 저장/로드
# =========================
def load_equations() -> Dict[str, Dict[str, str]]:
    if EQUATIONS_PATH.exists():
        with open(EQUATIONS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_EQUATIONS.copy()

def save_equations(eqs: Dict[str, Dict[str, str]]):
    with open(EQUATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(eqs, f, ensure_ascii=False, indent=2)

# =========================
# UI
# =========================
st.markdown("<h1 style='text-align:center;'>🍎 사과 과실 품질 예측 (API 자동 재학습)</h1>", unsafe_allow_html=True)

colA, colB, colC = st.columns([1,1,1])
with colA:
    cultivar = st.radio("품종", ["홍로","후지"], horizontal=True)
with colB:
    region = st.selectbox("지역", list(AREA_CODE.keys()), index=1)
with colC:
    start_year = st.number_input("학습 시작 연도(라벨/환경 수집 시작 연도)", min_value=2005, max_value=datetime.now().year-2, value=2012, step=1)

eq_store = load_equations()

# ---- 재학습 버튼
st.markdown("### 🔁 자동 재학습(과거 데이터 수집 → 회귀 적합 → 식 저장)")
if st.button("API만으로 자동 재학습 실행"):
    with st.spinner("과거 라벨/환경 데이터를 API로 수집하고 있습니다..."):
        try:
            new_eq, info_df, train_df = fit_auto_equations_from_api(cultivar, region, start_year=int(start_year))
            st.success("재학습 완료! 아래 정보와 새 회귀식을 확인하세요.")
            st.dataframe(info_df, use_container_width=True)

            # 저장소에 반영
            if cultivar not in eq_store:
                eq_store[cultivar] = {}
            eq_store[cultivar].update(new_eq)
            save_equations(eq_store)

            st.subheader("갱신된 회귀식")
            st.json(new_eq)

        except Exception as e:
            st.error(f"재학습 실패: {e}")

st.markdown("---")
# ---- 예측 버튼
st.markdown("### 🔎 올해 예측 실행(실측+예상 월별 환경 사용)")
run = st.button("자동조회 & 예측")

if run:
    cur_year, cur_month = get_today_ym()
    s_mon, e_mon = cultivar_window(cultivar)
    s_ym = f"{cur_year:04d}-{s_mon:02d}"
    e_ym_real = f"{cur_year:04d}-{min(e_mon, cur_month):02d}"

    with st.spinner("올해 월별 환경 실측 불러오는 중..."):
        payload = fetch_aws_stat(region, "D", s_ym, e_ym_real)
    if "error" in payload:
        st.error("API 응답(JSON) 파싱 실패")
        st.code(payload.get("raw","")[:800])
        st.stop()
    df_cur = json_to_dataframe(payload)

    st.subheader("올해 월별 실측(환경)")
    if df_cur.empty:
        st.warning("올해 실측 데이터가 없어 과거 평균으로만 채웁니다.")
    else:
        st.dataframe(df_cur, use_container_width=True)

    # 과거(기후평년) 확보
    past_payload = fetch_aws_stat(region, "D", f"{max(cur_year-15,2010):04d}-01", f"{cur_year-1:04d}-12")
    past_df = json_to_dataframe(past_payload)
    # 과실데이터 학습 시작 연도 2019년으로 고정
    start_year = 2019
    past_payload = fetch_aws_stat(region, "D", f"{start_year:04d}-01", f"{cur_year-1:04d}-12")
    past_df = json_to_dataframe(past_payload)
    # 2019년부터 현재까지 데이터만 사용
    if not past_df.empty:
        past_df = past_df[past_df["연도"] >= start_year]
    env_all = pd.concat([env_m, past_df], ignore_index=True) if not past_df.empty else env_m
    env_all = env_all[env_all["연도"] >= start_year]

    filled_this_year = fill_missing_or_future_with_climatology(env_all, cur_year, cultivar, mode=mode)

    # 회귀식 로드
    eq_by_cultivar = load_equations()
    equations = eq_by_cultivar.get(cultivar, {})

    if not equations:
        st.warning("현재 저장된 회귀식이 없습니다. 먼저 [자동 재학습]을 실행해 식을 생성/저장하세요.")
        st.stop()

    preds = {}
    for tgt, formula in equations.items():
        try:
            preds[tgt] = apply_equation_row(row, formula)
        except Exception as e:
            preds[tgt] = f"에러: {e}"

    st.subheader(f"예측 결과  |  품종: {cultivar}  |  지역: {region}  |  연도: {cur_year}")
    pred_df = pd.DataFrame([preds]).T.reset_index()
    pred_df.columns = ["항목", "예측값(올해)"]
    st.dataframe(pred_df.set_index("항목").T, use_container_width=True)

    # 다운로드
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ 올해 환경 wide(특징) CSV",
            env_wide.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{region}_{cultivar}_{cur_year}_env_wide.csv",
            mime="text/csv")
    with c2:
        st.download_button("⬇️ 예측 결과 CSV",
            pred_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{region}_{cultivar}_{cur_year}_preds.csv",
            mime="text/csv")
else:
    st.info("① 품종/지역/학습시작연도 선택 → ② [API만으로 자동 재학습 실행] → ③ [자동조회 & 예측] 순서로 이용하세요.")

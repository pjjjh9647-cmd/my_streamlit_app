# -*- coding: utf-8 -*-
# 파일명: 예측_2025_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path

st.set_page_config(page_title="사과 과실품질 2025 예측", layout="wide")
st.title("사과 과실품질 2025 예측 (홍로·후지)")

# -----------------------------
# 경로 설정
# -----------------------------
BASE_ENV = Path(r"C:\Users\User\Desktop\mba\환경데이터")
FILE_INTEGRATED = BASE_ENV / "기상데이터_통합.xlsx"        # 평년 계산용
DIR_2025 = BASE_ENV / "2025"                              # 2025년 엑셀 5개가 들어있는 폴더
OUTDIR = BASE_ENV / "_OUT_2025"
OUTDIR.mkdir(exist_ok=True)

# -----------------------------
# 회귀식 (질문 제공 원문 그대로)
# -----------------------------
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

# -----------------------------
# 유틸: 엑셀 로딩
# -----------------------------
NUM_COLS_KOR = [
    "평균기온","최고기온","최저기온","강우량","일사량","습도","평균풍속","최대풍속","결로시간"
]
RENAME_STD = {  # 내부 표준 키 (영문) → 한국어 원본 컬럼을 그대로 쓰되, 가공 시 사용
    "tmean": "평균기온",
    "tmax": "최고기온",
    "tmin": "최저기온",
    "prcp": "강우량",
    "rad": "일사량",
    "humid": "습도",
    "wind_mean": "평균풍속",
    "wind_gust": "최대풍속",
    "condense": "결로시간",
}

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

REGION_CANDIDATES = ["지역명","지역","지점명","지점","관측소","지역코드","행정구역"]

def _try_read_with_various_headers(xlsx: Path):
    # header 행을 0,1,2 순서로 시도
    for h in [0,1,2]:
        try:
            book = pd.read_excel(xlsx, sheet_name=None, header=h)
            # 최소한 열이 있는 시트만
            ok = {sn:df for sn,df in book.items() if isinstance(df, pd.DataFrame) and not df.empty}
            if ok:
                return ok, h
        except Exception:
            pass
    # 마지막 시도로 header=None 후 첫 행을 헤더로 승격
    try:
        book = pd.read_excel(xlsx, sheet_name=None, header=None)
        ok = {}
        for sn,df in book.items():
            if df is None or df.empty: 
                continue
            df = df.copy()
            df.columns = df.iloc[0].astype(str).str.strip()
            df = df.iloc[1:].reset_index(drop=True)
            ok[sn] = df
        if ok:
            return ok, "auto"
    except Exception:
        pass
    return {}, None

def _infer_region_col(cols: list[str]) -> str|None:
    cols_norm = [str(c).strip() for c in cols]
    for cand in REGION_CANDIDATES:
        if cand in cols_norm:
            return cand
    # 비슷한 이름(공백 포함 등) 탐색
    joined = " ".join(cols_norm)
    for cand in REGION_CANDIDATES:
        for c in cols_norm:
            if cand.replace(" ","") == str(c).replace(" ",""):
                return c
    return None

def _infer_region_value_from_context(xlsx: Path, sheet_name: str) -> str|None:
    # 파일명/시트명에서 한글 연속 문자열 추정
    fname = xlsx.stem
    # 우선순위: 시트명 -> 파일명
    for s in [sheet_name, fname]:
        m = re.search(r'([가-힣]{2,})', str(s))
        if m:
            return m.group(1)
    return None

def read_all_sheets(xlsx: Path) -> pd.DataFrame:
    book, used_header = _try_read_with_various_headers(xlsx)
    frames = []
    for sname, df in book.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        # 지역·일자 컬럼 결정
        region_col = _infer_region_col(df.columns.tolist())
        date_col = None
        for c in ["일자","날짜","DATE","date","일시"]:
            if c in df.columns:
                date_col = c
                break
        if date_col is None:
            # 날짜가 없으면 이 시트는 스킵
            continue

        g = pd.DataFrame()
        # region
        if region_col:
            g["region"] = df[region_col].astype(str).str.strip()
        else:
            # 지역 컬럼이 없으면 컨텍스트로 추정
            guessed = _infer_region_value_from_context(xlsx, sname)
            if guessed is None:
                # 이 파일·시트는 지역을 알 수 없으므로 스킵
                continue
            g["region"] = guessed

        # date
        g["date"] = pd.to_datetime(df[date_col], errors="coerce")
        g = g.dropna(subset=["date"]).copy()

        # 수치 컬럼들
        for k in ["평균기온","최고기온","최저기온","강우량","일사량","습도","평균풍속","최대풍속","결로시간"]:
            if k in df.columns:
                g[k] = pd.to_numeric(df[k], errors="coerce")

        if len(g):
            frames.append(g)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    # 지역명 공백 정리
    out["region"] = out["region"].astype(str).str.replace(r"\s+", "", regex=True)
    return out

def read_folder_2025(dirpath: Path) -> pd.DataFrame:
    files = sorted([p for p in dirpath.glob("*.xlsx") if p.is_file()])
    frames = []
    meta = []
    for f in files:
        df = read_all_sheets(f)
        if not df.empty:
            df["__source_file"] = f.name
            frames.append(df)
            meta.append((f.name, df.columns.tolist(), len(df)))
    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame()

    # 디버그용: 어떤 파일에서 무엇을 읽었는지 화면에 표시
    with st.expander("파일별 로드 요약", expanded=False):
        if meta:
            for name, cols, n in meta:
                st.write(f"{name} → {n} rows, cols: {cols}")
        else:
            st.write("읽힌 파일이 없습니다.")
    return out


# -----------------------------
# 월별 집계와 평년 계산
# -----------------------------
MEAN_VARS = ["평균기온","최고기온","최저기온","습도","평균풍속","최대풍속","결로시간","일사량"]
SUM_VARS  = ["강우량","일사량"]  # 일사량_sum도 필요하므로 sum에도 포함

def monthly_agg(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    gb = df.groupby(["region","year","month"], as_index=False)
    out = gb.agg({**{v:"mean" for v in MEAN_VARS if v in df.columns},
                  **{v:"sum"  for v in SUM_VARS  if v in df.columns}})
    return out

def compute_normals_monthly(integrated_df: pd.DataFrame) -> pd.DataFrame:
    # 월별 평년: 평균형은 월평균의 평균, 합계형은 월합의 평균
    m = monthly_agg(integrated_df)
    # 평균형
    mean_cols = [v for v in MEAN_VARS if v in m.columns]
    sum_cols  = [v for v in SUM_VARS  if v in m.columns]
    cols = ["region","month"] + mean_cols + sum_cols
    m = m[cols].copy()
    normals = (m.groupby(["region","month"], as_index=False)
                 .agg({**{v:"mean" for v in mean_cols},
                       **{v:"mean" for v in sum_cols}}))
    normals["year"] = 2025  # 채움 대상 연도표시용
    return normals

# -----------------------------
# 2025 데이터 보정(빈 달을 평년으로 채움)
# -----------------------------
def fill_missing_months_with_normals(m2025: pd.DataFrame, normals: pd.DataFrame, region: str) -> pd.DataFrame:
    # 해당 region만
    a = m2025[m2025["region"]==region].copy()
    # 월 1~12 강제 프레임
    all_months = pd.DataFrame({"region":[region]*12, "year":[2025]*12, "month":list(range(1,13))})
    a = all_months.merge(a, on=["region","year","month"], how="left")
    n = normals[normals["region"]==region].copy()
    a = a.merge(n, on=["region","month"], how="left", suffixes=("","_norm"))
    # 채움 규칙: 합계형은 *_norm로 대체, 평균형도 *_norm로 대체
    for v in set(MEAN_VARS+SUM_VARS):
        if v in a.columns and f"{v}_norm" in a.columns:
            a[v] = a[v].where(a[v].notna(), a[f"{v}_norm"])
            a.drop(columns=[f"{v}_norm"], inplace=True)
    return a

# -----------------------------
# 회귀식 파서 및 평가
# -----------------------------
term_pat = re.compile(r'([+\-]?\s*[0-9.eE]+)\s*·\s*([^\s+]+)')

def parse_equation(eq: str):
    # 좌변=우변 분리
    if "=" in eq:
        rhs = eq.split("=",1)[1]
    else:
        rhs = eq
    rhs = rhs.strip()
    # 상수항
    # 예: "667.243 -0.59·X + ..." → 앞의 숫자 추출
    m0 = re.match(r'^\s*([+\-]?\s*[0-9.eE]+)', rhs)
    intercept = float(m0.group(1).replace(" ","")) if m0 else 0.0
    # 항목들
    terms = term_pat.findall(rhs)
    parsed = []
    for coef_str, varname in terms:
        c = float(coef_str.replace(" ",""))
        v = varname.strip()
        parsed.append((c, v))
    return intercept, parsed

def build_feature_dict(monthly_df: pd.DataFrame) -> dict:
    # 월별 집계에서 식에서 요구하는 이름 규칙으로 딕셔너리 구성
    # 예: 평균풍속_mean_m06, 강우량_sum_m07, 일사량_sum_m04 등
    feat = {}
    bym = monthly_df.set_index("month")
    for mm in range(1,13):
        if mm in bym.index:
            row = bym.loc[mm]
            # 평균형
            for v in ["평균기온","최고기온","최저기온","습도","평균풍속","최대풍속","결로시간","일사량"]:
                if v in monthly_df.columns:
                    feat[f"{v}_mean_m{mm:02d}"] = float(row.get(v,np.nan))
            # 합계형
            for v in ["강우량","일사량"]:
                col = v
                if col in monthly_df.columns:
                    feat[f"{v}_sum_m{mm:02d}"] = float(row.get(col,np.nan))
    return feat

def evaluate_equation(eq: str, feat: dict) -> float:
    intercept, terms = parse_equation(eq)
    val = intercept
    for c, v in terms:
        val += c * float(feat.get(v, np.nan))
    return val

# -----------------------------
# UI: 데이터 로드
# -----------------------------
# 데이터 로드
df_integrated = read_all_sheets(FILE_INTEGRATED)
df_2025 = read_folder_2025(DIR_2025)

if df_integrated.empty:
    st.error("기상데이터_통합.xlsx에서 데이터를 읽지 못했습니다. 시트 헤더/컬럼명을 확인하세요.")
    st.stop()
if df_2025.empty:
    st.error("2025 폴더에서 유효한 데이터를 읽지 못했습니다. 파일/시트의 헤더와 컬럼명을 확인하세요.")
    st.stop()

# region 필수 점검
if "region" not in df_2025.columns:
    st.error(f"2025 데이터에 region 컬럼이 없습니다. 파일명/시트명에서 지역 추정을 하지 못했습니다. "
             f"파일 이름에 지역명이 포함되도록 수정하거나, 엑셀에 '지역명' 컬럼을 추가하세요.")
    st.write("df_2025 columns:", df_2025.columns.tolist())
    st.stop()

regions_2025 = sorted([r for r in df_2025["region"].dropna().unique().tolist() if str(r).strip()!=""])
if not regions_2025:
    st.error("2025 데이터에서 유효한 지역명이 없습니다. 파일명/시트명/컬럼을 확인하세요.")
    st.dataframe(df_2025.head(30))
    st.stop()


regions_2025 = sorted(df_2025["region"].dropna().unique().tolist())
region = st.selectbox("지역 선택", options=regions_2025, index=0)

cultivar = st.radio("품종", options=["홍로","후지"], index=0, horizontal=True)

# -----------------------------
# 집계 및 평년보정
# -----------------------------
m_2025_all = monthly_agg(df_2025)
normals_all = compute_normals_monthly(df_integrated)

m_2025_region = fill_missing_months_with_normals(m_2025_all, normals_all, region)

st.subheader("2025 월별 집계(빈 달은 평년으로 채움)")
show_cols = ["region","year","month"] + [c for c in m_2025_region.columns if c not in ("region","year","month")]
st.dataframe(m_2025_region[show_cols], use_container_width=True)

# -----------------------------
# 예측 실행
# -----------------------------
st.markdown("---")
st.subheader(f"{cultivar} 2025 예측 결과")

feat_dict = build_feature_dict(m_2025_region[["region","year","month"] + [c for c in m_2025_region.columns if c not in ("region","year","month")]])

equations = EQUATIONS_BY_CULTIVAR[cultivar]
pred_rows = []
for target, eq in equations.items():
    try:
        yhat = evaluate_equation(eq, feat_dict)
    except Exception:
        yhat = np.nan
    pred_rows.append({"품종":cultivar, "지역":region, "연도":2025, "지표":target, "예측값":yhat})

pred_df = pd.DataFrame(pred_rows)
st.dataframe(pred_df, use_container_width=True)

# 저장
csv_path = OUTDIR / f"예측_{cultivar}_{region}_2025.csv"
pred_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
st.success(f"CSV 저장: {csv_path}")

# 참고로 어떤 설명변수가 실제로 사용되었는지 표로 제공
st.markdown("---")
st.subheader("사용된 설명변수 값 미리보기")
need_vars = []
for eq in equations.values():
    _, terms = parse_equation(eq)
    for _, v in terms:
        if v not in need_vars:
            need_vars.append(v)
need_preview = {k: feat_dict.get(k, np.nan) for k in need_vars}
need_df = pd.DataFrame([need_preview]).T.reset_index()
need_df.columns = ["변수명","대입값(2025 또는 평년)"]
st.dataframe(need_df, use_container_width=True)

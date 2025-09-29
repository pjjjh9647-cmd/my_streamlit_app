# -*- coding: utf-8 -*-
r"""
군위(군위군)만 사용하여 (논문식 접근)
- 월별 기상 피처 + BIO19, 과실 타깃을 '연도' 기준으로 병합
- 품종별로 분리하여 타깃 vs (월별기상+BIO) 상관 → 공선성 제거(|r|>=0.75) → 상위 3개 변수만 선택
- 품종×타깃별 OLS 적합 (모델식/지표 저장)
- 결과는 _ANALYSIS_OUT 폴더에 저장

실행:
  & "C:\Users\User\AppData\Local\Programs\Python\Python310\python.exe" "C:\newvariety\run_gunwi_analysis.py"

필요 패키지:
  pip install pandas numpy statsmodels matplotlib seaborn openpyxl
"""
import warnings
warnings.filterwarnings("ignore")

import re
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm

# ---------------------------------------------------------
# 경로
# ---------------------------------------------------------
BASE = Path(r"C:\Users\User\Desktop\mba\환경데이터")
BIO_FILE   = BASE / "_OUT" / "bioclim_19_variables_Gunwi.csv"
ENV_FILE   = BASE / "기상데이터_통합.xlsx"
FRUIT_FILE = BASE / "과실데이터_통합.xlsx"

OUTDIR = BASE / "_ANALYSIS_OUT"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 공용 함수
# ---------------------------------------------------------
def save_df(df, path):
    if isinstance(df, pd.DataFrame) and not df.empty:
        df.to_csv(path, encoding="utf-8-sig", index=False)
        print(f"[저장] {path}")
    else:
        print(f"[SKIP] 비어있어 저장 안함: {path.name}")

def to_year_series(s: pd.Series) -> pd.Series:
    """여러 형태(문자 YYYY, YYYY-MM-DD, YYYYMMDD 정수/문자, 엑셀 직렬일)를 연도로 변환"""
    s2 = s.copy()
    # 연도만(YYYY)
    m_year = s2.astype(str).str.fullmatch(r"\s*(19|20)\d{2}\s*")
    if m_year.fillna(False).all():
        return pd.to_numeric(s2, errors="coerce")

    # YYYYMMDD
    def looks_like_yyyymmdd(x):
        try:
            xs = str(int(float(x))).strip()
        except Exception:
            xs = str(x)
        return bool(re.fullmatch(r"(19|20)\d{6}", xs))
    mask_ymd = s2.apply(looks_like_yyyymmdd)
    if mask_ymd.any():
        tmp = s2[mask_ymd].apply(lambda x: str(int(float(x))).strip())
        y = pd.to_datetime(tmp, format="%Y%m%d", errors="coerce").dt.year
        s2.loc[mask_ymd] = y

    # 엑셀 직렬일
    def looks_like_excel_serial(x):
        try:
            v = float(x)
            return 10000 <= v <= 60000
        except Exception:
            return False
    mask_serial = s2.apply(looks_like_excel_serial)
    if mask_serial.any():
        y = pd.to_datetime(s2[mask_serial].astype(float), unit="D", origin="1899-12-30", errors="coerce").dt.year
        s2.loc[mask_serial] = y

    # 일반 날짜
    s2 = pd.to_datetime(s2, errors="coerce").dt.year
    return pd.to_numeric(s2, errors="coerce").astype("Int64")

def extract_gunwi(df):
    if "지역명" not in df.columns:
        return None
    mask = df["지역명"].astype(str).str.contains("군위", na=False)
    out = df.loc[mask].copy()
    return out if not out.empty else None

def find_cultivar_col(df: pd.DataFrame):
    # 품종 컬럼 자동 탐지
    candidates = [c for c in df.columns if any(k in str(c) for k in ["품종","품종명","품종코드","품종구분"])]
    return candidates[0] if candidates else None

# ---------------------------------------------------------
# 1) 로딩
# ---------------------------------------------------------
print("\n[1] 파일 로딩...")
bio = pd.read_csv(BIO_FILE)
env_all   = pd.read_excel(ENV_FILE,   sheet_name=None)
fruit_all = pd.read_excel(FRUIT_FILE, sheet_name=None)

env_frames   = [x for x in (extract_gunwi(df) for df in env_all.values())   if x is not None]
fruit_frames = [x for x in (extract_gunwi(df) for df in fruit_all.values()) if x is not None]
if not env_frames:
    raise SystemExit("기상데이터에서 '군위' 행을 찾지 못했습니다.")
if not fruit_frames:
    raise SystemExit("과실데이터에서 '군위' 행을 찾지 못했습니다.")

env_raw   = pd.concat(env_frames, ignore_index=True)
fruit_raw = pd.concat(fruit_frames, ignore_index=True)

# ---------------------------------------------------------
# 2) 월별 기상 피처 생성 (군위)
#    - 연도, 월 기준 집계: tmean/tmax/tmin: mean, prcp: sum, rad/humid: mean
#    - 피벗: tmean_m04, prcp_m09 등
# ---------------------------------------------------------
print("[2] 월별 기상 피처 생성...")
if "일자" not in env_raw.columns:
    raise SystemExit("기상데이터에 '일자' 컬럼이 필요합니다.")

env_raw["date"]  = pd.to_datetime(env_raw["일자"], errors="coerce")
env_raw["year"]  = env_raw["date"].dt.year
env_raw["month"] = env_raw["date"].dt.month

# 숫자 변환
for col, agg in [("평균기온","mean"), ("최고기온","mean"), ("최저기온","mean"), ("강우량","sum")]:
    if col in env_raw.columns:
        env_raw[col] = pd.to_numeric(env_raw[col], errors="coerce")
if "일사량" in env_raw.columns: env_raw["일사량"] = pd.to_numeric(env_raw["일사량"], errors="coerce")
if "습도"  in env_raw.columns: env_raw["습도"]  = pd.to_numeric(env_raw["습도"],  errors="coerce")

agg_map = {"평균기온":"mean","최고기온":"mean","최저기온":"mean","강우량":"sum"}
if "일사량" in env_raw.columns: agg_map["일사량"] = "mean"
if "습도"  in env_raw.columns: agg_map["습도"]  = "mean"

env_monthly = (env_raw.dropna(subset=["year","month"])
                      .groupby(["year","month"], as_index=False)
                      .agg(agg_map)
                      .rename(columns={"평균기온":"tmean","최고기온":"tmax","최저기온":"tmin",
                                       "강우량":"prcp","일사량":"rad","습도":"humid"}))

# 피벗: 변수×월 → 컬럼
def wide_month(df_sub: pd.DataFrame, cols=("tmean","tmax","tmin","prcp","rad","humid")):
    cols = [c for c in cols if c in df_sub.columns]
    if not cols: return pd.DataFrame()
    w = df_sub.pivot_table(index="year", columns="month", values=cols)
    w.columns = [f"{v}_m{m:02d}" for v, m in w.columns.to_flat_index()]
    return w.reset_index()

env_mwide = wide_month(env_monthly)

# ---------------------------------------------------------
# 3) BIO (군위만, 연도 평균)
# ---------------------------------------------------------
print("[3] BIO 집계...")
bio_gw = bio[bio["region"].astype(str).str.contains("군위", na=False)].copy()
bio_gw["year"] = pd.to_numeric(bio_gw["year"], errors="coerce").astype("Int64")
bio_y = bio_gw.groupby("year", as_index=False).mean(numeric_only=True)

# ---------------------------------------------------------
# 4) 과실: 연도, 품종별 집계 (숫자만 평균)
# ---------------------------------------------------------
print("[4] 과실 연도×품종 집계...")
# 연도 파싱
if "year" in fruit_raw.columns:
    fruit_raw["year"] = to_year_series(fruit_raw["year"])
elif "연도" in fruit_raw.columns:
    fruit_raw["year"] = to_year_series(fruit_raw["연도"])
elif "일자" in fruit_raw.columns:
    fruit_raw["year"] = to_year_series(fruit_raw["일자"])
else:
    raise SystemExit("과실데이터에서 '연도/일자'를 찾지 못했습니다.")

cultivar_col = find_cultivar_col(fruit_raw)
if cultivar_col is None:
    print("[경고] 과실데이터에 품종 식별 컬럼(예: 품종/품종명)이 보이지 않습니다. 전부 하나로 간주합니다.")
    fruit_raw["_품종임시"] = "ALL"
    cultivar_col = "_품종임시"

drop_keys = {"지역명","일자","연도","year",cultivar_col}
fruit_conv = fruit_raw.copy()
for c in fruit_conv.columns:
    if c not in drop_keys:
        fruit_conv[c] = pd.to_numeric(fruit_conv[c], errors="coerce")
num_cols = [c for c in fruit_conv.columns if c not in drop_keys and pd.api.types.is_numeric_dtype(fruit_conv[c])]
if not num_cols:
    raise SystemExit("과실데이터에서 집계할 숫자 컬럼이 없습니다.")

fruit_agg = (fruit_conv.dropna(subset=["year"])
                        .groupby([cultivar_col,"year"], as_index=False)[num_cols].mean())

# ---------------------------------------------------------
# 5) 병합: (과실: 연도×품종) ←→ (BIO: 연도) ←→ (월별기상wide: 연도)
# ---------------------------------------------------------
print("[5] 병합/디버그...")
merged_all = (fruit_agg
              .merge(bio_y, on="year", how="inner")
              .merge(env_mwide, on="year", how="left"))

# 디버그용 연도 분포
bio_years   = sorted(map(int, pd.Series(bio_y["year"]).dropna().unique()))
env_years   = sorted(map(int, pd.Series(env_mwide["year"]).dropna().unique()))
fruit_years = sorted(map(int, pd.Series(fruit_agg["year"]).dropna().unique()))
common_years = sorted(set(bio_years) & set(env_years) & set(fruit_years))
pd.DataFrame({"BIO":bio_years}).to_csv(OUTDIR/"_디버그_BIO_years.csv", encoding="utf-8-sig", index=False)
pd.DataFrame({"ENV":env_years}).to_csv(OUTDIR/"_디버그_ENV_years.csv", encoding="utf-8-sig", index=False)
pd.DataFrame({"FRUIT":fruit_years}).to_csv(OUTDIR/"_디버그_FRUIT_years.csv", encoding="utf-8-sig", index=False)
print("BIO year:", bio_years)
print("ENV year:", env_years)
print("FRUIT year:", fruit_years)
print("교집합:", common_years)

# ---------------------------------------------------------
# 6) 품종별 분석 (상관→공선성 제거→최대 3변수 선택→OLS)
# ---------------------------------------------------------
print("[6] 품종별 분석 시작...")
by_cultivar_dir = OUTDIR / "_BY_CULTIVAR"
by_cultivar_dir.mkdir(exist_ok=True)

# ✅ 품종별 허용 월 범위 정의
month_range = {
    "홍로": list(range(5, 10)),   # 5~9월
    "후지": list(range(5, 11)),   # 5~10월
}

all_summ_lines = []
for cultivar, g in merged_all.groupby(cultivar_col):
    subdir = by_cultivar_dir / str(cultivar)
    subdir.mkdir(parents=True, exist_ok=True)

    g = g[g["year"].isin(common_years)].copy()
    if g.empty:
        print(f" - [{cultivar}] 공통 연도 없음 → 스킵")
        continue

    # 후보 기상/월별/BIO 컬럼
    valid_months = month_range.get(str(cultivar), list(range(5, 10)))  # 기본 5~9월
    month_suffix = [f"m{m:02d}" for m in valid_months]

    climate_cols = [
        c for c in g.columns
        if c.startswith("BIO") or any(c.endswith(suf) for suf in month_suffix)
    ]

    # 타깃(과실) 컬럼
    fruit_cols_cand = [c for c in num_cols if c in g.columns]
    if not fruit_cols_cand:
        print(f" - [{cultivar}] 과실 타깃 없음 → 스킵")
        continue

    # 상관행렬
    corr = g.corr(numeric_only=True)
    # 저장(전체 상관 테이블)
    save_df(corr, subdir/"상관_전체_테이블.csv")

    # 타깃-기상 상관만 발췌
    fruit_cols_use   = [c for c in fruit_cols_cand if c in corr.index]
    climate_cols_use = [c for c in climate_cols if c in corr.columns]
    if not fruit_cols_use or not climate_cols_use:
        print(f" - [{cultivar}] 상관 추출 대상 컬럼 없음 → 스킵")
        continue

    corr_tg = corr.loc[fruit_cols_use, climate_cols_use].copy()
    save_df(corr_tg, subdir/"상관_타깃vs기상.csv")

    # (논문 느낌) 임계치를 단계적으로 낮춰 후보 확보
    selected_per_target = {}
    for tgt in fruit_cols_use:
        got = None
        for TH in [0.60, 0.55, 0.50]:
            cm = corr_tg.loc[tgt].abs().sort_values(ascending=False)
            cand = cm[cm >= TH].index.tolist()
            if cand:
                got = cand
                break
        selected_per_target[tgt] = got or []

    # 공선성 제거(|r|>=0.75) + 상위 3개 제한 → 각 타깃별 최종 선택
    final_sel = {}
    for tgt, cands in selected_per_target.items():
        cands = [c for c in cands if c in g.columns]
        # 공선성 제거
        keep = []
        for v in cands:
            ok = True
            for u in keep:
                r = abs(g[[u, v]].corr(numeric_only=True).iloc[0,1])
                if r >= 0.75:
                    ok = False
                    break
            if ok:
                keep.append(v)
            if len(keep) >= 3:  # 최대 3개
                break
        final_sel[tgt] = keep

    # 선택 결과 요약 저장
    rows = []
    for tgt, vars_ in final_sel.items():
        rows.append({"target": tgt, "selected_vars": ", ".join(vars_) if vars_ else ""})
    save_df(pd.DataFrame(rows), subdir/"선택변수_요약.csv")

    # OLS 적합 (타깃별)
    ols_lines = [f"=== 품종: {cultivar} ==="]
    for tgt, Xcols in final_sel.items():
        if not Xcols:
            ols_lines.append(f"[{tgt}] 선택변수 없음 → 회귀 생략")
            continue
        df = g[["year", tgt] + Xcols].dropna()
        if len(df) < 5:
            ols_lines.append(f"[{tgt}] 표본 부족(n={len(df)}) → 회귀 생략")
            continue
        model = sm.OLS(df[tgt], sm.add_constant(df[Xcols])).fit()
        line = (f"\n--- 타깃: {tgt} ---\n"
                f"n={len(df)}, R²={model.rsquared:.3f}, adj.R²={model.rsquared_adj:.3f}, "
                f"F={model.fvalue:.3f}, p(F)={model.f_pvalue:.3g}\n"
                f"계수(유의성):\n{model.params.to_string()}\n\n"
                f"p-values:\n{model.pvalues.to_string()}\n")
        # 모델식 출력(사인/반올림)
        betas = model.params
        terms = [f"{betas['const']:.4g}"]
        for x in Xcols:
            coef = betas[x]
            sign = " + " if coef >= 0 else " - "
            terms.append(f"{sign}{abs(coef):.4g}·{x}")
        formula = f"모델식: {tgt} ≈ " + "".join(terms)
        line += formula + "\n"
        ols_lines.append(line)

    # 저장
    with open(subdir/"OLS_요약.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(ols_lines))
    print(f"[저장] {subdir/'OLS_요약.txt'}")

    all_summ_lines.extend(ols_lines)

# 전체 묶음 요약
with open(OUTDIR/"_품종별_OLS_전체요약.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(all_summ_lines))
print(f"[저장] {OUTDIR/'_품종별_OLS_전체요약.txt'}")

print("\n[완료] 결과 폴더:", OUTDIR)
for p in sorted(OUTDIR.glob("**/*")):
    if p.is_file():
        print(" -", p)

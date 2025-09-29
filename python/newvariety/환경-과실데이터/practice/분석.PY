# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# ===== 경로 설정 =====
ENV_FILE   = r"C:\Users\User\Desktop\환경데이터\기상데이터_통합.xlsx"
FRUIT_FILE = r"C:\Users\User\Desktop\과실데이터\과실데이터_통합.xlsx"
OUTDIR     = Path(r"C:\Users\User\Desktop\분석결과")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ===== 환경 변수 목록 =====
ENV_COLS = [
    "평균기온","최고기온","최저기온",
    "습도","강우량","일사량","결로시간",
    "평균풍속","최대풍속"
]

# ===== 타깃 변수 목록(과실 특성) 필요에 맞게 추가/수정 =====
TARGETS = ["당도","과중","산도","L","a","b"]

# ===== 1) 환경데이터 월 단위 요약 =====
def build_env_monthly(env_file: str) -> pd.DataFrame:
    env = pd.read_excel(env_file)
    # 필수 컬럼 존재 확인
    req_cols = {"지역명","일자"} | set(ENV_COLS)
    missing = req_cols - set(env.columns)
    if missing:
        raise ValueError(f"환경데이터에 다음 컬럼이 없습니다: {sorted(missing)}")

    env["일자"] = pd.to_datetime(env["일자"], errors="coerce")
    if env["일자"].isna().all():
        raise ValueError("환경데이터 일자 컬럼을 datetime으로 변환하지 못했습니다")

    env["연도"] = env["일자"].dt.year
    env["월"]   = env["일자"].dt.month

    # 집계 방식 정의
    agg_map = {
        "평균기온":"mean",
        "최고기온":"mean",
        "최저기온":"mean",
        "습도":"mean",
        "강우량":"sum",
        "일사량":"sum",
        "결로시간":"sum",
        "평균풍속":"mean",
        "최대풍속":"mean"
    }

    env_month = env.groupby(["지역명","연도","월"], as_index=False).agg(agg_map)
    return env_month

# ===== 2) 과실데이터 로드(연도 정리) =====
def load_fruit_yearly(fruit_file: str) -> pd.DataFrame:
    fruit = pd.read_excel(fruit_file)

    # 지역명, 품종 필수
    req_cols = {"지역명","품종"}
    if not req_cols.issubset(fruit.columns):
        raise ValueError(f"과실데이터에 다음 컬럼이 필요합니다: {sorted(req_cols)}")

    # 연도 만들기
    # 과실파일의 일자 컬럼이 연도 숫자 또는 문자열이라면 그대로 사용
    year_col = None
    if "연도" in fruit.columns:
        year_col = "연도"
    elif "일자" in fruit.columns:
        # 일자가 YYYY 또는 날짜일 수 있으므로 변환 시도
        # 1) 우선 연도로 강제 변환
        yr = pd.to_numeric(fruit["일자"], errors="coerce")
        if yr.notna().any():
            fruit["연도"] = yr.astype("Int64")
            year_col = "연도"
        else:
            # 2) 날짜로 파싱 후 연도 추출
            dt = pd.to_datetime(fruit["일자"], errors="coerce")
            if dt.notna().any():
                fruit["연도"] = dt.dt.year
                year_col = "연도"
    if year_col is None:
        raise ValueError("과실데이터에서 연도 정보를 만들 수 없습니다. 연도 또는 일자 컬럼을 확인해 주세요")

    return fruit

# ===== 3) 월 단위 환경 + 연 단위 과실 병합 =====
def build_merged(env_month: pd.DataFrame, fruit: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(env_month, fruit, on=["지역명","연도"], how="left")
    # 품종 결측 행 제거(필요 시 유지 가능)
    if "품종" in merged.columns:
        merged = merged[merged["품종"].notna()].copy()
    return merged

# ===== 4) 품종 × 월별 OLS 회귀 =====
def run_monthly_ols(merged: pd.DataFrame, targets, env_cols, outdir: Path):
    coef_rows   = []
    metric_rows = []

    # 품종 목록
    if "품종" not in merged.columns:
        raise ValueError("병합 데이터에 품종 컬럼이 없습니다")

    cultivars = sorted(merged["품종"].dropna().unique().tolist())
    if not cultivars:
        raise ValueError("품종 값이 비어 있습니다")

    # 월 1~12 전체 품종 반복
    for cultivar in cultivars:
        gdf = merged[merged["품종"] == cultivar].copy()
        for target in targets:
            if target not in gdf.columns:
                continue
            for m in range(1, 13):
                df_m = gdf[gdf["월"] == m].copy()
                cols_exist = [c for c in env_cols if c in df_m.columns]
                if not cols_exist:
                    continue

                # 결측 제거
                data = df_m[cols_exist + [target]].dropna()
                # 표본 최소 확보 규칙
                if len(data) < max(10, len(cols_exist) + 5):
                    # 표본이 너무 적으면 불안정 → 생략
                    continue

                X = sm.add_constant(data[cols_exist], has_constant="add")
                y = data[target]
                try:
                    model = sm.OLS(y, X).fit()
                except Exception:
                    continue

                coef_dict = model.params.to_dict()

                # 회귀식 문자열
                def fmt(v): return f"{v:.6g}"
                eq = f"{target} = {fmt(coef_dict.get('const', 0.0))}"
                for name in cols_exist:
                    if name in coef_dict:
                        sign = "+" if coef_dict[name] >= 0 else ""
                        eq += f" {sign}{fmt(coef_dict[name])}*{name}"

                # 콘솔 로그
                print(f"[품종 {cultivar}] 월 {m:02d} 타깃 {target}  N={len(data)}  Adj.R2={model.rsquared_adj:.3f}")
                print(f"식  {eq}")

                # 계수 저장
                for name, val in coef_dict.items():
                    coef_rows.append({
                        "품종": cultivar,
                        "타깃": target,
                        "월": m,
                        "변수": name,
                        "계수": float(val),
                        "p_value": float(model.pvalues.get(name, np.nan)),
                        "N": int(len(data))
                    })

                # 성능 저장
                metric_rows.append({
                    "품종": cultivar,
                    "타깃": target,
                    "월": m,
                    "N": int(len(data)),
                    "R2": float(model.rsquared),
                    "Adj_R2": float(model.rsquared_adj),
                    "AIC": float(model.aic),
                    "BIC": float(model.bic)
                })

    # DataFrame 저장
    if coef_rows:
        coef_df = pd.DataFrame(coef_rows).sort_values(["품종","타깃","월","변수"])
        coef_path = outdir / "월별_품종별_회귀계수.csv"
        coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")
        print(f"회귀계수 저장  {coef_path}")
    else:
        print("저장할 회귀계수가 없습니다")

    if metric_rows:
        metric_df = pd.DataFrame(metric_rows).sort_values(["품종","타깃","월"])
        metric_path = outdir / "월별_품종별_성능요약.csv"
        metric_df.to_csv(metric_path, index=False, encoding="utf-8-sig")
        print(f"성능요약 저장  {metric_path}")
    else:
        print("저장할 성능요약이 없습니다")

# ===== 메인 =====
def main():
    # 1. 월 단위 환경 요약
    env_month = build_env_monthly(ENV_FILE)

    # 2. 과실데이터 로드
    fruit_year = load_fruit_yearly(FRUIT_FILE)

    # 3. 병합
    merged = build_merged(env_month, fruit_year)

    # 참고용으로 병합 결과 저장
    merged_path = OUTDIR / "병합데이터_월단위.xlsx"
    merged.to_excel(merged_path, index=False)
    print(f"병합데이터 저장  {merged_path}")

    # 4. 회귀 실행
    run_monthly_ols(merged, TARGETS, ENV_COLS, OUTDIR)

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== 경로 설정: 분석 파이프라인과 동일 OUTDIR 사용 =====
OUTDIR = Path(r"C:\Users\User\Desktop\분석결과")  # 파이프라인 OUTDIR와 동일하게
PATTERN = "통합회귀_환경만_MULTI_FIXED_*.xlsx"    # 파이프라인 산출물 패턴

# ===== 시각화 기본 옵션 =====
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ===== 보조: 최신 결과 파일 찾기 =====
def find_latest_result():
    files = sorted(OUTDIR.glob(PATTERN))
    if not files:
        raise FileNotFoundError("분석 산출 엑셀을 찾을 수 없습니다.")
    return files[-1]

# ===== 시트 접두어에서 타깃/품종/월범위 파싱 =====
# 예) "과중_홍로(m04-m08)_02_Summary"에서 과중, 홍로, m04-m08 추출
RE_PREFIX = re.compile(r"^(?P<trait>.+?)(?:_(?P<cultivar>[^()]+))?\((?P<mm>m\d{2}-m\d{2})\)")

def parse_prefix(sheet_name):
    # 시트명은 "..._02_Summary" 형태 -> 앞부분만 자르기
    base = sheet_name.rsplit("_", 1)[0]
    m = RE_PREFIX.match(base)
    if m:
        d = m.groupdict()
        return d.get("trait"), d.get("cultivar") or "", d.get("mm")
    # 폴백: 전체를 trait로
    return base, "", ""

# ===== 단위/표기(선택) =====
TRAIT_UNITS = {
    "과중": "g",
    "종경": "mm",
    "횡경": "mm",
    "당도": "°Brix",
    "산도": "%",
    "경도": "N",
    "L": "", "a": "", "b": ""
}

# ===== 메인: 엑셀에서 요약/예측/계수 수집 =====
def collect_from_excel(xlsx_path: Path):
    xls = pd.ExcelFile(xlsx_path)
    rows = []
    preds_list = []   # 각 타깃별 산점도 그리기용
    coefs_list = []   # 각 타깃별 계수 바차트용

    for sh in xls.sheet_names:
        if sh.endswith("_02_Summary"):
            df = pd.read_excel(xls, sh)
            trait, cultivar, mm = parse_prefix(sh)
            # 요약 표에서 value 추출
            kv = dict(zip(df["metric"], df["value"]))
            rows.append({
                "trait": trait, "cultivar": cultivar, "month_window": mm,
                "n": float(kv.get("n_samples", np.nan)),
                "R2_in": float(kv.get("r2_in_sample", np.nan)),
                "RMSE_in": float(kv.get("rmse_in_sample", np.nan)),
                "alpha": float(kv.get("lasso_alpha", np.nan))
            })

        elif sh.endswith("_04_Preds"):
            trait, cultivar, mm = parse_prefix(sh)
            dfp = pd.read_excel(xls, sh)
            dfp["trait"] = trait
            dfp["cultivar"] = cultivar
            dfp["month_window"] = mm
            preds_list.append(dfp[["trait","cultivar","month_window","y_true","y_pred"]])

        elif sh.endswith("_03_Coeff"):
            trait, cultivar, mm = parse_prefix(sh)
            cdf = pd.read_excel(xls, sh)
            cdf["trait"] = trait
            cdf["cultivar"] = cultivar
            cdf["month_window"] = mm
            coefs_list.append(cdf)

    perf = pd.DataFrame(rows).sort_values(["cultivar","trait"]).reset_index(drop=True)
    preds = pd.concat(preds_list, ignore_index=True) if preds_list else pd.DataFrame()
    coefs = pd.concat(coefs_list, ignore_index=True) if coefs_list else pd.DataFrame()
    return perf, preds, coefs

# ===== 그림 1: 지표별 R² 막대그래프 =====
def plot_r2_bar(perf, cultivar=None):
    df = perf.copy()
    if cultivar is not None:
        df = df[df["cultivar"] == cultivar]

    # 같은 지표가 여러 월범위/모델로 중복되면 최대값 기준(논문에서 대표치로 흔히 사용)
    dfm = df.groupby("trait", as_index=False)["R2_in"].max()
    order = dfm.sort_values("R2_in", ascending=False)["trait"].tolist()

    plt.figure(figsize=(8, 4))
    plt.bar(dfm["trait"], dfm["R2_in"])
    plt.xticks(rotation=45, ha="right")
    ttl = f"지표별 설명력(R²) 비교" + (f" - {cultivar}" if cultivar else "")
    plt.title(ttl)
    plt.ylabel("R² (in-sample)")
    plt.tight_layout()
    out = OUTDIR / f"Fig_R2_by_trait{'_'+cultivar if cultivar else ''}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out

# ===== 그림 2: 지표별 RMSE 막대그래프 =====
def plot_rmse_bar(perf, cultivar=None):
    df = perf.copy()
    if cultivar is not None:
        df = df[df["cultivar"] == cultivar]
    dfm = df.groupby("trait", as_index=False)["RMSE_in"].median()

    plt.figure(figsize=(8, 4))
    plt.bar(dfm["trait"], dfm["RMSE_in"])
    plt.xticks(rotation=45, ha="right")
    ttl = f"지표별 RMSE 비교" + (f" - {cultivar}" if cultivar else "")
    plt.title(ttl)
    plt.ylabel("RMSE (단위는 지표별 상이)")
    plt.tight_layout()
    out = OUTDIR / f"Fig_RMSE_by_trait{'_'+cultivar if cultivar else ''}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out

# ===== 그림 3: 관측 vs 예측 산점도(지표별 패널) =====
def plot_obs_pred_facets(preds, cultivar=None, max_panels=9):
    df = preds.copy()
    if df.empty:
        return None
    if cultivar is not None:
        df = df[df["cultivar"] == cultivar]

    traits = df["trait"].dropna().unique().tolist()[:max_panels]
    n = len(traits)
    if n == 0:
        return None

    cols = min(3, n)
    rows = int(np.ceil(n/cols))
    plt.figure(figsize=(4*cols, 3.5*rows))

    for i, t in enumerate(traits, 1):
        sub = df[df["trait"] == t]
        ax = plt.subplot(rows, cols, i)
        ax.scatter(sub["y_pred"], sub["y_true"], s=8, alpha=0.6)
        lims = [
            np.nanmin([sub["y_pred"].min(), sub["y_true"].min()]),
            np.nanmax([sub["y_pred"].max(), sub["y_true"].max()])
        ]
        ax.plot(lims, lims, "--", lw=1)
        ax.set_title(t)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")

    ttl = "Observed vs Predicted" + (f" - {cultivar}" if cultivar else "")
    plt.suptitle(ttl, y=1.02)
    plt.tight_layout()
    out = OUTDIR / f"Fig_Obs_vs_Pred{'_'+cultivar if cultivar else ''}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return out

# ===== 그림 4: 특정 지표의 계수 바차트 =====
def plot_coefficients_bar(coefs, trait, cultivar=None, top_k=12):
    df = coefs.copy()
    if df.empty:
        return None
    df = df[df["feature"] != "const"]
    df = df[df["trait"] == trait]
    if cultivar is not None:
        df = df[df["cultivar"] == cultivar]
    if df.empty:
        return None
    # 유의미한 계수 우선 정렬: |coef| 큰 상위
    df = df.sort_values("coef", key=lambda s: s.abs(), ascending=False).head(top_k)

    plt.figure(figsize=(6, max(3, 0.35*len(df))))
    plt.barh(df["feature"], df["coef"])
    plt.gca().invert_yaxis()
    ttl = f"{trait} 회귀계수" + (f" - {cultivar}" if cultivar else "")
    plt.title(ttl)
    plt.xlabel("Coefficient")
    plt.tight_layout()
    out = OUTDIR / f"Fig_Coeff_{trait}{'_'+cultivar if cultivar else ''}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    return out

# ===== 표 1: 성능 요약표 CSV (논문 Table용 초안) =====
def export_performance_table(perf, cultivar=None):
    df = perf.copy()
    if cultivar is not None:
        df = df[df["cultivar"] == cultivar]
    # 지표별 대표치(최대 R², 중앙 RMSE) 선택
    df_r2 = df.sort_values("R2_in", ascending=False).groupby("trait", as_index=False).first()
    df_rmse = df.groupby("trait", as_index=False)["RMSE_in"].median().rename(columns={"RMSE_in":"RMSE_med"})
    tab = pd.merge(df_r2[["trait","R2_in","n"]], df_rmse, on="trait", how="left")
    # 단위 열 추가
    tab["unit"] = tab["trait"].map(TRAIT_UNITS).fillna("")
    out = OUTDIR / f"Table_Performance{'_'+cultivar if cultivar else ''}.csv"
    tab.to_csv(out, index=False, encoding="utf-8-sig")
    return out

def main():
    latest = find_latest_result()
    print("읽는 파일:", latest)
    perf, preds, coefs = collect_from_excel(latest)

    # 품종별로 따로/전체 모두 생성
    cultivars = sorted([c for c in perf["cultivar"].unique().tolist() if c])
    cultivars = cultivars or [None]  # 품종 정보가 없으면 전체만

    for cv in ([None] + cultivars) if cultivars != [None] else [None]:
        t1 = export_performance_table(perf, cv)
        f1 = plot_r2_bar(perf, cv)
        f2 = plot_rmse_bar(perf, cv)
        f3 = plot_obs_pred_facets(preds, cv)
        # 대표 지표 예: 과중/종경/횡경의 계수 바차트
        for trait in ["과중","종경","횡경"]:
            plot_coefficients_bar(coefs, trait, cv)

        print("완료:",
              "\n -", t1,
              "\n -", f1,
              "\n -", f2,
              "\n -", f3 or "산점도 없음(예측 시트 없음)")

if __name__ == "__main__":
    main()

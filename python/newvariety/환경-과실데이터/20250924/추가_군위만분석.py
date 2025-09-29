# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== 사용자 경로 (분석 파이프라인 OUTDIR와 동일하게 맞추세요) =====
OUTDIR = Path(r"C:\Users\User\Desktop\mba\분석결과")
PATTERN = "통합회귀_환경만_MULTI_FIXED_군위.xlsx"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ===== 유틸 =====
RE_PREFIX = re.compile(r"^(?P<trait>.+?)(?:_(?P<cultivar>[^()]+))?\((?P<mm>m\d{2}-m\d{2})\)")

def find_latest_xlsx():
    files = sorted(OUTDIR.glob(PATTERN))
    if not files:
        raise FileNotFoundError("분석 산출 엑셀을 찾을 수 없습니다.")
    return files[-1]

def parse_prefix(sheet_name):
    base = sheet_name.rsplit("_", 1)[0]
    m = RE_PREFIX.match(base)
    if m:
        d = m.groupdict()
        return d.get("trait"), (d.get("cultivar") or ""), d.get("mm")
    return base, "", ""

# 지표 단위(표/축 라벨용, 필요시 수정)
TRAIT_UNITS = {
    "과중": "g", "종경": "mm", "횡경": "mm",
    "당도": "°Brix", "산도": "%", "경도": "N",
    "L": "", "a": "", "b": ""
}

# ===== 엑셀에서 필요한 시트만 수집 =====
def collect_core(xlsx_path: Path):
    xls = pd.ExcelFile(xlsx_path)
    perf_rows, preds_rows, coef_rows = [], [], []
    for sh in xls.sheet_names:
        if sh.endswith("_02_Summary"):
            df = pd.read_excel(xls, sh)
            trait, cultivar, mm = parse_prefix(sh)
            kv = dict(zip(df["metric"], df["value"]))
            perf_rows.append({
                "trait": trait, "cultivar": cultivar, "mm": mm,
                "n": float(kv.get("n_samples", np.nan)),
                "R2": float(kv.get("r2_in_sample", np.nan)),
                "RMSE": float(kv.get("rmse_in_sample", np.nan))
            })
        elif sh.endswith("_04_Preds"):
            t, c, mm = parse_prefix(sh)
            dfp = pd.read_excel(xls, sh)
            dfp["trait"], dfp["cultivar"], dfp["mm"] = t, c, mm
            preds_rows.append(dfp[["trait","cultivar","mm","y_true","y_pred"]])
        elif sh.endswith("_03_Coeff"):
            t, c, mm = parse_prefix(sh)
            cdf = pd.read_excel(xls, sh)
            cdf["trait"], cdf["cultivar"], cdf["mm"] = t, c, mm
            coef_rows.append(cdf)

    perf = pd.DataFrame(perf_rows)
    preds = pd.concat(preds_rows, ignore_index=True) if preds_rows else pd.DataFrame()
    coefs = pd.concat(coef_rows, ignore_index=True) if coef_rows else pd.DataFrame()
    return perf, preds, coefs

# ===== 그림 1: 성능 요약 (R² & RMSE) =====
def plot_perf(perf, cultivar=None):
    df = perf.copy()
    title_tag = "전체" if cultivar in (None, "") else cultivar
    if cultivar not in (None, ""):
        df = df[df["cultivar"] == cultivar]
    # 지표별 대표치: R² 최대, RMSE 중앙값
    r2_rep = df.sort_values("R2", ascending=False).groupby("trait", as_index=False).first()[["trait","R2","n"]]
    rmse_rep = df.groupby("trait", as_index=False)["RMSE"].median().rename(columns={"RMSE":"RMSE_med"})
    tab = pd.merge(r2_rep, rmse_rep, on="trait", how="left").fillna(np.nan)

    # 저장: 표 CSV
    out_csv = OUTDIR / f"PPT_Table_Performance_{title_tag}.csv"
    tab.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 막대그래프 2장 (R², RMSE)
    order = tab.sort_values("R2", ascending=False)["trait"].tolist()
    # R²
    plt.figure(figsize=(8,4))
    plt.bar(tab["trait"], tab["R2"])
    plt.xticks(rotation=45, ha="right")
    plt.title(f"지표별 설명력(R²) - {title_tag}")
    plt.ylabel("R² (in-sample)")
    plt.tight_layout()
    out_r2 = OUTDIR / f"PPT_Fig_R2_{title_tag}.png"
    plt.savefig(out_r2, dpi=300); plt.close()

    # RMSE
    plt.figure(figsize=(8,4))
    plt.bar(tab["trait"], tab["RMSE_med"])
    plt.xticks(rotation=45, ha="right")
    plt.title(f"지표별 오차(RMSE, 대표치) - {title_tag}")
    plt.ylabel("RMSE")
    plt.tight_layout()
    out_rmse = OUTDIR / f"PPT_Fig_RMSE_{title_tag}.png"
    plt.savefig(out_rmse, dpi=300); plt.close()

    return out_csv, out_r2, out_rmse, tab

# ===== 그림 2: 대표 산점도 2장 (R² 상위 2개 지표) =====
def plot_top2_scatter(preds, perf, cultivar=None, min_r2=0.0):
    dfp = preds.copy()
    title_tag = "전체" if cultivar in (None, "") else cultivar
    if cultivar not in (None, ""):
        dfp = dfp[dfp["cultivar"] == cultivar]
    # 상위 2개 지표 선택
    p2 = perf.copy()
    if cultivar not in (None, ""):
        p2 = p2[p2["cultivar"] == cultivar]
    top = p2[p2["R2"] >= min_r2].sort_values("R2", ascending=False)["trait"].unique().tolist()
    if len(top) < 2:
        top = p2.sort_values("R2", ascending=False)["trait"].unique().tolist()
    top = top[:2]

    outs = []
    for t in top:
        sub = dfp[dfp["trait"] == t].dropna(subset=["y_true","y_pred"])
        if sub.empty: 
            continue
        plt.figure(figsize=(4.5,4.2))
        plt.scatter(sub["y_pred"], sub["y_true"], s=10, alpha=0.6)
        lo = np.nanmin([sub["y_pred"].min(), sub["y_true"].min()])
        hi = np.nanmax([sub["y_pred"].max(), sub["y_true"].max()])
        plt.plot([lo,hi],[lo,hi],"--", lw=1)
        u = TRAIT_UNITS.get(t, "")
        plt.title(f"Observed vs Predicted: {t} ({title_tag})")
        plt.xlabel(f"Predicted {f'({u})' if u else ''}")
        plt.ylabel(f"Observed {f'({u})' if u else ''}")
        plt.tight_layout()
        out = OUTDIR / f"PPT_Fig_ObsPred_{t}_{title_tag}.png"
        plt.savefig(out, dpi=300); plt.close()
        outs.append(out)
    return outs

# ===== 그림 3: 대표 계수 바차트 1장 (R² 최상위 지표) =====
def plot_top_coeff(coefs, perf, cultivar=None, top_k=12):
    dfc = coefs.copy()
    title_tag = "전체" if cultivar in (None, "") else cultivar
    if dfc.empty:
        return None
    dfc = dfc[dfc["feature"] != "const"]
    if cultivar not in (None, ""):
        dfc = dfc[dfc["cultivar"] == cultivar]

    # R² 최상위 지표 1개
    p2 = perf.copy()
    if cultivar not in (None, ""):
        p2 = p2[p2["cultivar"] == cultivar]
    if p2.empty: 
        return None
    best_trait = p2.sort_values("R2", ascending=False)["trait"].iloc[0]

    sub = dfc[dfc["trait"] == best_trait]
    if sub.empty: 
        return None
    sub = sub.sort_values("coef", key=lambda s: s.abs(), ascending=False).head(top_k)

    plt.figure(figsize=(6, max(3, 0.35*len(sub))))
    plt.barh(sub["feature"], sub["coef"])
    plt.gca().invert_yaxis()
    plt.title(f"{best_trait} 회귀계수 - {title_tag}")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    out = OUTDIR / f"PPT_Fig_Coeff_{best_trait}_{title_tag}.png"
    plt.savefig(out, dpi=300); plt.close()
    return out

def main():
    xlsx = find_latest_xlsx()
    print("읽는 파일:", xlsx)
    perf, preds, coefs = collect_core(xlsx)

    # 품종 목록(없으면 전체만)
    cultivars = sorted([c for c in perf["cultivar"].unique().tolist() if c]) or [None]

    # 전체(품종 무시) + 품종별 생성
    for cv in [None] + ([c for c in cultivars if c is not None]):
        csv_path, r2_fig, rmse_fig, tab = plot_perf(perf, cv)
        print("성능요약:", csv_path, r2_fig, rmse_fig)
        scatters = plot_top2_scatter(preds, perf, cv, min_r2=0.0)
        print("대표 산점도:", [str(p) for p in scatters] or "없음")
        coeff = plot_top_coeff(coefs, perf, cv, top_k=12)
        print("대표 계수:", coeff or "없음")

if __name__ == "__main__":
    main()

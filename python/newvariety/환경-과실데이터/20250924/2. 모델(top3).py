# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer

# -----------------------------
# 0. 환경설정
# -----------------------------
mpl.rcParams['font.family'] = 'Malgun Gothic'   # 한글 폰트
mpl.rcParams['axes.unicode_minus'] = False

BASE = Path(r"C:/Users/User/Desktop/환경데이터")

BIO_FILE = BASE / "bioclim_19_variables.csv"
WEATHER_FILE = BASE / "기상데이터_통합.xlsx"
FRUIT_FILES = {
    "홍로": BASE / "홍로.xlsx",
    "후지": BASE / "후지.xlsx"
}

# -----------------------------
# 1. 데이터 불러오기
# -----------------------------
bio = pd.read_csv(BIO_FILE)
weather = pd.read_excel(WEATHER_FILE)

# 기상 월 집계
weather["date"] = pd.to_datetime(weather["일자"], errors="coerce")
weather = weather.dropna(subset=["date"]).copy()
weather["year"] = weather["date"].dt.year
weather["month"] = weather["date"].dt.month

monthly = (weather.groupby(["지역명","year","month"], as_index=False)
           .agg({
               "평균기온":"mean",
               "최고기온":"mean",
               "최저기온":"mean",
               "강우량":"sum",
               "일사량":"mean"
           })
           .rename(columns={"지역명":"region","평균기온":"tmean",
                            "최고기온":"tmax","최저기온":"tmin",
                            "강우량":"prcp","일사량":"rad"}))

ranges = {"홍로": range(5,9), "후지": range(5,11)}

def summarize_months(grp, months):
    g = grp[grp["month"].isin(months)]
    return pd.Series({
        "tmean_mean": g["tmean"].mean(),
        "tmax_max": g["tmax"].max(),
        "tmin_min": g["tmin"].min(),
        "prcp_sum": g["prcp"].sum(),
        "rad_mean": g["rad"].mean()
    })

# -----------------------------
# 2. 유틸 함수
# -----------------------------
def render_linear_formula(coef, intercept, feature_names, target, thr=1e-8, topn=None):
    pairs = [(fn, float(c)) for fn, c in zip(feature_names, coef) if abs(c) > thr]
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    if topn is not None:
        pairs = pairs[:topn]
    terms = [f"{round(v,6)}*{k}" for k, v in pairs]
    return f"{target} = " + (" + ".join(terms) + " + " if terms else "") + f"{round(float(intercept),6)}"

def save_pred_plot(y_true, y_pred, title, path_png):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.7)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.close()

# -----------------------------
# 3. 품종별 처리
# -----------------------------
targets = ["과중","종경","횡경","L","a","b","경도","당도","산도"]

for cultivar, fruit_file in FRUIT_FILES.items():
    print(f"[INFO] {cultivar} 처리 중...")

    OUT = BASE / "outputs" / cultivar
    (OUT / "figs" / "pred_vs_actual").mkdir(parents=True, exist_ok=True)

    # 과실 데이터
    fruit = pd.read_excel(fruit_file)
    fruit_clean = fruit.copy()
    fruit_clean["region"] = fruit_clean["지역명"].astype(str).str.strip()
    fruit_clean["year"] = fruit_clean["일자"].astype(int)
    fruit_clean["cultivar"] = cultivar

    # 기상 요약
    rows = []
    for (region, year), grp in monthly.groupby(["region","year"]):
        rows.append({"region":region,"year":year,"cultivar":cultivar,
                     **summarize_months(grp, ranges[cultivar])})
    weather_feat = pd.DataFrame(rows)

    # BIO 병합
    bio_feat = bio.assign(cultivar=cultivar)
    feat = weather_feat.merge(bio_feat, on=["region","year","cultivar"], how="left")

    # 최종 병합
    df_merged = fruit_clean.merge(feat, on=["region","year","cultivar"], how="inner")

    # 설명변수/타깃
    id_cols = ["품종","일자","지역명","region","year","cultivar"]
    X_all = df_merged.drop(columns=[c for c in id_cols if c in df_merged.columns], errors="ignore")
    X_cols = [c for c in X_all.columns if c not in targets]
    X_all = X_all[X_cols]

    imputer = SimpleImputer(strategy="median")
    X_all_imputed = pd.DataFrame(imputer.fit_transform(X_all), columns=X_cols)

    # 결과 저장 준비
    results = []
    formulas_best = {}

    for target in targets:
        if target not in df_merged.columns:
            continue
        data = pd.concat([X_all_imputed, df_merged[target]], axis=1).dropna(subset=[target])
        if len(data) < 30:
            results.append({"target": target, "n_samples": len(data), "Best_Model": "N/A"})
            continue

        X = data.drop(columns=[target])
        y = data[target].astype(float)
        feats = X.columns
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Linear
        lr = LinearRegression().fit(X_train, y_train)
        pred_lr = lr.predict(X_test)
        lr_r2 = r2_score(y_test, pred_lr)

        # Lasso
        lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
        pred_lasso = lasso.predict(X_test)
        lasso_r2 = r2_score(y_test, pred_lasso)

        # RF
        rf = RandomForestRegressor(n_estimators=500, random_state=42).fit(X_train, y_train)
        pred_rf = rf.predict(X_test)
        rf_r2 = r2_score(y_test, pred_rf)

        # Best 선택
        perf = [("Linear", lr_r2), ("Lasso", lasso_r2), ("RF", rf_r2)]
        best_model = max(perf, key=lambda x: x[1])

        if best_model[0] == "Linear":
            formulas_best[target] = render_linear_formula(lr.coef_, lr.intercept_, feats, target)
        elif best_model[0] == "Lasso":
            formulas_best[target] = render_linear_formula(lasso.coef_, lasso.intercept_, feats, target)
        else:
            # RF Surrogate (Top3)
            fi = pd.Series(rf.feature_importances_, index=feats)
            top_feats = fi.sort_values(ascending=False).head(3).index
            rf_pred_all = rf.predict(X)  # RF 전체 예측
            sur = LinearRegression().fit(X[top_feats], rf_pred_all)
            sur_formula = render_linear_formula(sur.coef_, sur.intercept_, top_feats, target)
            formulas_best[target] = f"[RF Surrogate (Top3)] {sur_formula}"

        results.append({
            "target": target,
            "n_samples": len(data),
            "Linear_R2": lr_r2,
            "Lasso_R2": lasso_r2,
            "RF_R2": rf_r2,
            "Best_Model": best_model[0]
        })

    # 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_excel(OUT / f"model_summary_{cultivar}.xlsx", index=False)

    with open(OUT / f"best_formulas_{cultivar}.txt", "w", encoding="utf-8") as f:
        for tgt, formula in formulas_best.items():
            f.write(f"{tgt}: {formula}\n")

    print(f"[완료] {cultivar} 결과 저장 -> {OUT}")

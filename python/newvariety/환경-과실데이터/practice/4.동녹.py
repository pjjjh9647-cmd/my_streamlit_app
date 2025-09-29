"""
동녹 예측: OLS+VIF(설명), Ridge(공선성 완화), RandomForest(비선형) 비교 스크립트
- 입력: 환경.xlsx(월별 기온/습도/일사량 등), 과실.xlsx(동녹 포함)
- 병합 키: '지역'
- 산출: 결과_동녹예측 폴더에 CSV/PNG/TXT 저장
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")  # 과도한 경고 숨김(필요시 해제)

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

# ----------------------------
# 0) 사용자 환경 설정
# ----------------------------
ENV_PATH   = r"C:\Users\User\Desktop\일\주산지현장연구\환경.xlsx"
FRUIT_PATH = r"C:\Users\User\Desktop\일\주산지현장연구\과실.xlsx"
OUT_DIR    = r"C:\Users\User\Desktop\일\주산지현장연구\결과_동녹예측"

os.makedirs(OUT_DIR, exist_ok=True)

# 한글 폰트 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 재현성
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ----------------------------
# 1) 데이터 로드 및 병합
# ----------------------------
df_env   = pd.read_excel(ENV_PATH)
df_fruit = pd.read_excel(FRUIT_PATH)
df = pd.merge(df_env, df_fruit, on="지역", how="inner")

# 환경 변수 선택: 월별 기온/습도/일사량 전부
# (원하시면 startswith(("강수량_", "풍속_"))도 추가 가능)
env_cols = [c for c in df.columns if c.startswith(("기온_", "습도_", "일사량_"))]
target   = "동녹"   # 종속변수(열 이름은 데이터에 맞게 유지)

# 숫자형 변환 및 결측 처리
X_all = df[env_cols].apply(pd.to_numeric, errors="coerce")
y_all = pd.to_numeric(df[target], errors="coerce")
mask  = (~X_all.isna().any(axis=1)) & (~y_all.isna())
X = X_all.loc[mask].copy()
y = y_all.loc[mask].copy()

# 분산이 0(상수)인 열 제거(OLS/VIF 안정화용)
zero_var_cols = X.columns[X.std(axis=0) < 1e-12].tolist()
if zero_var_cols:
    X.drop(columns=zero_var_cols, inplace=True)

print(f"[정보] 사용 표본 수: {len(y)} / 사용 변수 수: {X.shape[1]}")
X.to_csv(os.path.join(OUT_DIR, "X_환경변수_정리.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame({"동녹": y.values}).to_csv(os.path.join(OUT_DIR, "y_동녹.csv"),
                                    index=False, encoding="utf-8-sig")


# ----------------------------
# 2) 진단: 동녹과의 상관(절대값 순)
# ----------------------------
corr_to_y = X.join(y).corr()[target].drop(target).sort_values(key=np.abs, ascending=False)
plt.figure(figsize=(6, max(4, len(corr_to_y)*0.25)))
sns.barplot(x=corr_to_y.values, y=corr_to_y.index)
plt.title("동녹과 환경 변수 상관계수(절대값 순)")
plt.xlabel("상관계수")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "상관계수_막대그래프.png"), dpi=150)
plt.close()
corr_to_y.to_csv(os.path.join(OUT_DIR, "상관계수_동녹대비.csv"), encoding="utf-8-sig")


# --- A안: VIF 반복 제거 + OLS 적합 ---
def drop_high_vif(X_df, thresh=10.0, max_iter=100):
    Xw = X_df.copy()
    for _ in range(max_iter):
        if Xw.shape[1] <= 2:  # 변수 2개 이하이면 중단(모형 붕괴 방지)
            break
        Xc = sm.add_constant(Xw)
        vifs = pd.Series([variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])],
                         index=Xc.columns)
        vifs = vifs.drop("const", errors="ignore")
        if vifs.empty or np.isnan(vifs.values).any():
            break
        max_var, max_v = vifs.idxmax(), vifs.max()
        if max_v <= thresh:
            break
        Xw = Xw.drop(columns=[max_var])
    return Xw, vifs.sort_values(ascending=False) if 'vifs' in locals() and not vifs.empty else (Xw, pd.Series(dtype=float))

X_vif, vif_series = drop_high_vif(X, thresh=10.0)
# VIF 결과 확인(원하면 CSV로 저장)
print("[VIF 최종]:\n", vif_series.head(10))

# 변수 너무 적으면(혹은 비어지면) 상관 상위 2개로 보정
if X_vif.shape[1] < 2:
    top2 = corr_to_y.index[:2].tolist()
    X_vif = X[top2].copy()

X_ols = sm.add_constant(X_vif)
ols_model = sm.OLS(y, X_ols).fit()
print("\n[A안 OLS+VIF] 남은 변수:", list(X_vif.columns))
print(ols_model.summary())

# 예측 vs 실제 / 잔차 플롯 저장
y_pred_ols = ols_model.predict(X_ols)
plt.figure(figsize=(5,5))
sns.scatterplot(x=y, y=y_pred_ols)
mn, mx = min(y.min(), y_pred_ols.min()), max(y.max(), y_pred_ols.max())
plt.plot([mn, mx], [mn, mx], "r--"); plt.xlabel("실제 동녹"); plt.ylabel("예측 동녹")
plt.title("A안 OLS: 예측 vs 실제"); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "A_OLS_예측vs실제.png"), dpi=150); plt.close()

resid = y - y_pred_ols
plt.figure(figsize=(5,4))
sns.residplot(x=y_pred_ols, y=resid, lowess=True, line_kws={'color':'red'})
plt.xlabel("예측값"); plt.ylabel("잔차"); plt.title("A안 OLS: 잔차 플롯"); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "A_OLS_잔차플롯.png"), dpi=150); plt.close()

with open(os.path.join(OUT_DIR, "A_OLS_VIF_요약.txt"), "w", encoding="utf-8") as f:
    f.write(str(ols_model.summary()))
vif_series.to_csv(os.path.join(OUT_DIR, "A_VIF_최종.csv"), encoding="utf-8-sig")

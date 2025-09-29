# -*- coding: utf-8 -*-
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text

from pipeline_core import (
    COL_DATE, COL_REGION, CANDIDATE_COLS, KEY_COLS,
    run_single_target, load_latest_model, save_model_registry, _norm_cultivar_name
)

# =========================
# 설정
# =========================
USE_DB = True  # DB에서 불러오면 True, 엑셀에서 불러오면 False

# DB 예시 (수정하세요)
DB_URL = "mssql+pyodbc://USER:PASS@SERVER/DBNAME?driver=ODBC+Driver+17+for+SQL+Server"

SQL_ENV = """
SELECT 지역명, 일자, 평균기온, 최고기온, 최저기온, 습도, 강우량, 일사량, 결로시간, 평균풍속, 최대풍속
FROM environment_daily
WHERE 일자 IS NOT NULL
"""
SQL_FRUIT = """
SELECT 지역명, 품종, 일자, 계측시간, 과중, 경도, 산도, 당도, 착색비율
FROM fruit_targets
"""

# 엑셀 경로 (엑셀 모드일 때)
ENV_XLSX   = r"C:\Users\User\Desktop\환경데이터\기상데이터_통합.xlsx"
FRUIT_XLSX = r"C:\Users\User\Desktop\과실데이터\과실데이터_통합.xlsx"

# 모델 레지스트리 위치
REGISTRY_ROOT = Path(r"C:\Users\User\Desktop\orchard_models_registry")
# 품종별·타깃별로 폴더를 나누어 저장합니다: REGISTRY_ROOT/<cultivar or ALL>/<target>

# 교체 기준(검증은 간단히 in-sample 기준, 필요시 시계열 CV로 확장)
MIN_R2  = 0.40
MAX_RMSE_INCREASE_RATIO = 1.00  # 기존 RMSE보다 나빠지면 교체 안 함

TARGET_COLUMNS = ["과중","경도","산도","당도","착색비율"]  # 환경과 병합 가능한 숫자형 타깃

# =========================
# 데이터 로딩
# =========================
def load_data():
    if USE_DB:
        eng = create_engine(DB_URL)
        env = pd.read_sql(text(SQL_ENV), eng)
        fruit = pd.read_sql(text(SQL_FRUIT), eng)
    else:
        env = pd.read_excel(ENV_XLSX)
        fruit = pd.read_excel(FRUIT_XLSX)

    # year, month 생성
    env = env.copy()
    env[COL_DATE] = pd.to_datetime(env[COL_DATE], errors="coerce")
    env["year"] = env[COL_DATE].dt.year
    env["month"] = env[COL_DATE].dt.month

    # 과실 year 생성
    if "year" not in fruit.columns:
        if "일자" in fruit.columns:
            fruit["year"] = pd.to_datetime(fruit["일자"], errors="coerce").dt.year
        elif "계측시간" in fruit.columns:
            fruit["year"] = pd.to_datetime(fruit["계측시간"], errors="coerce").dt.year

    return env, fruit

def available_env_cols(env):
    return [c for c in CANDIDATE_COLS if c in env.columns]

# =========================
# 학습 & 등록
# =========================
def train_and_maybe_register():
    env, fruit = load_data()
    env_cols = available_env_cols(env)

    if "품종" in fruit.columns:
        cultivars = fruit["품종"].dropna().astype(str).map(_norm_cultivar_name).unique().tolist()
    else:
        cultivars = [None]

    report_rows = []
    for cv in cultivars:
        fsub = fruit if cv is None else fruit[fruit["품종"].astype(str).map(_norm_cultivar_name) == cv]
        cv_key = cv if cv is not None else "ALL"

        for tgt in [c for c in TARGET_COLUMNS if c in fsub.columns]:
            try:
                model, summary, corr_s, merged = run_single_target(env, fsub, env_cols, tgt, cultivar_name=cv)

                # 레지스트리 위치
                model_dir = REGISTRY_ROOT / cv_key / tgt
                old_model, old_meta = load_latest_model(model_dir)

                accept = False
                reason = "first_model"
                if old_model is None:
                    # 최초 등록
                    accept = summary["r2_in_sample"] >= MIN_R2
                    reason = "first_model_threshold"
                else:
                    # 간단한 교체 규칙
                    old_r2   = float(old_meta.get("r2_in_sample", -1))
                    old_rmse = float(old_meta.get("rmse_in_sample", 1e9))
                    new_r2   = summary["r2_in_sample"]
                    new_rmse = summary["rmse_in_sample"]

                    better_r2 = new_r2 >= max(MIN_R2, old_r2)
                    not_worse_rmse = (new_rmse <= old_rmse * (1.0 + MAX_RMSE_INCREASE_RATIO))
                    accept = better_r2 and not_worse_rmse
                    reason = f"compare(old_r2={old_r2}, new_r2={new_r2}, old_rmse={old_rmse}, new_rmse={new_rmse})"

                if accept:
                    meta = {
                        **summary,
                        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "rows_used": int(len(merged)),
                        "env_cols": env_cols,
                        "target": tgt,
                        "cultivar": cv_key,
                    }
                    ver_dir = save_model_registry(model_dir, model, meta)
                    status = f"REGISTERED -> {ver_dir.name}"
                else:
                    status = "SKIPPED(under_threshold_or_worse)"

                report_rows.append({
                    "cultivar": cv_key, "target": tgt,
                    "r2": summary["r2_in_sample"], "rmse": summary["rmse_in_sample"],
                    "n": summary["n_samples"], "status": status, "reason": reason
                })
            except Exception as e:
                report_rows.append({
                    "cultivar": cv_key, "target": tgt,
                    "r2": None, "rmse": None, "n": None,
                    "status": f"ERROR: {e}", "reason": "exception"
                })

    report = pd.DataFrame(report_rows)
    return report

if __name__ == "__main__":
    rep = train_and_maybe_register()
    out = REGISTRY_ROOT / f"train_report_{datetime.now():%Y%m%d_%H%M%S}.csv"
    rep.to_csv(out, index=False, encoding="utf-8-sig")
    print("훈련 리포트 저장:", out)
    print(rep)

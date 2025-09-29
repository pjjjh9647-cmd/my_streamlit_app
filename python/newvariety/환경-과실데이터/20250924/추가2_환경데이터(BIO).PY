# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re

FILE = Path(r"C:\Users\User\Desktop\mba\환경데이터\기상데이터_통합.xlsx")
OUT  = FILE.parent / "_OUT"; OUT.mkdir(exist_ok=True)

# ===== 설정: 군위 데이터만 사용 =====
# 예: ['군위']이면 '군위', '대구군위', '군위군' 등 '군위'가 포함된 지역명을 모두 포함
TARGET_REGION_KEYWORDS = ['군위']     # 필요시 ['군위','대구군위']처럼 확장

def _compile_region_pattern(keywords):
    # 공백 제거 후 키워드들을 OR로 묶은 정규식
    ks = [re.escape(k.strip()) for k in keywords if str(k).strip()]
    if not ks:
        return None
    return re.compile('|'.join(ks), flags=re.IGNORECASE)

REGION_PATTERN = _compile_region_pattern(TARGET_REGION_KEYWORDS)

# ===== 1) 읽기: 시트마다 단일 헤더(지역명, 일자, 평균기온, 최고기온, 최저기온, 습도, 강우량, 일사량 …) =====
def read_book(xlsx: Path):
    book = pd.read_excel(xlsx, sheet_name=None, header=0)  # 단일 헤더 전제
    frames = []
    for sname, df in book.items():
        if df.empty:
            print(f"[SKIP] 빈 시트: {sname}")
            continue
        df.columns = [str(c).strip() for c in df.columns]
        # 필수 컬럼 존재 확인
        need = ["지역명","일자","최고기온","최저기온","강우량"]
        if not set(need).issubset(df.columns):
            print(f"[WARN] 시트 '{sname}' 필수 컬럼 누락 -> {set(need)-set(df.columns)}; 스킵")
            continue

        g = pd.DataFrame()
        g["region"] = df["지역명"].astype(str).str.strip()
        g["date"]   = pd.to_datetime(df["일자"], errors="coerce")
        g = g.dropna(subset=["date"]).copy()
        g["year"]  = g["date"].dt.year
        g["month"] = g["date"].dt.month

        # 기온
        tmax = pd.to_numeric(df["최고기온"], errors="coerce")
        tmin = pd.to_numeric(df["최저기온"], errors="coerce")
        g["tmax"]  = tmax
        g["tmin"]  = tmin
        if "평균기온" in df.columns:
            g["tmean"] = pd.to_numeric(df["평균기온"], errors="coerce")
        else:
            g["tmean"] = (tmax + tmin) / 2.0

        # 강우량
        g["prcp"] = pd.to_numeric(df["강우량"], errors="coerce")

        # 선택: 일사량/습도
        if "일사량" in df.columns:
            g["rad"] = pd.to_numeric(df["일사량"], errors="coerce")
        if "습도" in df.columns:
            g["humid"] = pd.to_numeric(df["습도"], errors="coerce")

        print(f"[OK] 시트 '{sname}': {len(g)} rows")
        frames.append(g)

    if not frames:
        print("[FAIL] 어떤 시트도 기준 컬럼을 찾지 못했습니다."); sys.exit(0)
    return pd.concat(frames, ignore_index=True).sort_values(["region","date"])

daily_all = read_book(FILE)

# ===== 지역 필터: 군위만 사용 =====
if REGION_PATTERN is None:
    print("[WARN] 지역 키워드가 비어 있어 필터링 없이 전체 사용합니다.")
    daily = daily_all.copy()
else:
    mask = daily_all["region"].str.contains(REGION_PATTERN, na=False)
    daily = daily_all[mask].copy()
    kept_regions = sorted(daily["region"].dropna().unique().tolist())
    print(f"[FILTER] 지역 키워드={TARGET_REGION_KEYWORDS} → 유지 지역 {len(kept_regions)}개: {kept_regions}")
    print(f"[FILTER] 행 수: {len(daily)} / 전체 {len(daily_all)}")

if daily.empty:
    print("[FAIL] 군위 지역으로 필터링한 결과가 비었습니다. 지역명 표기를 다시 확인하세요.")
    sys.exit(0)

# 저장(미리보기)
daily.to_parquet(OUT/"daily_merged_Gunwi.parquet", index=False)
daily.head(30).to_csv(OUT/"daily_head30_Gunwi.csv", index=False, encoding="utf-8-sig")

# ===== 2) 월 집계 =====
agg = {"tmean":"mean","tmax":"mean","tmin":"mean","prcp":"sum"}
for opt in ["rad","humid"]:
    if opt in daily.columns: agg[opt] = "mean"

monthly = (daily.groupby(["region","year","month"], as_index=False).agg(agg))
monthly.to_parquet(OUT/"monthly_agg_Gunwi.parquet", index=False)
monthly.head(30).to_csv(OUT/"monthly_head30_Gunwi.csv", index=False, encoding="utf-8-sig")

# ===== 3) BIOCLIM(온도·강수만 사용) =====
def bioclim(df_m: pd.DataFrame) -> pd.DataFrame:
    rows, eps = [], 1e-9
    for (region, year), g in df_m.groupby(["region","year"]):
        g = g.set_index("month").reindex(range(1,13))
        tmean, tmaxm, tminm, prcp = g["tmean"], g["tmax"], g["tmin"], g["prcp"]
        BIO1 = tmean.mean()
        BIO2 = (tmaxm - tminm).mean()
        BIO5 = tmaxm.max()
        BIO6 = tminm.min()
        BIO7 = BIO5 - BIO6
        BIO3 = (BIO2/(BIO7+eps))*100.0
        BIO4 = tmean.std(ddof=0)*100.0

        def r3_mean(x):
            x2 = pd.concat([x, x.iloc[:2]])
            return pd.Series([x2.iloc[i:i+3].mean() for i in range(12)], index=range(1,13))
        def r3_sum(x):
            x2 = pd.concat([x, x.iloc[:2]])
            return pd.Series([x2.iloc[i:i+3].sum() for i in range(12)], index=range(1,13))

        tmean_q, prcp_q = r3_mean(tmean), r3_sum(prcp)
        wq, cq = int(tmean_q.idxmax()), int(tmean_q.idxmin())
        wetq, driq = int(prcp_q.idxmax()), int(prcp_q.idxmin())

        rows.append(dict(
            region=region, year=year,
            BIO1=BIO1, BIO2=BIO2, BIO3= (BIO2/(BIO7+eps))*100.0, BIO4=BIO4,
            BIO5=BIO5, BIO6=BIO6, BIO7=BIO7,
            BIO8=tmean_q.loc[wetq], BIO9=tmean_q.loc[driq],
            BIO10=tmean_q.loc[wq],  BIO11=tmean_q.loc[cq],
            BIO12=prcp.sum(), BIO13=prcp.max(), BIO14=prcp.min(),
            BIO15=(prcp.std(ddof=0)/(prcp.mean()+eps))*100.0,
            BIO16=prcp_q.loc[wetq], BIO17=prcp_q.loc[driq],
            BIO18=prcp_q.loc[wq],  BIO19=prcp_q.loc[cq]
        ))
    return pd.DataFrame(rows)

bio = bioclim(monthly)
bio.to_csv(OUT/"bioclim_19_variables_Gunwi.csv", index=False, encoding="utf-8-sig")

# ===== 4) 품종별 월구간 피처(홍로 5–8, 후지 5–10) =====
def wide_month(df_sub: pd.DataFrame, cols=("tmean","tmax","tmin","prcp","rad","humid")):
    cols = [c for c in cols if c in df_sub.columns]
    if not cols: return pd.DataFrame()
    w = df_sub.pivot_table(index=["region","year"], columns="month", values=cols)
    w.columns = [f"{v}_m{m:02d}" for v,m in w.columns.to_flat_index()]
    return w.reset_index()

def make_feat(df_m, months, cultivar):
    g = df_m[df_m["month"].isin(months)].copy()
    feats = (g.groupby(["region","year"])
               .agg(tmean_mean=("tmean","mean"),
                    tmean_std =("tmean", lambda x: float(np.nanstd(x, ddof=0))),
                    tmax_max  =("tmax","max"),
                    tmin_min  =("tmin","min"),
                    prcp_sum  =("prcp","sum"),
                    prcp_mean =("prcp","mean"))).reset_index()
    if "rad" in g.columns:   feats["rad_mean"] = g.groupby(["region","year"])["rad"].mean().values
    if "humid" in g.columns: feats["humid_mean"] = g.groupby(["region","year"])["humid"].mean().values
    wide = wide_month(g)
    out = feats.merge(wide, on=["region","year"], how="left")
    out.insert(2, "cultivar", cultivar)
    out.insert(3, "months_used", f"{months[0]}-{months[-1]}")
    return out

hongro = make_feat(monthly, list(range(5,9)),  "Hongro")   # 5~8
fuji   = make_feat(monthly, list(range(5,11)), "Fuji")     # 5~10

hongro.to_csv(OUT/"features_Hongro_5to8_Gunwi.csv", index=False, encoding="utf-8-sig")
fuji.to_csv(OUT/"features_Fuji_5to10_Gunwi.csv",   index=False, encoding="utf-8-sig")

print("완료:", OUT)
print(" - daily_head30_Gunwi.csv / monthly_head30_Gunwi.csv 미리보기 생성")
print(" - bioclim_19_variables_Gunwi.csv, features_Hongro_5to8_Gunwi.csv, features_Fuji_5to10_Gunwi.csv")

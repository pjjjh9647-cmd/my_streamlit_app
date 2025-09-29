# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import requests
import io
import re
import os




# 첫번째 탭: 분석결과 (tab7)
tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs([
    "분석결과", 
    "과실품질 예측", 
    "분석결과(군위)", 
    "과실품질 예측(군위)", 
    "적엽 전후 처리 결과",
    "과실 이미지 분석 결과",  
    "과실품질 예측 / 적합 품종 추천"
])


with tab1:
    st.title("분석 결과")
    BASE_DIR = Path(__file__).parent / "mba" / "분석결과" / "관계시각화2"
    CULTIVAR_DIRS = {
        "홍로": BASE_DIR / "홍로",
        "후지": BASE_DIR / "후지",
    }
    cultivar = st.radio("품종 선택", list(CULTIVAR_DIRS.keys()), horizontal=True, key="radio_tab1")
    folder = CULTIVAR_DIRS[cultivar]
    if not folder.exists():
        st.error("해당 폴더가 없습니다. 경로 오타 또는 드라이브 접근 권한을 확인하세요.")
        st.stop()
    IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
    TAB_EXT = (".csv", ".xlsx")
    all_imgs = sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT])
    all_tabs = sorted([p for p in folder.rglob("*") if p.suffix.lower() in TAB_EXT])
    mode = st.segmented_control("표시 유형", options=["이미지", "표(데이터)"], key="seg_tab1")
    if mode == "이미지":
        if not all_imgs:
            st.warning("표시할 이미지가 없습니다.")
        else:
            view = st.radio("방식 선택", ["갤러리(썸네일)", "단일 파일"], horizontal=True, key="view_tab1")
            if view == "갤러리(썸네일)":
                thumbs = st.slider("한 줄에 몇 장", min_value=2, max_value=6, value=4)
                rows = (len(all_imgs) + thumbs - 1) // thumbs
                st.caption(f"총 {len(all_imgs)}개 이미지")
                idx = 0
                for _ in range(rows):
                    cols = st.columns(thumbs, gap="small")
                    for c in cols:
                        if idx >= len(all_imgs):
                            break
                        p = all_imgs[idx]
                        with c:
                            st.image(str(p), caption=str(p.relative_to(folder)))
                        idx += 1
            else:
                sel = st.selectbox("이미지 선택", [str(p.relative_to(folder)) for p in all_imgs])
                path = folder / sel
                st.image(str(path), caption=str(path))
    else:
        if not all_tabs:
            st.warning("표시할 CSV/XLSX 파일이 없습니다.")
        else:
            path = all_tabs[0]
            st.subheader("예측 정확도")
            try:
                if path.suffix.lower() == ".csv":
                    df = pd.read_csv(path)
                else:
                    df = pd.read_excel(path)
                st.dataframe(df)
            except Exception as e:
                st.error(f"불러오기 실패: {e}")

# 두번째 탭: 과실품질 예측 (3. streamlit(홍로,후지4~)_수정3.py)
with tab2:
   # -*- coding: utf-8 -*-
# 파일명: 기상자동불러오기3.py
    import streamlit as st
    import pandas as pd
    import numpy as np
    import requests
    from datetime import datetime
    import io
    import matplotlib.pyplot as plt
    from typing import Optional
    import re

# ▶ 한글 폰트 자동 설정 (개선판: 경로 안정화 + 시스템 폰트 우선)
from pathlib import Path
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st

plt.rcParams["font.size"] = 6

def _set_korean_font():
    # 1) 앱 폴더 기준으로 fonts/를 정확히 찾도록 절대경로화
    BASE = Path(__file__).parent
    bundled_candidates = [
        BASE / "fonts" / "NanumGothic.ttf",
        BASE / "fonts" / "NotoSansKR-Regular.otf",
    ]

    # 2) 번들 폰트(저장소에 넣은 TTF/OTF)가 있으면 최우선 사용
    for p in bundled_candidates:
        if p.exists():
            try:
                fm.fontManager.addfont(str(p))
                family = fm.FontProperties(fname=str(p)).get_name()
                matplotlib.rcParams["font.family"] = family
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                pass

    # 3) 시스템에 깔려 있을 법한 한글 폰트 후보 (Streamlit Cloud는 보통 Noto 계열 있음)
    preferred = [
        "Noto Sans CJK KR", "Noto Sans KR",       # 리눅스/Cloud에서 기대되는 폰트
        "NanumGothic", "Nanum Gothic",            # 나눔고딕(시스템 설치된 경우)
        "Malgun Gothic",                          # 윈도우
        "AppleGothic",                            # 맥
    ]

    # 현재 시스템에 등록된 폰트 이름 집합
    sys_fonts = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in sys_fonts:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return

    # 4) 그래도 못 찾았으면 최소한 마이너스 깨짐만 방지 + 안내
    matplotlib.rcParams["axes.unicode_minus"] = False
    st.warning("한글 폰트를 찾지 못했습니다. 필요하면 /fonts 에 TTF/OTF(예: NanumGothic.ttf)를 넣어주세요.")

_set_korean_font()

# ---------------------------
# 공통 유틸: 문자열/품질표 정규화
# ---------------------------
def _clean_str(s: str) -> str:
    s = str(s)
    s = s.replace("\xa0", " ")                 # NBSP -> space
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*\(\s*", " (", s)          # "( " -> " ("
    s = re.sub(r"\s*\)\s*", ")", s)           # " )" -> ")"
    return s


def normalize_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    """전년도 과실품질 테이블: 멀티헤더/공백/기호 정리 + 지역/수확일자 정규화"""
    out = df.copy()

    # 멀티헤더 → 단일 문자열
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            " ".join([str(x) for x in tup if str(x) != "nan"]).strip()
            for tup in out.columns.values
        ]

    # 헤더 클린업
    out.columns = [_clean_str(c) for c in out.columns]

    # 흔한 별칭 통일
    alias = {
        "경도 평균(N/ø11mm)": "경도평균(N/ø11mm)",
        "경도 평균 (N/ø11mm)": "경도평균(N/ø11mm)",
        "착색 (Hunter L)": "착색(Hunter L)",
        "착색 (Hunter a)": "착색(Hunter a)",
        "착색 (Hunter b)": "착색(Hunter b)",
    }
    out = out.rename(columns={k: v for k, v in alias.items() if k in out.columns})

    # 지역/수확일자 정리
    if "지역" in out.columns:
        out["지역"] = out["지역"].map(_clean_str)
    if "수확일자" in out.columns:
        out["수확일자"] = pd.to_datetime(out["수확일자"], errors="coerce")

    return out


def get_first_col_by_pattern(df: pd.DataFrame, pattern: str) -> Optional[str]:
    """정규식 패턴으로 컬럼명 1개 찾기(대소문자 무시)"""
    pat = re.compile(pattern, flags=re.IGNORECASE)
    for c in df.columns:
        if pat.search(str(c)):
            return c
    return None


    # -------------------------------------------------
    # 기본 UI
    # -------------------------------------------------

    st.set_page_config(page_title="🍎 사과 과실 품질 예측", layout="wide")
    st.markdown("<h1 style='text-align: center;'>🍎 사과 과실 품질 예측</h1>", unsafe_allow_html=True)

    # -------------------------------------------------
    # 지역 코드
    # -------------------------------------------------
    AREA_CODE = {
        "경기화성": "324",
        "경북영주": "331",
        "경북청송": "332",
        "대구군위": "333",
        "경남거창": "334",
        "전북장수": "335",
        "경기포천": "337",
        "충북충주": "338",
    }

    # 과실품질 페이지의 지역 표기 매핑
    REGION_NAME_MAP = {
        "경기화성": "화성",
        "경북영주": "영주",
        "경북청송": "청송",
        "대구군위": "군위",
        "경남거창": "거창",
        "전북장수": "장수",
        "경기포천": "포천",
        "충북충주": "충주",
    }

    # -------------------------------------------------
    # 회귀식
    # -------------------------------------------------
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

    # -------------------------------------------------
    # 유틸
    # -------------------------------------------------
    def _ensure_numeric(df: pd.DataFrame, cols):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def cultivar_window(cultivar: str):
        if cultivar == "홍로":
            return 4, 8
        return 4, 10  # 후지 기본

    def get_today_ym():
        now = datetime.now()
        return now.year, now.month

    # -------------------------------------------------
    # 사이트 AJAX → JSON (AWS 통계)
    # -------------------------------------------------
    @st.cache_data(show_spinner=False, ttl=300)
    def fetch_aws_stat(region_name: str, stat_gb_code: str, s_ym: str, e_ym: str) -> dict:
        session = requests.Session()
        session.get("https://fruit.nihhs.go.kr/apple/aws/awsStat.do", timeout=20)

        form = {
            "pageIndex": "1",
            "searchNum": "0",
            "areaName": region_name,
            "areaCode": AREA_CODE.get(region_name, region_name),
            "statmethod": stat_gb_code,
        }
        if stat_gb_code == "A":
            form["wetherDtBgn"] = f"{s_ym}-01"
            form["wetherDtEnd"] = f"{e_ym}-30"
        elif stat_gb_code in ("B", "C"):
            form["wetherDtM"] = s_ym
        elif stat_gb_code == "D":
            form["wetherDtBgn2"] = s_ym
            form["wetherDtEnd2"] = e_ym
            form["wetherDtBgn"] = s_ym
            form["wetherDtEnd"] = e_ym

        resp = session.post(
            "https://fruit.nihhs.go.kr/apple/aws/awsStatList.do",
            data=form, timeout=30,
            headers={"Referer": "https://fruit.nihhs.go.kr/apple/aws/awsStat.do"}
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {"error": "JSON 파싱 실패", "raw": resp.text[:4000]}

    # -------------------------------------------------
    # JSON → 표
    # -------------------------------------------------
    def json_to_dataframe(payload: dict) -> pd.DataFrame:
        if not payload or "result" not in payload:
            return pd.DataFrame()
        res = payload.get("result", [])
        if len(res) == 0:
            return pd.DataFrame()

        method = payload.get("mainAwsVO", {}).get("statmethod")
        raw = pd.DataFrame(res)

        with st.expander("🔧 API 원시 컬럼 보기", expanded=False):
            st.write(sorted(list(raw.columns)))

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

        if method == "A":
            want = ["일자","평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]
        elif method == "B":
            want = ["일자","평균기온","최고기온","최저기온","습도","강우량","일사량","평균풍속","최대풍속"]
        elif method == "C":
            want = ["순","평균기온","최고기온","최저기온","습도","강우량","일사량","평균풍속","최대풍속"] if "순" in raw.columns \
                else ["일자","평균기온","최고기온","최저기온","습도","강우량","일사량","평균풍속","최대풍속"]
        else:  # "D"
            want = ["월","평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"] \
                if "월" in raw.columns else \
                ["일자","평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]

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

    # -------------------------------------------------
    # 일/순 → 월별 집계
    # -------------------------------------------------
    def agg_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
        if not {"연도","월"}.issubset(df.columns):
            return df.copy()
        agg_map = {
            "평균기온":"mean","최고기온":"mean","최저기온":"mean","습도":"mean",
            "강우량":"sum","일사량":"sum","결로시간":"sum",
            "평균풍속":"mean","최대풍속":"mean"
        }
        use_cols = {k:v for k,v in agg_map.items() if k in df.columns}
        out = df.groupby(["연도","월"], as_index=False).agg(use_cols)
        return out.sort_values(["연도","월"]).reset_index(drop=True)

    # -------------------------------------------------
    # 월별 가로 확장 피처 (_mean_mMM / _sum_mMM)
    # -------------------------------------------------
    def build_wide_month_feats(env_m: pd.DataFrame) -> pd.DataFrame:
        if not {"연도","월"}.issubset(env_m.columns):
            raise ValueError("env_m에 '연도','월' 컬럼이 필요합니다.")
        num_cols = [c for c in env_m.columns if c not in ("연도","월") and pd.api.types.is_numeric_dtype(env_m[c])]
        mean_agg = env_m.groupby(["연도","월"], as_index=False)[num_cols].mean()
        sum_agg  = env_m.groupby(["연도","월"], as_index=False)[num_cols].sum()
        wide_mean = None
        for m in range(1, 13):
            sub = mean_agg[mean_agg["월"] == m].drop(columns=["월"]).copy()
            sub = sub.rename(columns={c: f"{c}_mean_m{m:02d}" for c in num_cols})
            wide_mean = sub if wide_mean is None else pd.merge(wide_mean, sub, on="연도", how="outer")
        wide_sum = None
        for m in range(1, 13):
            sub = sum_agg[sum_agg["월"] == m].drop(columns=["월"]).copy()
            sub = sub.rename(columns={c: f"{c}_sum_m{m:02d}" for c in num_cols})
            wide_sum = sub if wide_sum is None else pd.merge(wide_sum, sub, on="연도", how="outer")
        wide = pd.merge(wide_mean, wide_sum, on="연도", how="outer").fillna(0)
        return wide

    # -------------------------------------------------
    # 회귀식 적용
    # -------------------------------------------------
    def apply_equation_row(row: pd.Series, eq_str: str) -> float:
        rhs = eq_str.split("=", 1)[1].strip().replace("·", "*")
        cols = sorted(row.index.tolist(), key=len, reverse=True)
        expr = rhs
        for c in cols:
            expr = expr.replace(c, f"row[{repr(c)}]")
        return float(eval(expr, {"__builtins__": {}}, {"row": row, "np": np}))

    # -------------------------------------------------
    # 미확보 월 채우기
    # -------------------------------------------------
    def fill_missing_or_future_with_climatology(env_m: pd.DataFrame, target_year: int, cultivar: str, mode: str = "last3") -> pd.DataFrame:
        need_cols = {"연도","월"}
        if not need_cols.issubset(env_m.columns):
            raise ValueError("env_m에는 '연도','월' 컬럼이 있어야 합니다.")
        s_mon, e_mon = cultivar_window(cultivar)
        cur_year, cur_mon = get_today_ym()

        cur = env_m[env_m["연도"] == target_year].copy()
        hist = env_m[env_m["연도"] < target_year].copy()
        if hist.empty:
            return cur
        if mode == "last3":
            last_years = sorted(hist["연도"].unique())[-3:]
            hist = hist[hist["연도"].isin(last_years)]

        num_cols = [c for c in env_m.columns if c not in ("연도","월") and pd.api.types.is_numeric_dtype(env_m[c])]
        climo = hist.groupby("월", as_index=False)[num_cols].mean()

        months_window = list(range(s_mon, e_mon+1))
        have = set(cur["월"].tolist())
        future_cut = cur_mon if target_year == cur_year else 12
        future_months = [m for m in months_window if (target_year == cur_year and m > future_cut) or (target_year > cur_year)]
        missing_months = [m for m in months_window if m not in have]

        to_fill = sorted(set(future_months) | set(missing_months))
        if to_fill:
            fill_rows = climo[climo["월"].isin(to_fill)].copy()
            if fill_rows.empty:
                climo_all = env_m[env_m["연도"] < target_year].groupby("월", as_index=False)[num_cols].mean()
                fill_rows = climo_all[climo_all["월"].isin(to_fill)].copy()
            fill_rows.insert(0, "연도", target_year)
            cur = pd.concat([cur, fill_rows], ignore_index=True, axis=0)

        cur = cur[(cur["월"] >= s_mon) & (cur["월"] <= e_mon)]
        cur = cur.sort_values(["연도","월"]).reset_index(drop=True)
        for c in num_cols:
            cur[c] = pd.to_numeric(cur[c], errors="coerce")
        return cur

    # -------------------------------------------------
    # 전년도 과실품질 테이블
    # -------------------------------------------------
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
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        try:
            import io as _io
            tables = pd.read_html(_io.StringIO(r.text))
        except Exception as e:
            st.warning(f"전년도 과실품질 표 파싱에 실패했습니다: {e}\n"
                    "lxml/html5lib 설치 여부를 확인해 주세요. (pip install lxml html5lib)")
            return {}

        cleaned = []
        for t in tables:
            t2 = normalize_quality_columns(t)
            if set(["지역","수확일자"]).issubset(set(t2.columns)):
                cleaned.append(t2)

        result = {}
        if len(cleaned) >= 1:
            result["this"] = cleaned[0]
        if len(cleaned) >= 2:
            result["last"] = cleaned[1]
        return result

    def pick_region_row(qdf: pd.DataFrame, region_disp_name: str) -> Optional[pd.Series]:
        if qdf is None or qdf.empty:
            return None
        region_disp_name = _clean_str(region_disp_name)
        if "지역" not in qdf.columns:
            return None
        tmp = qdf.copy()
        tmp["지역"] = tmp["지역"].map(_clean_str)
        if "수확일자" in tmp.columns and not np.issubdtype(tmp["수확일자"].dtype, np.datetime64):
            tmp["수확일자"] = pd.to_datetime(tmp["수확일자"], errors="coerce")
        sub = tmp[tmp["지역"] == region_disp_name]
        if sub.empty:
            return None
        sub = sub.sort_values("수확일자", ascending=False, na_position="last")
        return sub.iloc[0]

    # -------------------------------------------------
    # 사이드바
    # -------------------------------------------------
    # -------------------------------------------------
    # 조회 조건 (메인 화면 상단에 표시)
    # -------------------------------------------------

    # 품종 선택 (라디오 버튼)

    cultivar = st.radio(
    "품종 선택", ["홍로", "후지"], key="cultivar_radio"
)


    # 지역 선택 (드롭다운)

    region = st.selectbox(
        "지역 선택",
        list(AREA_CODE.keys()),
        index=1
    )

    # 🔎 버튼
    run = st.button("🔎 자동조회 & 예측")

    # 예상 날씨 방법은 제거하고, 항상 'all'(전체 과거 평균)으로 고정
    mode = "all"


    # -------------------------------------------------
    # 실행
    # -------------------------------------------------
    if run:
        try:
            cur_year, cur_month = get_today_ym()
            s_mon, e_mon = cultivar_window(cultivar)
            s_ym = f"{cur_year:04d}-{s_mon:02d}"
            e_ym_real = f"{cur_year:04d}-{min(e_mon, cur_month):02d}"

            with st.spinner("기상 데이터를 불러오고 있습니다..."):
                payload = fetch_aws_stat(region, "D", s_ym, e_ym_real)

            if "error" in payload:
                st.error("응답이 JSON 형식이 아닙니다.")
                st.code(payload.get("raw","")[:1000])
                st.stop()

            df = json_to_dataframe(payload)
            st.subheader("올해 월별 실측 데이터(기상)")
            if df.empty:
                st.warning("올해 기간 내 실측 데이터가 없습니다. 과거 기후평년만으로 채워 예측합니다.")
            else:
                st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error("예상치 못한 오류가 발생했습니다. 아래 예외를 확인해 주세요.")
            st.exception(e)

        env_m = df.copy()
        if not env_m.empty and "연도" not in env_m.columns and "월" in env_m.columns:
            env_m["연도"] = env_m["월"].astype(str).str[:4].astype(int)
            env_m["월"] = env_m["월"].astype(str).str[5:7].astype(int)

        past_payload = fetch_aws_stat(region, "D", f"{max(cur_year-15,2010):04d}-01", f"{cur_year-1:04d}-12")
        past_df = json_to_dataframe(past_payload)
        env_all = pd.concat([env_m, past_df], ignore_index=True) if not past_df.empty else env_m

        filled_this_year = fill_missing_or_future_with_climatology(env_all, cur_year, cultivar, mode=mode)

        st.subheader("예측에 사용된 월별 데이터(올해, 미래월은 예상 날씨로 대체)")
        st.dataframe(filled_this_year, use_container_width=True)

        try:
            env_for_wide = pd.concat([env_all[env_all["연도"] != cur_year], filled_this_year], ignore_index=True)
            wide = build_wide_month_feats(env_for_wide)
        except Exception as e:
            st.error(f"월별 피처 생성 실패: {e}")
            st.stop()

        if cur_year not in set(wide["연도"].astype(int).tolist()):
            st.error("예측 연도 행을 구성하지 못했습니다.")
            st.stop()

        row = wide[wide["연도"] == cur_year].iloc[0]

        EQUATIONS = EQUATIONS_BY_CULTIVAR.get(cultivar, {})
        preds = {}
        for tgt, formula in EQUATIONS.items():
            try:
                preds[tgt] = apply_equation_row(row, formula)
            except Exception as e:
                preds[tgt] = f"에러: {e}"

        st.subheader(f"회귀식 예측 결과  품종: {cultivar}  연도: {cur_year}")

        pred_df = pd.DataFrame([preds]).T.reset_index()
        pred_df.columns = ["항목", "예측값(올해)"]

        # 행/열 바꾸기
        pred_df_t = pred_df.set_index("항목").T.reset_index(drop=True)

        st.dataframe(pred_df_t, use_container_width=True)


        # -----------------------------
        # 전년도 과실품질 표 불러와서 지역 행 추출
        # -----------------------------
        with st.spinner("전년도 과실품질 데이터를 불러오는 중..."):
            qdict = fetch_quality_tables(cur_year, cur_year-1, cultivar)

        region_disp = REGION_NAME_MAP.get(region, region)
        last_row = None
        if qdict and "last" in qdict and qdict["last"] is not None and not qdict["last"].empty:
            q_last = normalize_quality_columns(qdict["last"])
            with st.expander("전년도 테이블 실제 컬럼명(정규화 후)"):
                st.write(list(q_last.columns))
            last_row = pick_region_row(q_last, region_disp)

        if last_row is not None:
            # 8개 지표만 패턴 기반으로 매핑
            patterns = {
                "과중": r"^과중",
                "종경": r"^종경",
                "횡경": r"^횡경",
                # 경도평균(N/ø11mm) 주변 표기 변형 허용
                "경도": r"(경도\s*평균|경도평균|N\s*/?\s*ø?\s*11\s*mm)",
                # °Brix / ˚Brix 모두 허용
                "당도": r"^당도(\s*\((°|˚)?\s*Brix\))?",
                "산도": r"^산도(\s*\(%\))?",
                "L":   r"착색.*Hunter\s*L\b",
                "a":   r"착색.*Hunter\s*a\b",
                "b":   r"착색.*Hunter\s*b\b",
            }

            def to_float(x):
                try:
                    return float(str(x).replace(",", "").strip())
                except Exception:
                    return None

            rows = []
            # last_row는 Series라서 컬럼 탐색을 위해 1행 DataFrame으로 변환
            last_df_for_match = last_row.to_frame().T
            for k, pat in patterns.items():
                col = get_first_col_by_pattern(last_df_for_match, pat)
                last_val = to_float(last_row[col]) if col else None
                pred_val = preds.get(k, None)
                rows.append([k, pred_val, last_val])

            compare_df = pd.DataFrame(rows, columns=["항목","예측값(올해)","전년도 실제값"])
            st.subheader(f"올해 예측 vs 전년도 실제  비교  지역: {region_disp}  품종: {cultivar}")

            # 항목을 열로, 값 구분을 인덱스로
            compare_df_t = compare_df.set_index("항목").T
            compare_df_t.index.name = ""  # 인덱스 제목 제거
            st.dataframe(compare_df_t, use_container_width=True)


            # ====== 그래프 섹션 (레이아웃/색상/크기 반영) ======
            PRED_COLOR = "#87CEEB"   # 예측(올해): 하늘색
            LAST_COLOR = "#800080"   # 전년도: 자줏빛

            def _pick(df, item):
                r = df[df["항목"] == item]
                if r.empty: return np.nan, np.nan
                p = pd.to_numeric(r["예측값(올해)"].values[0], errors="coerce")
                l = pd.to_numeric(r["전년도 실제값"].values[0], errors="coerce")
                return p, l

            # ── 1) 과실 크기(과중·횡경·종경 한 그래프에)
            size_items = ["과중", "횡경", "종경"]
            x = np.arange(len(size_items))
            y_pred = []; y_last = []
            for it in size_items:
                p, l = _pick(compare_df, it)
                y_pred.append(np.nan if pd.isna(p) else float(p))
                y_last.append(np.nan if pd.isna(l) else float(l))

            if not (all(pd.isna(y_pred)) and all(pd.isna(y_last))):
                fig, ax = plt.subplots(figsize=(3.5, 2.6))
                w = 0.35
                if not all(pd.isna(y_pred)):
                    ax.bar(x - w/2, np.nan_to_num(y_pred, nan=0.0), width=w, label="예측(올해)", color=PRED_COLOR)
                if not all(pd.isna(y_last)):
                    ax.bar(x + w/2, np.nan_to_num(y_last, nan=0.0), width=w, label="전년도", color=LAST_COLOR)
                ax.set_xticks(x); ax.set_xticklabels(size_items)
                ax.set_title("과실 크기")
                ax.grid(axis="y", linestyle=":", alpha=0.35)
                # 테두리선 얇게, 위/오른쪽 제거
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_linewidth(0.7)
                ax.spines["bottom"].set_linewidth(0.7)
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig, use_container_width=False)
            else:
                st.info("과실 크기(과중/횡경/종경) 데이터가 없어 그래프를 표시하지 않습니다.")

            # 공통: 단일 항목 막대 그래프(작게)
            def _bar_single(item, title):
                p, l = _pick(compare_df, item)
                if pd.isna(p) and pd.isna(l): 
                    return
                xs, ys, cs = [], [], []
                if not pd.isna(p): xs.append("예측(올해)"); ys.append(float(p)); cs.append(PRED_COLOR)
                if not pd.isna(l): xs.append("전년도");     ys.append(float(l)); cs.append(LAST_COLOR)
                fig, ax = plt.subplots(figsize=(2.5, 1.8))
                ax.bar(np.arange(len(xs)), ys, tick_label=xs, color=cs, width=0.35)
                ax.set_title(title)
                ax.grid(axis="y", linestyle=":", alpha=0.35)
                # 테두리선 얇게, 위/오른쪽 제거
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_linewidth(0.7)
                ax.spines["bottom"].set_linewidth(0.7)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=False)

            # ── 2) 경도, 3) 당도, 4) 산도 (가로로 나란히)
            c1, c2, c3 = st.columns(3)

            with c1:
                # 경도 그래프 y축 최대값 70으로 설정
                p, l = _pick(compare_df, "경도")
                if pd.isna(p) and pd.isna(l): 
                    pass
                else:
                    xs, ys, cs = [], [], []
                    if not pd.isna(p): xs.append("예측(올해)"); ys.append(float(p)); cs.append(PRED_COLOR)
                    if not pd.isna(l): xs.append("전년도");     ys.append(float(l)); cs.append(LAST_COLOR)
                    fig, ax = plt.subplots(figsize=(2.5, 1.8))
                    ax.bar(np.arange(len(xs)), ys, tick_label=xs, color=cs, width=0.35)
                    ax.set_title("경도")
                    ax.grid(axis="y", linestyle=":", alpha=0.35)
                    ax.set_ylim(top=70)
                    # 테두리선 얇게, 위/오른쪽 제거
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_linewidth(0.7)
                    ax.spines["bottom"].set_linewidth(0.7)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=False)

            with c2:
                _bar_single("당도", "당도")
            with c3:
                _bar_single("산도", "산도")

            # ── 5) 착색도 L/a/b (꺾은선 그래프, 세로 맨 아래)
            tone_items = ["L", "a", "b"]
            x = np.arange(len(tone_items))
            y_pred = []; y_last = []
            for it in tone_items:
                p, l = _pick(compare_df, it)
                y_pred.append(np.nan if pd.isna(p) else float(p))
                y_last.append(np.nan if pd.isna(l) else float(l))

            if not (all(pd.isna(y_pred)) and all(pd.isna(y_last))):
                fig, ax = plt.subplots(figsize=(3.5, 2.6))
                if not all(pd.isna(y_pred)):
                    ax.plot(x, y_pred, marker="o", linewidth=2, label="예측(올해)", color=PRED_COLOR)
                if not all(pd.isna(y_last)):
                    ax.plot(x, y_last, marker="o", linewidth=2, label="전년도", color=LAST_COLOR)
                ax.set_xticks(x); ax.set_xticklabels(tone_items)
                ax.set_title("착색도")
                ax.grid(axis="y", linestyle=":", alpha=0.35)
                # 테두리선 얇게, 위/오른쪽 제거
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_linewidth(0.7)
                ax.spines["bottom"].set_linewidth(0.7)
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig, use_container_width=False)
            else:
                st.info("착색도(L/a/b) 데이터가 없어 그래프를 표시하지 않습니다.")
            # ====== /그래프 섹션 끝 ======



        else:
            st.warning("전년도 과실품질에서 해당 지역 행을 찾지 못했습니다. 품종/지역 조합을 바꿔보세요.")

        # 원자료 다운로드
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "⬇️ 올해 실측 월별 CSV",
                df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{region}_{cultivar}_{cur_year}_monthly_measured.csv",
                mime="text/csv",
                disabled=df.empty
            )
        with c2:
            st.download_button(
                "⬇️ 올해 예측에 사용한 월별 CSV",
                filled_this_year.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{region}_{cultivar}_{cur_year}_monthly_used.csv",
                mime="text/csv"
            )
        with c3:
            st.download_button(
                "⬇️ 회귀식 예측 결과 CSV",
                pred_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{region}_{cultivar}_{cur_year}_predictions.csv",
                mime="text/csv"
            )
    else:
        st.info("좌측에서 지역과 품종을 고른 뒤 자동조회 & 예측 버튼을 눌러주세요. 올해 남은 월은 ‘예상 날씨(최근 3년 또는 전체 과거 평균)’로 채워 예측합니다.")

# ==========================================================
# 탭1: 분석결과(군위) - 이미지/표 뷰어 (홍로/후지)
# ==========================================================
with tab3:
    st.title("분석결과(군위)")
    BASE_DIR = Path(__file__).parent / "mba" / "분석결과" / "군위"
    CULTIVARS = ["홍로", "후지"]

    # 결과 자산(이미지/표) 자동 검색: 하위에 '홍로','후지' 폴더가 있거나, 파일명에 품종명이 들어있는 경우 모두 지원
    IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
    TAB_EXT = (".csv", ".xlsx")

    def get_assets_for_cultivar(cultivar: str):
        # 1) 폴더 모드: BASE_DIR/홍로, BASE_DIR/후지 존재 시
        folder1 = BASE_DIR / cultivar
        if folder1.exists():
            imgs = sorted([p for p in folder1.rglob("*") if p.suffix.lower() in IMG_EXT])
            tabs = sorted([p for p in folder1.rglob("*") if p.suffix.lower() in TAB_EXT])
            root = folder1
        else:
            # 2) 파일명 필터 모드: BASE_DIR 내 파일명에 품종 포함
            imgs = sorted([p for p in BASE_DIR.glob("*") if (p.suffix.lower() in IMG_EXT and cultivar in p.stem)])
            tabs = sorted([p for p in BASE_DIR.glob("*") if (p.suffix.lower() in TAB_EXT and cultivar in p.stem)])
            root = BASE_DIR
        return root, imgs, tabs

    cultivar = st.radio("품종 선택", CULTIVARS, horizontal=True, key="radio_tab2")

    folder, all_imgs, all_tabs = get_assets_for_cultivar(cultivar)
    if not BASE_DIR.exists():
        st.error(f"군위 결과 폴더가 없습니다: {BASE_DIR}")
        st.stop()

    mode = st.segmented_control("표시 유형", options=["이미지", "표(데이터)"], key="seg_tab2")
    if mode == "이미지":
        if not all_imgs:
            st.warning("표시할 이미지가 없습니다. (경로/파일명을 확인하세요)")
        else:
            view = st.radio("방식 선택", ["갤러리(썸네일)", "단일 파일"], horizontal=True, key="view_tab2")
            if view == "갤러리(썸네일)":
                thumbs = st.slider("한 줄에 몇 장", min_value=2, max_value=6, value=4, key="thumbs_slider_tab3")
                rows = (len(all_imgs) + thumbs - 1) // thumbs
                st.caption(f"총 {len(all_imgs)}개 이미지")
                idx = 0
                for _ in range(rows):
                    cols = st.columns(thumbs, gap="small")
                    for c in cols:
                        if idx >= len(all_imgs):
                            break
                        p = all_imgs[idx]
                        with c:
                            st.image(str(p), caption=str(p.relative_to(folder)))
                        idx += 1
            else:
                sel = st.selectbox("이미지 선택", [str(p.relative_to(folder)) for p in all_imgs])
                path = folder / sel
                st.image(str(path), caption=str(path))
    else:
        if not all_tabs:
            st.warning("표시할 CSV/XLSX 파일이 없습니다.")
        else:
            sel = st.selectbox("표 선택", [str(p.relative_to(folder)) for p in all_tabs])
            path = folder / sel
            st.subheader("표시 중: " + str(path.name))
            try:
                if path.suffix.lower() == ".csv":
                    df = pd.read_csv(path, encoding="utf-8-sig")
                else:
                    df = pd.read_excel(path)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"불러오기 실패: {e}")

# ==========================================================
# 탭2: 과실품질 예측(군위 고정)
# ==========================================================
with tab4:
    st.markdown("과실품질 예측(군위)")

    # ---------------------------
    # 지역/코드: 군위 전용
    # ---------------------------
    AREA_CODE = {"대구군위": "333"}  # API용 키
    REGION_NAME_MAP = {"대구군위": "군위"}  # 화면 표시명

    # ---------------------------
    # 회귀식 (군위용)
    # ---------------------------
    EQUATIONS_BY_CULTIVAR = {
    "홍로": {
        "과중": "과중 = 157.237 -137.847·평균풍속_mean_m08 +1.312·최고기온_mean_m07 +0.205849·강우량_sum_m05 +3.9929·최저기온_mean_m07",
        "종경": "종경 = 36.9151 +0.613954·최저기온_mean_m04 +0.244159·습도_mean_m04 +0.577148·최고기온_mean_m07 +0.0405461·강우량_sum_m05",
        "횡경": "횡경 = 89.2207 -19.1684·평균풍속_mean_m08 +0.00285688·결로시간_mean_m04 +0.0137361·강우량_sum_m05",
        "L":   "L = -107.653 +4.02276·최고기온_mean_m06 +0.557626·최고기온_mean_m08 +0.357053·습도_mean_m05 +1.23389·최대풍속_mean_m08 +5.79158·평균풍속_mean_m07",
        "a":   "a = 28.916 +26.4391·최대풍속_mean_m05 -2.04984·최고기온_mean_m06 -0.525233·평균기온_mean_m08 -7.60126·최대풍속_mean_m08 -0.0905347·습도_mean_m05",
        "b":   "b = -44.3799 -0.0102677·최대풍속_mean_m05 +1.63975·최고기온_mean_m06 +0.201082·습도_mean_m05 +1.29407·최대풍속_mean_m08",
        "경도": "경도 = 53.7403 +0.0406574·일사량_sum_m08 +0.0109639·결로시간_mean_m04 -5.99111·최대풍속_mean_m05 +0.00101735·결로시간_mean_m05",
        "당도": "당도 = -3.77559 +0.845922·최고기온_mean_m06 -0.000435945·강우량_sum_m07 +0.00284123·결로시간_mean_m04 -0.343821·평균기온_mean_m06 +0.0040703·강우량_sum_m06",
        "산도": "산도 = 0.23201 -0.000127841·결로시간_mean_m05 -1.60354e-05·강우량_sum_m05 +0.0112801·평균풍속_mean_m06 +0.0202028·최대풍속_mean_m05 -7.47002e-05·결로시간_mean_m04",
    },
    "후지": {
        "과중": "과중 = 280.193 -14.9938·최대풍속_mean_m04 +7.01039·평균기온_mean_m09 -0.216694·일사량_sum_m10",
        "종경": "종경 = 67.5821 +0.707095·최저기온_mean_m04 -0.0488712·일사량_sum_m10 +0.880706·최고기온_mean_m09",
        "횡경": "횡경 = 95.1048 +0.181127·습도_mean_m04 -0.0132518·일사량_sum_m04 -0.276929·습도_mean_m08 +0.487867·최고기온_mean_m05",
        "L":   "L = -25.351 +0.317467·습도_mean_m04 +0.063049·최저기온_mean_m09 +0.310649·습도_mean_m10 +1.19993·최고기온_mean_m10",
        "a":   "a = 24.306 -0.0581428·강우량_sum_m04 +7.83592·평균풍속_mean_m08 -0.515246·최고기온_mean_m10 +6.87267·평균풍속_mean_m05",
        "b":   "b = 18.3498 -0.0117526·습도_mean_m10 -0.0255657·일사량_sum_m10 -5.44522·평균풍속_mean_m09 +0.493188·최고기온_mean_m04",
        "경도": "경도 = 44.1071 -0.0341814·강우량_sum_m10 +0.00537892·강우량_sum_m07 +4.75772·최대풍속_mean_m08 +1.15897·최저기온_mean_m04",
        "당도": "당도 = 7.35627 +2.56347·최대풍속_mean_m08 -0.0270168·습도_mean_m09 +0.0171314·평균기온_mean_m06 +0.525697·최저기온_mean_m06 -0.231632·최고기온_mean_m05",
        "산도": "산도 = 1.14472 +0.169768·평균풍속_mean_m08 -0.0438612·최고기온_mean_m10 +5.80039e-05·일사량_sum_m04 +0.103592·평균풍속_mean_m05 -0.000376686·강우량_sum_m04",
    },
}


    # ---------------------------
    # 유틸/전처리 함수
    # ---------------------------
    def _clean_str(s: str) -> str:
        s = str(s).replace("\xa0", " ")
        s = s.replace("\n", " ").replace("\r", " ")
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

    def _ensure_numeric(df: pd.DataFrame, cols):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def cultivar_window(cultivar: str):
        return (4, 8) if cultivar == "홍로" else (4, 10)

    def get_today_ym():
        now = datetime.now()
        return now.year, now.month

    @st.cache_data(show_spinner=False, ttl=300)
    def fetch_aws_stat(region_name: str, stat_gb_code: str, s_ym: str, e_ym: str) -> dict:
        session = requests.Session()
        session.get("https://fruit.nihhs.go.kr/apple/aws/awsStat.do", timeout=20)
        form = {
            "pageIndex": "1",
            "searchNum": "0",
            "areaName": region_name,
            "areaCode": AREA_CODE.get(region_name, region_name),
            "statmethod": stat_gb_code,
        }
        if stat_gb_code == "A":
            form["wetherDtBgn"] = f"{s_ym}-01"; form["wetherDtEnd"] = f"{e_ym}-30"
        elif stat_gb_code in ("B", "C"):
            form["wetherDtM"] = s_ym
        else:  # "D"
            form["wetherDtBgn2"] = s_ym; form["wetherDtEnd2"] = e_ym
            form["wetherDtBgn"]  = s_ym; form["wetherDtEnd"]  = e_ym
        resp = session.post(
            "https://fruit.nihhs.go.kr/apple/aws/awsStatList.do",
            data=form, timeout=30,
            headers={"Referer": "https://fruit.nihhs.go.kr/apple/aws/awsStat.do"}
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {"error": "JSON 파싱 실패", "raw": resp.text[:4000]}

    def json_to_dataframe(payload: dict) -> pd.DataFrame:
        if not payload or "result" not in payload:
            return pd.DataFrame()
        res = payload.get("result", [])
        if len(res) == 0:
            return pd.DataFrame()
        method = payload.get("mainAwsVO", {}).get("statmethod")
        raw = pd.DataFrame(res)
        with st.expander("🔧 API 원시 컬럼 보기", expanded=False):
            st.write(sorted(list(raw.columns)))
        rename_map = {
            "statsDt": "일자", "wetherDt": "일자",
            "dalyWetherAvrgTp": "평균기온", "wetherAvgTp": "평균기온",
            "dalyWetherMxmmTp": "최고기온", "wetherMaxTp": "최고기온",
            "dalyWetherMummTp": "최저기온", "wetherMinTp": "최저기온",
            "dalyWetherAvrgHd": "습도", "WetherAvgHd": "습도",
            "dalyWetherTtalRainqy": "강우량", "wetherMaxRainqy": "강우량",
            "dalyWetherMxmmSolradqy": "일사량", "wetherMaxSolradqy": "일사량", "wetherSumSolradqy": "일사량",
            "dalyWetherMxmmCondenstime": "결로시간", "wetherMaxCondenstime": "결로시간", "wetherSumCondenstime": "결로시간",
            "dalyWetherAvrgWs": "평균풍속", "wetherAvgWs": "평균풍속",
            "dalyWetherMxmmWs": "최대풍속", "wetherMaxWs": "최대풍속",
            "wetherDtMonth": "월", "wetherDt": "월",
        }
        raw = raw.rename(columns=rename_map)
        if method == "A":
            want = ["일자","평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]
        elif method == "B":
            want = ["일자","평균기온","최고기온","최저기온","습도","강우량","일사량","평균풍속","최대풍속"]
        elif method == "C":
            want = ["순","평균기온","최고기온","최저기온","습도","강우량","일사량","평균풍속","최대풍속"] if "순" in raw.columns \
                else ["일자","평균기온","최고기온","최저기온","습도","강우량","일사량","평균풍속","최대풍속"]
        else:
            want = ["월","평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"] \
                if "월" in raw.columns else \
                ["일자","평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]
        use_cols = [c for c in want if c in raw.columns]
        if not use_cols:
            return pd.DataFrame()
        df = raw[use_cols].copy()
        numcands = ["평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]
        df = _ensure_numeric(df, [c for c in numcands if c in df.columns])
        if "일자" in df.columns and df["일자"].dtype != "int64":
            df["일자"] = pd.to_datetime(df["일자"], errors="coerce")
            if "연도" not in df.columns: df["연도"] = df["일자"].dt.year
            if "월"   not in df.columns: df["월"]   = df["일자"].dt.month
        elif "월" in df.columns:
            df["연도"] = pd.to_numeric(df["월"].astype(str).str[:4], errors="coerce")
            df["월"]   = pd.to_numeric(df["월"].astype(str).str[-2:], errors="coerce")
        if {"연도","월"}.issubset(df.columns):
            df = df.sort_values(["연도","월"]).reset_index(drop=True)
        return df

    def agg_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
        if not {"연도","월"}.issubset(df.columns):
            return df.copy()
        agg_map = {
            "평균기온":"mean","최고기온":"mean","최저기온":"mean","습도":"mean",
            "강우량":"sum","일사량":"sum","결로시간":"sum",
            "평균풍속":"mean","최대풍속":"mean"
        }
        use_cols = {k:v for k,v in agg_map.items() if k in df.columns}
        out = df.groupby(["연도","월"], as_index=False).agg(use_cols)
        return out.sort_values(["연도","월"]).reset_index(drop=True)

    def build_wide_month_feats(env_m: pd.DataFrame) -> pd.DataFrame:
        if not {"연도","월"}.issubset(env_m.columns):
            raise ValueError("env_m에 '연도','월' 컬럼이 필요합니다.")
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
        return wide

    def apply_equation_row(row: pd.Series, eq_str: str) -> float:
        rhs = eq_str.split("=", 1)[1].strip().replace("·", "*")
        cols = sorted(row.index.tolist(), key=len, reverse=True)
        expr = rhs
        for c in cols:
            expr = expr.replace(c, f"row[{repr(c)}]")
        return float(eval(expr, {"__builtins__": {}}, {"row": row, "np": np}))

    def fill_missing_or_future_with_climatology(env_m: pd.DataFrame, target_year: int, cultivar: str, mode: str = "all") -> pd.DataFrame:
        need_cols = {"연도","월"}
        if not need_cols.issubset(env_m.columns):
            raise ValueError("env_m에는 '연도','월' 컬럼이 있어야 합니다.")
        s_mon, e_mon = cultivar_window(cultivar)
        cur_year, cur_mon = get_today_ym()
        cur = env_m[env_m["연도"] == target_year].copy()
        hist = env_m[env_m["연도"] < target_year].copy()
        if hist.empty:
            return cur
        # 'all' == 전체 과거 평균 사용
        num_cols = [c for c in env_m.columns if c not in ("연도","월") and pd.api.types.is_numeric_dtype(env_m[c])]
        climo_all = hist.groupby("월", as_index=False)[num_cols].mean()
        months_window = list(range(s_mon, e_mon+1))
        have = set(cur["월"].tolist())
        future_cut = cur_mon if target_year == cur_year else 12
        future_months = [m for m in months_window if (target_year == cur_year and m > future_cut) or (target_year > cur_year)]
        missing_months = [m for m in months_window if m not in have]
        to_fill = sorted(set(future_months) | set(missing_months))
        if to_fill:
            fill_rows = climo_all[climo_all["월"].isin(to_fill)].copy()
            fill_rows.insert(0, "연도", target_year)
            cur = pd.concat([cur, fill_rows], ignore_index=True, axis=0)
        cur = cur[(cur["월"] >= s_mon) & (cur["월"] <= e_mon)].sort_values(["연도","월"]).reset_index(drop=True)
        for c in num_cols:
            cur[c] = pd.to_numeric(cur[c], errors="coerce")
        return cur

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
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        try:
            tables = pd.read_html(io.StringIO(r.text))
        except Exception as e:
            st.warning(f"전년도 과실품질 표 파싱 실패: {e}\n(lxml/html5lib 필요)")
            return {}
        cleaned = []
        for t in tables:
            t2 = normalize_quality_columns(t)
            if set(["지역","수확일자"]).issubset(set(t2.columns)):
                cleaned.append(t2)
        result = {}
        if len(cleaned) >= 1: result["this"] = cleaned[0]
        if len(cleaned) >= 2: result["last"] = cleaned[1]
        return result

    def pick_region_row(qdf: pd.DataFrame, region_disp_name: str) -> Optional[pd.Series]:
        if qdf is None or qdf.empty: return None
        region_disp_name = _clean_str(region_disp_name)
        if "지역" not in qdf.columns: return None
        tmp = qdf.copy()
        tmp["지역"] = tmp["지역"].map(_clean_str)
        if "수확일자" in tmp.columns and not np.issubdtype(tmp["수확일자"].dtype, np.datetime64):
            tmp["수확일자"] = pd.to_datetime(tmp["수확일자"], errors="coerce")
        sub = tmp[tmp["지역"] == region_disp_name]
        if sub.empty: return None
        sub = sub.sort_values("수확일자", ascending=False, na_position="last")
        return sub.iloc[0]

    # ---------------------------
    # 조회/예측 UI (군위 고정)
    # ---------------------------
    c1, c2 = st.columns([1,1])
    with c1:
        cultivar = st.radio("품종 선택", ["홍로", "후지"], horizontal=True, key="radio_tab3")
    with c2:
        st.text("지역 선택")
        st.info("군위(대구군위)로 고정")
    region = "대구군위"  # 내부 키
    region_disp = REGION_NAME_MAP[region]

    run = st.button("🔎 군위 자동조회 & 예측")
    mode = "all"  # 항상 전체 과거 평균 기반으로 미확보 월 채움

    if run:
        try:
            cur_year, cur_month = get_today_ym()
            s_mon, e_mon = cultivar_window(cultivar)
            s_ym = f"{cur_year:04d}-{s_mon:02d}"
            e_ym_real = f"{cur_year:04d}-{min(e_mon, cur_month):02d}"

            with st.spinner("기상 데이터를 불러오고 있습니다..."):
                payload = fetch_aws_stat(region, "D", s_ym, e_ym_real)

            if "error" in payload:
                st.error("응답이 JSON 형식이 아닙니다.")
                st.code(payload.get("raw","")[:1000])
                st.stop()

            df = json_to_dataframe(payload)
            st.subheader(f"올해 월별 실측 데이터(기상) - {region_disp}")
            if df.empty:
                st.warning("올해 기간 내 실측 데이터가 없습니다. 과거 평균으로 채워 예측합니다.")
            else:
                st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error("예상치 못한 오류가 발생했습니다.")
            st.exception(e)

        env_m = df.copy()
        if not env_m.empty and "연도" not in env_m.columns and "월" in env_m.columns:
            env_m["연도"] = env_m["월"].astype(str).str[:4].astype(int)
            env_m["월"]   = env_m["월"].astype(str).str[5:7].astype(int)

        past_payload = fetch_aws_stat(region, "D", f"{max(cur_year-15,2010):04d}-01", f"{cur_year-1:04d}-12")
        past_df = json_to_dataframe(past_payload)
        env_all = pd.concat([env_m, past_df], ignore_index=True) if not past_df.empty else env_m

        filled_this_year = fill_missing_or_future_with_climatology(env_all, cur_year, cultivar, mode=mode)

        st.subheader("예측에 사용된 월별 데이터(올해, 미래월은 과거 평균으로 대체)")
        st.dataframe(filled_this_year, use_container_width=True)

        try:
            env_for_wide = pd.concat([env_all[env_all["연도"] != cur_year], filled_this_year], ignore_index=True)
            wide = build_wide_month_feats(env_for_wide)
        except Exception as e:
            st.error(f"월별 피처 생성 실패: {e}")
            st.stop()

        if cur_year not in set(wide["연도"].astype(int).tolist()):
            st.error("예측 연도 행을 구성하지 못했습니다.")
            st.stop()

        row = wide[wide["연도"] == cur_year].iloc[0]

        EQUATIONS = EQUATIONS_BY_CULTIVAR.get(cultivar, {})
        preds = {}
        for tgt, formula in EQUATIONS.items():
            try:
                preds[tgt] = apply_equation_row(row, formula)
            except Exception as e:
                preds[tgt] = f"에러: {e}"

        st.subheader(f"회귀식 예측 결과  지역: {region_disp}  품종: {cultivar}  연도: {cur_year}")
        pred_df = pd.DataFrame([preds]).T.reset_index()
        pred_df.columns = ["항목", "예측값(올해)"]
        st.dataframe(pred_df.set_index("항목").T.reset_index(drop=True), use_container_width=True)

        # 전년도 과실품질 비교
        with st.spinner("전년도 과실품질 데이터를 불러오는 중..."):
            qdict = fetch_quality_tables(cur_year, cur_year-1, cultivar)

        last_row = None
        if qdict and "last" in qdict and qdict["last"] is not None and not qdict["last"].empty:
            q_last = normalize_quality_columns(qdict["last"])
            with st.expander("전년도 테이블 실제 컬럼명(정규화 후)"):
                st.write(list(q_last.columns))
            last_row = pick_region_row(q_last, region_disp)

        if last_row is not None:
            patterns = {
                "과중": r"^과중",
                "종경": r"^종경",
                "횡경": r"^횡경",
                "경도": r"(경도\s*평균|경도평균|N\s*/?\s*ø?\s*11\s*mm)",
                "당도": r"^당도(\s*\((°|˚)?\s*Brix\))?",
                "산도": r"^산도(\s*\(%\))?",
                "L":   r"착색.*Hunter\s*L\b",
                "a":   r"착색.*Hunter\s*a\b",
                "b":   r"착색.*Hunter\s*b\b",
            }
            def to_float(x):
                try:
                    return float(str(x).replace(",", "").strip())
                except Exception:
                    return None
            rows = []
            last_df_for_match = last_row.to_frame().T
            for k, pat in patterns.items():
                col = get_first_col_by_pattern(last_df_for_match, pat)
                last_val = to_float(last_row[col]) if col else None
                pred_val = preds.get(k, None)
                rows.append([k, pred_val, last_val])

            compare_df = pd.DataFrame(rows, columns=["항목","예측값(올해)","전년도 실제값"])
            st.subheader(f"올해 예측 vs 전년도 실제  비교  지역: {region_disp}  품종: {cultivar}")
            compare_df_t = compare_df.set_index("항목").T
            compare_df_t.index.name = ""
            st.dataframe(compare_df_t, use_container_width=True)

            # 작은 시각화들
            PRED_COLOR = "#87CEEB"
            LAST_COLOR = "#800080"
            def _pick(df, item):
                r = df[df["항목"] == item]
                if r.empty: return np.nan, np.nan
                p = pd.to_numeric(r["예측값(올해)"].values[0], errors="coerce")
                l = pd.to_numeric(r["전년도 실제값"].values[0], errors="coerce")
                return p, l

            size_items = ["과중", "횡경", "종경"]
            x = np.arange(len(size_items))
            y_pred, y_last = [], []
            for it in size_items:
                p, l = _pick(compare_df, it)
                y_pred.append(np.nan if pd.isna(p) else float(p))
                y_last.append(np.nan if pd.isna(l) else float(l))
            if not (all(pd.isna(y_pred)) and all(pd.isna(y_last))):
                fig, ax = plt.subplots(figsize=(3.5, 2.6))
                w = 0.35
                if not all(pd.isna(y_pred)):
                    ax.bar(x - w/2, np.nan_to_num(y_pred, nan=0.0), width=w, label="예측(올해)", color=PRED_COLOR)
                if not all(pd.isna(y_last)):
                    ax.bar(x + w/2, np.nan_to_num(y_last, nan=0.0), width=w, label="전년도", color=LAST_COLOR)
                ax.set_xticks(x); ax.set_xticklabels(size_items)
                ax.set_title("과실 크기")
                ax.grid(axis="y", linestyle=":", alpha=0.35)
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                ax.spines["left"].set_linewidth(0.7); ax.spines["bottom"].set_linewidth(0.7)
                ax.legend(); fig.tight_layout()
                st.pyplot(fig, use_container_width=False)

            def _bar_single(item, title, ylim_top=None):
                p, l = _pick(compare_df, item)
                if pd.isna(p) and pd.isna(l): 
                    return
                xs, ys, cs = [], [], []
                if not pd.isna(p): xs.append("예측(올해)"); ys.append(float(p)); cs.append(PRED_COLOR)
                if not pd.isna(l): xs.append("전년도");     ys.append(float(l)); cs.append(LAST_COLOR)
                fig, ax = plt.subplots(figsize=(2.5, 1.8))
                ax.bar(np.arange(len(xs)), ys, tick_label=xs, color=cs, width=0.35)
                ax.set_title(title); ax.grid(axis="y", linestyle=":", alpha=0.35)
                if ylim_top: ax.set_ylim(top=ylim_top)
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                ax.spines["left"].set_linewidth(0.7); ax.spines["bottom"].set_linewidth(0.7)
                fig.tight_layout(); st.pyplot(fig, use_container_width=False)

            c1, c2, c3 = st.columns(3)
            with c1: _bar_single("경도", "경도", ylim_top=70)
            with c2: _bar_single("당도", "당도")
            with c3: _bar_single("산도", "산도")

            tone_items = ["L", "a", "b"]
            x = np.arange(len(tone_items))
            y_pred, y_last = [], []
            for it in tone_items:
                p, l = _pick(compare_df, it)
                y_pred.append(np.nan if pd.isna(p) else float(p))
                y_last.append(np.nan if pd.isna(l) else float(l))
            if not (all(pd.isna(y_pred)) and all(pd.isna(y_last))):
                fig, ax = plt.subplots(figsize=(3.5, 2.6))
                if not all(pd.isna(y_pred)):
                    ax.plot(x, y_pred, marker="o", linewidth=2, label="예측(올해)", color=PRED_COLOR)
                if not all(pd.isna(y_last)):
                    ax.plot(x, y_last, marker="o", linewidth=2, label="전년도", color=LAST_COLOR)
                ax.set_xticks(x); ax.set_xticklabels(tone_items)
                ax.set_title("착색도")
                ax.grid(axis="y", linestyle=":", alpha=0.35)
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                ax.spines["left"].set_linewidth(0.7); ax.spines["bottom"].set_linewidth(0.7)
                ax.legend(); fig.tight_layout()
                st.pyplot(fig, use_container_width=False)
        else:
            st.warning("전년도 과실품질에서 군위 행을 찾지 못했습니다. 잠시 후 다시 시도해 보세요.")

        # 다운로드들
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "⬇️ 올해 실측 월별 CSV",
                df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{region_disp}_{cultivar}_{cur_year}_monthly_measured.csv",
                mime="text/csv",
                disabled=df.empty
            )
        with c2:
            st.download_button(
                "⬇️ 올해 예측에 사용한 월별 CSV",
                filled_this_year.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{region_disp}_{cultivar}_{cur_year}_monthly_used.csv",
                mime="text/csv"
            )
        with c3:
            st.download_button(
                "⬇️ 회귀식 예측 결과 CSV",
                pred_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{region_disp}_{cultivar}_{cur_year}_predictions.csv",
                mime="text/csv"
            )
    else:
        st.info("품종을 선택하고 ‘군위 자동조회 & 예측’ 버튼을 눌러 실행하세요.")

# ==========================================================
# 탭5: 적엽 전후 처리 결과
# ==========================================================

with tab5:
    st.title("🍃 적엽 전후 처리 결과")

    # ------------------------------
    # CSV 결과 표시
    # ------------------------------
    csv_path = Path(__file__).parent / "mba" / "defoliation" / "_out" / "result.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            st.subheader("분석 데이터 (result.csv)")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"CSV 불러오기 실패: {e}")
    else:
        st.warning("result.csv 파일이 없습니다.")

    # ------------------------------
    # Before / After Overlay 이미지 표시
    # ------------------------------
    st.subheader("Before / After Overlay 이미지")

    img_folder = Path(__file__).parent / "mba" / "defoliation" / "_out" / "overlays"
    IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")

    all_imgs = sorted([p for p in img_folder.glob("*") if p.suffix.lower() in IMG_EXT])

    if not all_imgs:
        st.warning("표시할 overlay 이미지가 없습니다.")
    else:
        view = st.radio("보기 방식", ["갤러리(썸네일)", "쌍으로 보기"], horizontal=True, key="view_tab5")

        if view == "갤러리(썸네일)":
            thumbs = st.slider("한 줄에 몇 장", min_value=2, max_value=6, value=4, key="thumbs_slider_tab5")
            rows = (len(all_imgs) + thumbs - 1) // thumbs
            st.caption(f"총 {len(all_imgs)}개 이미지")
            idx = 0
            for _ in range(rows):
                cols = st.columns(thumbs, gap="small")
                for c in cols:
                    if idx >= len(all_imgs):
                        break
                    p = all_imgs[idx]
                    with c:
                        st.image(str(p), caption=p.name)
                    idx += 1

        else:  # 쌍으로 보기 (before/after 자동 매칭)
            names = sorted(set(p.name.replace("_before_overlay.jpg", "").replace("_after_overlay.jpg", "")
                               for p in all_imgs))
            sel = st.selectbox("대상 선택", names)
            before = img_folder / f"{sel}_before_overlay.jpg"
            after = img_folder / f"{sel}_after_overlay.jpg"

            cols = st.columns(2)
            if before.exists():
                cols[0].image(str(before), caption=f"{sel} - Before")
            else:
                cols[0].warning("Before 이미지 없음")

            if after.exists():
                cols[1].image(str(after), caption=f"{sel} - After")
            else:
                cols[1].warning("After 이미지 없음")

# ==========================================================
# 탭6: 과실 이미지 모양·길이 분석 결과
# ==========================================================

with tab6:
    st.title("🍏 과실 이미지 모양·길이 분석 결과")


    # ------------------------------
    # 결과 데이터 불러오기 (엑셀/CSV)
    # ------------------------------
    csv_path = Path(__file__).parent / "python" / "goldenball" / "apple_pairs_results.csv"
    if csv_path.exists():
        try:
            if csv_path.suffix.lower() == ".csv":
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            else:
                df = pd.read_excel(csv_path)
            st.subheader("📊 과실 분석 결과 데이터")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"CSV/XLSX 불러오기 실패: {e}")
    else:
        st.warning("apple_pairs_results 파일이 없습니다. 경로를 확인하세요.")

    # ------------------------------
    # 이미지 뷰어
    # ------------------------------
    st.subheader("📷 분석된 과실 이미지 (Detection 결과)")

    img_folder = Path(__file__).parent / "python" / "goldenball" / "_debug_pairs"
    IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
    all_imgs = sorted([p for p in img_folder.glob("*") if p.suffix.lower() in IMG_EXT])

    if not all_imgs:
        st.warning("표시할 과실 이미지가 없습니다.")
    else:
        view = st.radio("보기 방식", ["갤러리(썸네일)", "단일 선택"], horizontal=True, key="view_tab6")

        if view == "갤러리(썸네일)":
            thumbs = st.slider("한 줄에 몇 장", min_value=2, max_value=6, value=4, key="thumbs_slider_tab6")
            rows = (len(all_imgs) + thumbs - 1) // thumbs
            st.caption(f"총 {len(all_imgs)}개 이미지")
            idx = 0
            for _ in range(rows):
                cols = st.columns(thumbs, gap="small")
                for c in cols:
                    if idx >= len(all_imgs):
                        break
                    p = all_imgs[idx]
                    with c:
                        st.image(str(p), caption=p.name)
                    idx += 1
        else:
            sel = st.selectbox("이미지 선택", [p.name for p in all_imgs])
            path = img_folder / sel
            st.image(str(path), caption=str(path.name))

# ==========================================================
# 탭7: 지역 기반 적합 품종 추천
# ==========================================================

with tab7:
    st.subheader("과실품질 예측 / 적합 품종 추천")

    # 지역 선택 (단순 선택창만, 데이터 없음)
    col_do, col_sigun, col_gueupmyeon = st.columns(3)
    sel_do = col_do.selectbox("도", ["선택", "경상북도", "강원도"], key="demo_do")
    sel_sigun = col_sigun.selectbox("시/군", ["선택", "영주시", "군위군", "평창군"], key="demo_sigun")
    sel_gueupmyeon = col_gueupmyeon.selectbox("구/읍/면", ["선택", "문정동", "군위읍", "대관령면"], key="demo_gueupmyeon")

    st.divider()

    # 품종 (실데이터 없음, 그냥 예시 선택만)
    st.markdown("#### 품종")
    st.info("여기에 품종이 표시됩니다 (데이터 없음).")


    st.divider()

    # 품종특징 / 주의사항 (데이터 없음 → 안내 문구만)
    st.markdown("#### 품종특징")
    st.info("여기에 품종특징이 표시됩니다 (데이터 없음).")

    st.markdown("#### 주의사항")
    st.info("여기에 주의사항이 표시됩니다 (데이터 없음).")

    st.caption("※ 현재는 단순 UI 데모이며, 실제 데이터는 연동되지 않았습니다.")

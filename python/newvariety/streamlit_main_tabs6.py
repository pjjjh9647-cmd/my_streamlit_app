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
# ===== 공통 유틸 & 경로 (탭5·6 전용) =====
import statsmodels.api as sm

# 실제 구현 전까지 임시 더미 — 일단 화면만 보이게
def gw_load_merged_dataset():
    import pandas as pd
    # 실제에선 (merged_all, fruit_agg, env_mwide, bio_y, meta) 반환해야 함
    # meta = (품종컬럼명, 과실숫자타깃들, 공통연도리스트)
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ("cultivar", [], [])

def gw_list_available_cultivars():
    return ["홍로", "후지"]

def gw_load_selected_vars(cv):
    import pandas as pd
    # target별 선택변수 (논문식)
    return pd.DataFrame({"target": ["과중"], "vars_list": [["BIO1", "BIO2"]]})

def gw_fit_ols(g, tgt, xcols):
    import pandas as pd
    # 실제에선 OLS 적합/평가 리턴
    class DummyModel:
        @property
        def model(self):
            class M: 
                exog_names = ["const"] + xcols
            return M()
        @property
        def params(self):
            import pandas as pd
            return pd.Series([0.0] * (1 + len(xcols)), index=["const"] + xcols)
        def predict(self, X):
            return pd.Series([0.0] * len(X))
    sub = pd.DataFrame({"year":[2024], tgt:[0.0], **{c:[0.0] for c in xcols}})
    mets = {"R2": 0.0, "RMSE": 0.0}
    return DummyModel(), sub, pd.Series([0.0]), mets

# 품종별 사용월 (탭5/6에서 필터링에 사용)
GW_MONTH_RANGE = {"홍로": list(range(4,9)), "후지": list(range(4,11))}

# 탭5/6 시각화 도움(크기/톤을 기존 탭과 맞춤: 작고 담백)
def gw_plot_coefficients(model, title=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.set_title(title or "Coefficients")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)

def gw_plot_observed_vs_pred(sub, yhat, tgt, title=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(sub[tgt].values, label="Observed", marker="o")
    ax.plot(yhat.values, label="Pred", marker="o")
    ax.set_title(title or "Observed vs Predicted")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)

def gw_plot_metric_bar(metric, value):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2.5, 1.8))
    ax.bar([metric], [value])
    ax.set_title(metric)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
# ===== /공통 유틸 끝 =====

# 첫번째 탭: 분석결과 (tab1~6)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["분석결과", "과실품질 예측", "분석결과(군위)", "과실품질 예측(군위)", "분석결과(군위·논문식)", "과실품질 예측(군위·논문식)"]
)


with tab1:
    st.title("분석 결과")
    BASE_DIR = Path(r"C:\Users\User\Desktop\mba\분석결과\관계시각화2")
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

    # ▶ 한글 폰트 자동 설정
    import matplotlib
    import matplotlib.font_manager as fm
    import os

    plt.rcParams["font.size"] = 6

    def _set_korean_font():
        bundled_candidates = [
            "fonts/NanumGothic.ttf",
            "fonts/NotoSansKR-Regular.otf",
            "NanumGothic.ttf",
            "NotoSansKR-Regular.otf",
        ]
        for p in bundled_candidates:
            if os.path.exists(p):
                try:
                    fm.fontManager.addfont(p)
                    family = fm.FontProperties(fname=p).get_name()
                    matplotlib.rcParams["font.family"] = family
                    matplotlib.rcParams["axes.unicode_minus"] = False
                    return
                except Exception:
                    pass
        preferred = ["Malgun Gothic", "AppleGothic", "NanumGothic",
                    "Noto Sans CJK KR", "Noto Sans KR", "NanumBarunGothic"]
        sys_fonts = set(f.name for f in fm.fontManager.ttflist)
        for name in preferred:
            if name in sys_fonts:
                matplotlib.rcParams["font.family"] = name
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
        matplotlib.rcParams["axes.unicode_minus"] = False
        st.warning("한글 폰트를 찾지 못했습니다. fonts/ 폴더에 나눔고딕(NanumGothic.ttf) 등을 넣어주세요.")

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
            out.columns = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip()
                        for tup in out.columns.values]

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
    def apply_equation_row6(row: pd.Series, eq_str: str) -> float:
        # 좌변 제거 + 곱점(·)을 *로
        rhs = eq_str.split("=", 1)[1].strip().replace("·", "*")

        # 수식에 등장할 수 있는 모든 변수명을 환경으로 전달
        # (row에 있는 키 전부를 넣어도 문제 없음)
        env = {str(k): (float(row[k]) if pd.notna(row[k]) else float('nan')) for k in row.index}

        # 필요하면 numpy도 전달 가능 (현재 수식엔 필요 없음)
        env["np"] = np

        return float(eval(rhs, {"__builtins__": {}}, env))


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
    BASE_DIR = Path(r"C:\Users\User\Desktop\mba\분석결과\군위")
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
        lhs = eq_str.split("=", 1)[0].strip()
        cols = [c for c in sorted(row.index.tolist(), key=len, reverse=True) if c != lhs]
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
# ==== [탭5·6용 공통 유틸 - 군위 논문식] ====
import statsmodels.api as sm
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# 분석 스크립트와 동일한 베이스 경로 (필요시 한 번만 수정)
GW_BASE = Path(r"C:\Users\User\Desktop\mba\환경데이터")
GW_BIO_FILE   = GW_BASE / "_OUT" / "bioclim_19_variables_Gunwi.csv"
GW_ENV_FILE   = GW_BASE / "기상데이터_통합.xlsx"
GW_FRUIT_FILE = GW_BASE / "과실데이터_통합.xlsx"
GW_OUTDIR     = GW_BASE / "_ANALYSIS_OUT"
GW_BYC_DIR    = GW_OUTDIR / "_BY_CULTIVAR"

# 품종별 허용 월(분석 스크립트와 동일)
GW_MONTH_RANGE = {
    "홍로": list(range(5, 10)),   # 5~9월
    "후지": list(range(5, 11)),   # 5~10월
}

def gw_to_year_series(s: pd.Series) -> pd.Series:
    import re as _re
    s2 = s.copy()
    m_year = s2.astype(str).str.fullmatch(r"\s*(19|20)\d{2}\s*")
    if m_year.fillna(False).all():
        return pd.to_numeric(s2, errors="coerce")
    def _looks_like_yyyymmdd(x):
        try: xs = str(int(float(x))).strip()
        except Exception: xs = str(x)
        return bool(_re.fullmatch(r"(19|20)\d{6}", xs))
    mask_ymd = s2.apply(_looks_like_yyyymmdd)
    if mask_ymd.any():
        tmp = s2[mask_ymd].apply(lambda x: str(int(float(x))).strip())
        y = pd.to_datetime(tmp, format="%Y%m%d", errors="coerce").dt.year
        s2.loc[mask_ymd] = y
    def _looks_like_excel_serial(x):
        try:
            v = float(x); return 10000 <= v <= 60000
        except Exception:
            return False
    mask_serial = s2.apply(_looks_like_excel_serial)
    if mask_serial.any():
        y = pd.to_datetime(s2[mask_serial].astype(float), unit="D", origin="1899-12-30", errors="coerce").dt.year
        s2.loc[mask_serial] = y
    s2 = pd.to_datetime(s2, errors="coerce").dt.year
    return pd.to_numeric(s2, errors="coerce").astype("Int64")

def gw_extract_gunwi(df: pd.DataFrame):
    if "지역명" not in df.columns: return None
    m = df["지역명"].astype(str).str.contains("군위", na=False)
    out = df.loc[m].copy()
    return out if not out.empty else None

def gw_find_cultivar_col(df: pd.DataFrame):
    cands = [c for c in df.columns if any(k in str(c) for k in ["품종","품종명","품종코드","품종구분"])]
    return cands[0] if cands else None

def gw_wide_month(df_sub: pd.DataFrame, cols=("tmean","tmax","tmin","prcp","rad","humid")):
    cols = [c for c in cols if c in df_sub.columns]
    if not cols: return pd.DataFrame()
    w = df_sub.pivot_table(index="year", columns="month", values=cols)
    w.columns = [f"{v}_m{m:02d}" for v, m in w.columns.to_flat_index()]
    return w.reset_index()

@st.cache_data(show_spinner=False)
def gw_load_merged_dataset():
    # 1) 로딩
    bio = pd.read_csv(GW_BIO_FILE)
    env_all   = pd.read_excel(GW_ENV_FILE,   sheet_name=None)
    fruit_all = pd.read_excel(GW_FRUIT_FILE, sheet_name=None)
    env_frames   = [x for x in (gw_extract_gunwi(df) for df in env_all.values())   if x is not None]
    fruit_frames = [x for x in (gw_extract_gunwi(df) for df in fruit_all.values()) if x is not None]
    if not env_frames or not fruit_frames:
        return None, None, None, None, None
    env_raw   = pd.concat(env_frames, ignore_index=True)
    fruit_raw = pd.concat(fruit_frames, ignore_index=True)
    # 2) 월별 집계 → wide
    if "일자" not in env_raw.columns: return None, None, None, None, None
    env_raw["date"]  = pd.to_datetime(env_raw["일자"], errors="coerce")
    env_raw["year"]  = env_raw["date"].dt.year
    env_raw["month"] = env_raw["date"].dt.month
    for col in ["평균기온","최고기온","최저기온","강우량","일사량","습도"]:
        if col in env_raw.columns:
            env_raw[col] = pd.to_numeric(env_raw[col], errors="coerce")
    agg_map = {"평균기온":"mean","최고기온":"mean","최저기온":"mean","강우량":"sum"}
    if "일사량" in env_raw.columns: agg_map["일사량"] = "mean"
    if "습도"  in env_raw.columns: agg_map["습도"]  = "mean"
    env_monthly = (env_raw.dropna(subset=["year","month"])
                          .groupby(["year","month"], as_index=False)
                          .agg(agg_map)
                          .rename(columns={"평균기온":"tmean","최고기온":"tmax","최저기온":"tmin",
                                           "강우량":"prcp","일사량":"rad","습도":"humid"}))
    env_mwide = gw_wide_month(env_monthly)
    # 3) BIO (군위만, 연도 평균)
    bio_gw = bio[bio["region"].astype(str).str.contains("군위", na=False)].copy()
    bio_gw["year"] = pd.to_numeric(bio_gw["year"], errors="coerce").astype("Int64")
    bio_y = bio_gw.groupby("year", as_index=False).mean(numeric_only=True)
    # 4) 과실 (연도×품종)
    if "year" in fruit_raw.columns: fruit_raw["year"] = gw_to_year_series(fruit_raw["year"])
    elif "연도" in fruit_raw.columns: fruit_raw["year"] = gw_to_year_series(fruit_raw["연도"])
    elif "일자" in fruit_raw.columns: fruit_raw["year"] = gw_to_year_series(fruit_raw["일자"])
    else: return None, None, None, None, None
    cultivar_col = gw_find_cultivar_col(fruit_raw)
    if cultivar_col is None:
        fruit_raw["_품종임시"] = "ALL"; cultivar_col = "_품종임시"
    drop_keys = {"지역명","일자","연도","year",cultivar_col}
    fruit_conv = fruit_raw.copy()
    for c in fruit_conv.columns:
        if c not in drop_keys:
            fruit_conv[c] = pd.to_numeric(fruit_conv[c], errors="coerce")
    num_cols = [c for c in fruit_conv.columns if c not in drop_keys and pd.api.types.is_numeric_dtype(fruit_conv[c])]
    fruit_agg = (fruit_conv.dropna(subset=["year"])
                            .groupby([cultivar_col,"year"], as_index=False)[num_cols].mean())
    # 5) 병합 + 교집합 연도
    merged_all = (fruit_agg.merge(bio_y, on="year", how="inner").merge(env_mwide, on="year", how="left"))
    bio_years   = sorted(map(int, pd.Series(bio_y["year"]).dropna().unique()))
    env_years   = sorted(map(int, pd.Series(env_mwide["year"]).dropna().unique()))
    fruit_years = sorted(map(int, pd.Series(fruit_agg["year"]).dropna().unique()))
    common_years = sorted(set(bio_years) & set(env_years) & set(fruit_years))
    return merged_all, fruit_agg, env_mwide, bio_y, (cultivar_col, num_cols, common_years)

def gw_list_available_cultivars():
    if not GW_BYC_DIR.exists(): return []
    return [p.name for p in GW_BYC_DIR.iterdir() if p.is_dir()]

def gw_load_selected_vars(cultivar: str):
    f = GW_BYC_DIR / cultivar / "선택변수_요약.csv"
    if not f.exists():
        return pd.DataFrame(columns=["target","selected_vars","vars_list"])
    df = pd.read_csv(f, encoding="utf-8-sig")
    if "selected_vars" in df.columns:
        df["vars_list"] = df["selected_vars"].fillna("").apply(lambda s: [v.strip() for v in str(s).split(",") if v.strip()])
    else:
        df["vars_list"] = [[] for _ in range(len(df))]
    return df

def gw_fit_ols(df: pd.DataFrame, y_col: str, x_cols: list):
    from math import sqrt as _sqrt
    sub = df[["year", y_col] + x_cols].dropna()
    if len(sub) < 5:
        return None, None, None, None
    X = sm.add_constant(sub[x_cols]); y = sub[y_col]
    model = sm.OLS(y, X).fit()
    yhat = model.predict(X)
    r2 = model.rsquared
    rmse = _sqrt(np.mean((y - yhat)**2))
    return model, sub, yhat, {"R2": r2, "RMSE": rmse}

def gw_plot_coefficients(model, title="Coefficients"):
    betas = model.params.drop(labels=["const"], errors="ignore")
    fig, ax = plt.subplots(figsize=(5,3))
    betas.plot(kind="bar", ax=ax)
    ax.set_title(title); ax.set_ylabel("Coefficient"); ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    st.pyplot(fig)

def gw_plot_observed_vs_pred(sub, yhat, y_col, title="Observed vs Predicted"):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.scatter(sub[y_col], yhat, s=30)
    lims = [min(sub[y_col].min(), yhat.min()), max(sub[y_col].max(), yhat.max())]
    ax.plot(lims, lims, linestyle="--")
    ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
    ax.set_title(title); ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

def gw_plot_metric_bar(name: str, value: float):
    fig, ax = plt.subplots(figsize=(3.5,2.2))
    ax.bar([name], [value])
    ax.set_ylim(0, max(value*1.2, 1e-9))
    for i, v in enumerate([value]): ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_title(name); ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    st.pyplot(fig)

# -------------------------
# TAB 5: 분석결과 요약 (군위·논문식)
# -------------------------
with tab5:
    st.subheader("품종×타깃별 모델 성능 요약 (R², RMSE) — 군위·논문식")

    merged_all, fruit_agg, env_mwide, bio_y, meta = gw_load_merged_dataset()
    if merged_all is None:
        st.error("군위 데이터 병합에 실패했습니다. 원자료 경로/형식을 확인하세요.")
        st.stop()
    cultivar_col, fruit_num_cols, common_years = meta

    cultivars_avail = gw_list_available_cultivars()
    if not cultivars_avail:
        st.warning("분석 결과가 비어 있습니다. 먼저 분석을 수행해 주세요.")
        st.stop()

    rows = []
    for cv in cultivars_avail:
        sel_df = gw_load_selected_vars(cv)
        if sel_df.empty:
            continue
        valid_months = GW_MONTH_RANGE.get(cv, list(range(5,10)))
        month_suffix = tuple(f"m{m:02d}" for m in valid_months)
        if merged_all.empty:
            continue
        g = merged_all.copy() if cultivar_col not in merged_all.columns else \
            merged_all[(merged_all[cultivar_col] == cv)]
        if "year" in g.columns and common_years:
            g = g[g["year"].isin(common_years)]

        for _, r in sel_df.iterrows():
            tgt = r["target"]
            xcols_raw = r["vars_list"] if isinstance(r["vars_list"], (list, tuple)) else []
            xcols = [x for x in xcols_raw if (x in g.columns)]
            if tgt not in g.columns or not xcols:
                continue
            model, sub, yhat, mets = gw_fit_ols(g, tgt, xcols)
            if model is None:
                continue
            rows.append({
                "품종": cv, "타깃": tgt, "변수": ", ".join(xcols),
                "표본수": len(sub), "R²": mets.get("R2", np.nan), "RMSE": mets.get("RMSE", np.nan),
            })

    if not rows:
        st.warning("요약표를 만들 데이터가 없습니다.")
        st.stop()

    perf_df = pd.DataFrame(rows).sort_values(["품종","타깃"]).reset_index(drop=True)
    st.dataframe(perf_df, use_container_width=True)
    st.download_button("⬇️ 요약표 CSV 다운로드",
                       perf_df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="분석결과_요약표(군위_논문식).csv")

    st.markdown("---")
    st.subheader("개별 모델 시각화")
    cv_sel = st.selectbox("품종 선택", options=sorted(perf_df["품종"].unique().tolist()), index=0, key="gw_viz_cv")
    tgt_opts = perf_df.loc[perf_df["품종"]==cv_sel, "타깃"].unique().tolist()
    tgt_sel = st.selectbox("타깃 선택", options=tgt_opts, index=0, key="gw_viz_tgt")
    row = perf_df[(perf_df["품종"]==cv_sel) & (perf_df["타깃"]==tgt_sel)].iloc[0]
    xcols = [c.strip() for c in str(row["변수"]).split(",") if c.strip()]

    g = merged_all.copy() if cultivar_col not in merged_all.columns else \
        merged_all[(merged_all[cultivar_col] == cv_sel)]
    if "year" in g.columns and common_years:
        g = g[g["year"].isin(common_years)]

    model, sub, yhat, mets = gw_fit_ols(g, tgt_sel, xcols)
    if model is None:
        st.info("표본이 부족하거나 입력 변수가 없습니다.")
    else:
        colA, colB = st.columns(2)
        with colA:
            gw_plot_coefficients(model, title=f"[{cv_sel} - {tgt_sel}] Coefficients")
        with colB:
            gw_plot_observed_vs_pred(sub, yhat, tgt_sel, title=f"[{cv_sel} - {tgt_sel}] Observed vs Predicted")
        colC, colD = st.columns(2)
        with colC:
            gw_plot_metric_bar("R²", mets.get("R2", np.nan))
        with colD:
            gw_plot_metric_bar("RMSE", mets.get("RMSE", np.nan))

# -------------------------
# TAB 6: 2024 과실품질 예측 (논문식 회귀식 사용, 로컬 파일만)
# -------------------------
with tab6:
    st.subheader("과실품질 예측 — 2024 예측 vs 2023 실제 (논문식 회귀식)")

    # --- 회귀식(사용자 제공) ---
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

    # --- 설정/경로 확인 ---
    if not GW_ENV_FILE.exists() or not GW_FRUIT_FILE.exists():
        st.error(f"파일을 찾을 수 없습니다.\n환경: {GW_ENV_FILE}\n과실: {GW_FRUIT_FILE}")
        st.stop()

    # --- 유틸: 월범위(홍로 4~8, 후지 4~10) ---
    def _month_window(cv: str):
        return (4, 8) if cv == "홍로" else (4, 10)

    # --- 유틸: 패턴으로 실제 과실 컬럼 찾기(2023 비교용) ---
    def _first_col_by_pattern(df: pd.DataFrame, pattern: str):
        pat = re.compile(pattern, flags=re.IGNORECASE)
        for c in df.columns:
            if pat.search(str(c)): return c
        return None

    # 과실 지표 정규화용 패턴(가능한 별칭 포함)
    TARGET_PATTERNS = {
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

    # --- 데이터 로드 ---
    env_all   = pd.read_excel(GW_ENV_FILE,   sheet_name=None)
    fruit_all = pd.read_excel(GW_FRUIT_FILE, sheet_name=None)

    # 지역 목록(있으면 선택)
    def _collect_regions(dfs):
        regs = []
        for df in dfs.values():
            if "지역명" in df.columns:
                regs += df["지역명"].dropna().astype(str).unique().tolist()
        return sorted(list(dict.fromkeys(regs)))  # unique + keep order

    region_opts = _collect_regions(env_all) or ["(전체)"]
    col1, col2 = st.columns([1,1])
    with col1:
        cultivar = st.radio("품종 선택", ["홍로","후지"], horizontal=True, key="tab6_cv")
    with col2:
        region = st.selectbox("지역 선택(환경/과실에서 공통으로 필터링)", options=region_opts, index=(region_opts.index("대구군위") if "대구군위" in region_opts else 0))

    # 지역 필터 함수(컬럼 없으면 전체)
    def _filter_region(df: pd.DataFrame) -> pd.DataFrame:
        if "지역명" in df.columns and region in region_opts and region != "(전체)":
            m = df["지역명"].astype(str) == str(region)
            return df.loc[m].copy()
        return df.copy()

    # --- 환경: 일→연/월, 월별 집계 (PATCH) ---
    env_frames = [_filter_region(df) for df in env_all.values()]
    env_raw = pd.concat(env_frames, ignore_index=True)

    if "일자" not in env_raw.columns:
        st.error("환경데이터에 '일자' 컬럼이 필요합니다.")
        st.stop()

    env_raw["date"]  = pd.to_datetime(env_raw["일자"], errors="coerce")
    env_raw["연도"]  = env_raw["date"].dt.year
    env_raw["월"]   = env_raw["date"].dt.month

    # 회귀식에 등장 가능한 원시 변수들
    base_mean_vars = ["평균기온","최고기온","최저기온","습도","평균풍속","최대풍속","결로시간"]
    base_sum_vars  = ["강우량","일사량","결로시간"]  # sum이 필요한 애들(결로시간은 mean/sum 둘 다 수식에 있을 수 있음)

    # 실제로 있는 컬럼만 쓰되, 나중에 없는 것은 0/NaN 으로 보정
    present_mean = [c for c in base_mean_vars if c in env_raw.columns]
    present_sum  = [c for c in base_sum_vars  if c in env_raw.columns]

    for c in present_mean + present_sum:
        env_raw[c] = pd.to_numeric(env_raw[c], errors="coerce")

    # 월별 mean
    env_mean = (env_raw.dropna(subset=["연도","월"])
                    .groupby(["연도","월"], as_index=False)[present_mean].mean()
                    .rename(columns={c: f"{c}_mean" for c in present_mean}))

    # 월별 sum
    env_sum  = (env_raw.dropna(subset=["연도","월"])
                    .groupby(["연도","월"], as_index=False)[present_sum].sum()
                    .rename(columns={c: f"{c}_sum" for c in present_sum}))

    # 합치기
    env_month = pd.merge(env_mean, env_sum, on=["연도","월"], how="outer").sort_values(["연도","월"]).reset_index(drop=True)

   # mean/sum 컬럼명 분리 정리
    for c in num_cols_env:
        if c in env_month.columns and (c, "mean") in env_month.columns.to_flat_index() if isinstance(env_month.columns, pd.MultiIndex) else False:
            pass  # (멀티인덱스일 가능성 방지용)
    # 멀티인덱스 대비
    if isinstance(env_month.columns, pd.MultiIndex):
        env_month.columns = ["연도","월"] + [f"{v}_{agg}" for v, agg in env_month.columns.tolist()[2:]]

    # wide: _mean_mMM / _sum_mMM 모두 생성
    def _wide_mean_sum(env_m: pd.DataFrame) -> pd.DataFrame:
        # mean 계열
        wide_mean = None
        for m in range(1, 13):
            sub = env_m[env_m["월"] == m][["연도"] + [f"{c}_mean" for c in num_cols_env if f"{c}_mean" in env_m.columns]].copy()
            sub = sub.rename(columns={f"{c}_mean": f"{c}_mean_m{m:02d}" for c in num_cols_env if f"{c}_mean" in env_m.columns})
            wide_mean = sub if wide_mean is None else pd.merge(wide_mean, sub, on="연도", how="outer")
        # sum 계열
        wide_sum = None
        for m in range(1, 13):
            sub = env_m[env_m["월"] == m][["연도"] + [f"{c}_sum" for c in num_cols_env if f"{c}_sum" in env_m.columns]].copy()
            sub = sub.rename(columns={f"{c}_sum": f"{c}_sum_m{m:02d}" for c in num_cols_env if f"{c}_sum" in env_m.columns})
            wide_sum = sub if wide_sum is None else pd.merge(wide_sum, sub, on="연도", how="outer")
        out = pd.merge(wide_mean, wide_sum, on="연도", how="outer")
        return out

    # 2024 채움: 월범위(품종별) 내에서 현재 데이터 없으면 과거 평균으로 대체
    def _fill_year_climo(env_m: pd.DataFrame, year_val: int, cv: str) -> pd.DataFrame:
        s_mon, e_mon = _month_window(cv)
        base = pd.DataFrame({"연도":[year_val]*12, "월":list(range(1,13))})
        cur  = env_m[env_m["연도"] == year_val]
        hist = env_m[env_m["연도"] <  year_val]
        out = base.merge(cur, on=["연도","월"], how="left")
        if not hist.empty:
            # 과거 전체 평균 사용
            cl_mean = hist.groupby("월", as_index=False)[[f"{c}_mean" for c in num_cols_env if f"{c}_mean" in env_m.columns]].mean()
            cl_sum  = hist.groupby("월", as_index=False)[[f"{c}_sum"  for c in num_cols_env if f"{c}_sum"  in env_m.columns]].mean()
            out = out.merge(cl_mean, on="월", how="left", suffixes=("","_clmo"))
            out = out.merge(cl_sum,  on="월", how="left", suffixes=("","_clmo2"))
            # 비어있으면 기후평년으로 채움
            for c in [f"{v}_mean" for v in num_cols_env if f"{v}_mean" in env_m.columns]:
                out[c] = np.where(out[c].notna(), out[c], out.get(f"{c}_clmo"))
            for c in [f"{v}_sum" for v in num_cols_env if f"{v}_sum in env_m.columns"]:
                pass  # 안전
            for v in num_cols_env:
                if f"{v}_sum" in env_m.columns and f"{v}_sum_clmo2" in out.columns:
                    c = f"{v}_sum"
                    out[c] = np.where(out[c].notna(), out[c], out[f"{c}_clmo2"])
            # 불필요한 *_clmo, *_clmo2 제거
            out = out[[col for col in out.columns if not (str(col).endswith("_clmo") or str(col).endswith("_clmo2"))]]
        # 품종 월범위 자르기(시각화/안전)
        out = out[(out["월"] >= s_mon) & (out["월"] <= e_mon)].copy()
        # 다시 12개월 형태로 만들기 위해 누락월은 climo로 채운 full 12개월 버전도 생성
        full = base.merge(out, on=["연도","월"], how="left")
        if not hist.empty:
            # 남은 월도 기후평년으로
            for v in num_cols_env:
                if f"{v}_mean" in env_m.columns:
                    c = f"{v}_mean"
                    cl = hist.groupby("월", as_index=False)[c].mean().rename(columns={c:"_cl"})
                    full = full.merge(cl, on="월", how="left")
                    full[c] = np.where(full[c].notna(), full[c], full["_cl"]); full = full.drop(columns=["_cl"])
                if f"{v}_sum" in env_m.columns:
                    c = f"{v}_sum"
                    cl = hist.groupby("월", as_index=False)[c].mean().rename(columns={c:"_cl"})
                    full = full.merge(cl, on="월", how="left")
                    full[c] = np.where(full[c].notna(), full[c], full["_cl"]); full = full.drop(columns=["_cl"])
        return full

    # 집계 테이블 컬럼 정리: 멀티인덱스 방지 + mean/sum 접미사 부여
    # (이미 위에서 mean/sum 형태로 나왔을 가능성 고려하여 보정)
    if not any(k.endswith("_mean") or k.endswith("_sum") for k in env_month.columns if k not in ["연도","월"]):
        # env_month가 '평균기온','평균기온1'(sum) 같이 안 들어왔다면 다시 분리
        cols_keep = ["연도","월"]
        # 원본에서 다시 mean/sum 나눠 생성
        tmp = env_raw.groupby(["연도","월"], as_index=False).agg({**{c:"mean" for c in num_cols_env}, **{c:"sum" for c in num_cols_env}})
        if isinstance(tmp.columns, pd.MultiIndex):
            tmp.columns = ["연도","월"] + [f"{v}_{agg}" for v, agg in tmp.columns.tolist()[2:]]
        env_month = tmp

    # 2024용 월별(클리모 채움) → wide 생성
    env2024_m = _fill_year_climo(env_month, 2024, cultivar)
    st.caption("예측에 사용된 2024 월별 집계(없는 월은 과거 평균으로 대체)")
    st.dataframe(env2024_m, use_container_width=True)

    # wide 생성(연도 2024 하나)
    env2024_m["연도"] = 2024
    wide_all = _wide_mean_sum(env2024_m)
    row2024 = wide_all[wide_all["연도"] == 2024].iloc[0]

    # --- 회귀식 적용 ---
    def apply_equation_row(row: pd.Series, eq_str: str) -> float:
        rhs = eq_str.split("=",1)[1].strip().replace("·","*")
        # 긴 이름부터 치환(부분 문자열 충돌 방지)
        for c in sorted(row.index.tolist(), key=len, reverse=True):
            rhs = rhs.replace(c, f"row[{repr(c)}]")
        return float(eval(rhs, {"__builtins__": {}}, {"row": row, "np": np}))

    EQU = EQUATIONS_BY_CULTIVAR[cultivar]
    preds = {}
    for tgt, formula in EQU.items():
        try:
            preds[tgt] = apply_equation_row(row2024, formula)
        except Exception as e:
            preds[tgt] = np.nan
            st.warning(f"[{tgt}] 수식 적용 오류: {e}")

    pred_df = pd.DataFrame([preds]).T.reset_index()
    pred_df.columns = ["항목","2024 예측"]

    # --- 2023 실제값(동일 지역·품종 평균) ---
    fruit_frames = [_filter_region(df) for df in fruit_all.values()]
    fruit_raw = pd.concat(fruit_frames, ignore_index=True)

    # 연도 파싱
    if "year" in fruit_raw.columns:
        fruit_raw["year"] = gw_to_year_series(fruit_raw["year"])
    elif "연도" in fruit_raw.columns:
        fruit_raw["year"] = gw_to_year_series(fruit_raw["연도"])
    elif "일자" in fruit_raw.columns:
        fruit_raw["year"] = gw_to_year_series(fruit_raw["일자"])
    else:
        fruit_raw["year"] = pd.NA

    cult_col = gw_find_cultivar_col(fruit_raw) or "_품종임시"
    if cult_col not in fruit_raw.columns:
        fruit_raw[cult_col] = "ALL"

    # 2023 행만, 선택 품종 필터
    f23 = fruit_raw[(fruit_raw["year"] == 2023) & (fruit_raw[cult_col] == cultivar)].copy()

    actual_rows = []
    for tgt, pat in TARGET_PATTERNS.items():
        col = _first_col_by_pattern(f23, pat)
        if col is not None:
            val = pd.to_numeric(f23[col], errors="coerce").mean()
        else:
            val = np.nan
        actual_rows.append([tgt, val])
    actual_df = pd.DataFrame(actual_rows, columns=["항목","2023 실제"])

    # --- 합치기 & 표시 ---
    out = pred_df.merge(actual_df, on="항목", how="left")
    st.subheader(f"{region} · {cultivar} — 2024 예측 vs 2023 실제")
    st.dataframe(out.set_index("항목").T, use_container_width=True)

    # --- 소형 그래프(같은 스타일) ---
    def _pick(df, item):
        r = df[df["항목"] == item]
        if r.empty: return np.nan, np.nan
        return pd.to_numeric(r["2024 예측"], errors="coerce").iloc[0], pd.to_numeric(r["2023 실제"], errors="coerce").iloc[0]

    PRED_COLOR = "#87CEEB"
    LAST_COLOR = "#800080"

    # 1) 과중/횡경/종경
    size_items = [i for i in ["과중","횡경","종경"] if i in out["항목"].values]
    if size_items:
        x = np.arange(len(size_items)); y_pred=[]; y_last=[]
        for it in size_items:
            p,l=_pick(out,it); y_pred.append(p); y_last.append(l)
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        w=0.35
        ax.bar(x-w/2, np.nan_to_num(y_pred,nan=0.0), width=w, label="2024 예측", color=PRED_COLOR)
        ax.bar(x+w/2, np.nan_to_num(y_last,nan=0.0), width=w, label="2023 실제", color=LAST_COLOR)
        ax.set_xticks(x); ax.set_xticklabels(size_items); ax.set_title("과실 크기")
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(); fig.tight_layout(); st.pyplot(fig, use_container_width=False)

    # 2) 경도/당도/산도
    def _bar_single(item, ylim_top=None):
        p,l = _pick(out,item)
        if pd.isna(p) and pd.isna(l): return
        xs=[]; ys=[]; cs=[]
        if not pd.isna(p): xs.append("2024 예측"); ys.append(float(p)); cs.append(PRED_COLOR)
        if not pd.isna(l): xs.append("2023 실제"); ys.append(float(l)); cs.append(LAST_COLOR)
        fig, ax = plt.subplots(figsize=(2.5,1.8))
        ax.bar(np.arange(len(xs)), ys, tick_label=xs, color=cs, width=0.35)
        ax.set_title(item); ax.grid(axis="y", linestyle=":", alpha=0.35)
        if ylim_top: ax.set_ylim(top=ylim_top)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout(); st.pyplot(fig, use_container_width=False)

    c1, c2, c3 = st.columns(3)
    with c1: _bar_single("경도", ylim_top=70)
    with c2: _bar_single("당도")
    with c3: _bar_single("산도")

    # 3) L/a/b
    tone_items = [i for i in ["L","a","b"] if i in out["항목"].values]
    if tone_items:
        x = np.arange(len(tone_items)); yp=[]; yl=[]
        for it in tone_items:
            p,l=_pick(out,it); yp.append(p); yl.append(l)
        fig, ax = plt.subplots(figsize=(3.5,2.6))
        if any(pd.notna(yp)): ax.plot(x, yp, marker="o", linewidth=2, label="2024 예측", color=PRED_COLOR)
        if any(pd.notna(yl)): ax.plot(x, yl, marker="o", linewidth=2, label="2023 실제", color=LAST_COLOR)
        ax.set_xticks(x); ax.set_xticklabels(tone_items); ax.set_title("착색도")
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(); fig.tight_layout(); st.pyplot(fig, use_container_width=False)

    # --- 다운로드 ---
    cA, cB = st.columns(2)
    with cA:
        st.download_button("⬇️ 예측/실제 비교표 CSV",
            out.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{region}_{cultivar}_2024_pred_vs_2023_actual(eq).csv",
            mime="text/csv")
    with cB:
        # 2024 예측에 사용된 wide 변수 전체 내보내기
        vars_df = pd.DataFrame([row2024]).T.reset_index()
        vars_df.columns = ["변수","값"]
        st.download_button("⬇️ 2024 입력변수(BUILT) CSV",
            vars_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{region}_{cultivar}_2024_env_wide_mean_sum.csv",
            mime="text/csv")


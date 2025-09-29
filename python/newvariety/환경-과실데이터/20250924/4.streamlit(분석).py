# -*- coding: utf-8 -*-
import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="분석결과 뷰어", layout="wide")
st.title("분석 결과")

# 고정 경로
BASE_DIR = Path(r"C:\Users\User\Desktop\mba\분석결과\관계시각화2")
CULTIVAR_DIRS = {
    "홍로": BASE_DIR / "홍로",
    "후지": BASE_DIR / "후지",
}

# 1) 상단 진단 정보
''

cultivar = st.radio("품종 선택", list(CULTIVAR_DIRS.keys()), horizontal=True)
folder = CULTIVAR_DIRS[cultivar]

''

if not folder.exists():
    st.error("해당 폴더가 없습니다. 경로 오타 또는 드라이브 접근 권한을 확인하세요.")
    st.stop()

# 2) 파일 수집: 하위 폴더까지 검색
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
TAB_EXT = (".csv", ".xlsx")

all_imgs = sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT])
all_tabs = sorted([p for p in folder.rglob("*") if p.suffix.lower() in TAB_EXT])

    # 진단: 발견된 파일 개수 박스 제거

# 3) 표시 모드 선택
mode = st.segmented_control("표시 유형", options=["이미지", "표(데이터)"])

if mode == "이미지":
    if not all_imgs:
        st.warning("표시할 이미지가 없습니다.")
    else:
        view = st.radio("방식 선택", ["갤러리(썸네일)", "단일 파일"], horizontal=True)

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
        # 파일 선택 UI 숨기고, 첫 번째 파일 자동 선택
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

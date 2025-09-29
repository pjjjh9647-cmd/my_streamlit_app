import pandas as pd
import glob
import re
from pathlib import Path

# 설정
FOLDER = r"C:\Users\User\Desktop\환경데이터"
OUTFILE = Path(FOLDER) / "기상데이터_통합.xlsx"

def extract_region(df):
    # 셀 어딘가에 '지역명'이 들어있는 텍스트를 찾아서 콜론 뒤를 추출
    for row in df.itertuples(index=False):
        for cell in row:
            if isinstance(cell, str) and "지역명" in cell:
                # 예: '지역명 : 경남거창' 또는 '지역명:경남거창'
                txt = cell
                # 콜론 기준으로 split
                parts = re.split(r"[:：]\s*", txt)
                if len(parts) >= 2 and parts[1]:
                    name = parts[1].strip()
                    return name[-2:]  # 뒤 2글자만
            # '지역명'과 값이 따로 셀에 있을 수도 있음: ['지역명', '경남거창']
            # 이를 대비해서 '지역명' 바로 오른쪽 셀을 지역명으로 처리
    # 위에서 못 찾았다면, 행 단위로 탐색
    for r in range(len(df)):
        row = df.iloc[r].astype(str).tolist()
        for c, v in enumerate(row):
            if "지역명" in v:
                # 오른쪽 셀 시도
                if c + 1 < len(row):
                    name = str(df.iat[r, c+1]).strip()
                    if name and name != "nan":
                        return name[-2:]
    return None

def find_data_start(df):
    # 첫 번째 컬럼에서 YYYY-MM-DD 형태가 처음 나오는 행을 데이터 시작으로 간주
    for i in range(len(df)):
        val = df.iat[i, 0]
        if pd.api.types.is_number(val):
            # 엑셀 일자 숫자일 수 있음
            return i
        if isinstance(val, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", val.strip()):
            return i
    return None

def read_one(filepath):
    raw = pd.read_excel(filepath, sheet_name=0, header=None, engine="openpyxl")
    region = extract_region(raw) or ""

    start = find_data_start(raw)
    if start is None:
        raise ValueError(f"데이터 시작 행을 찾지 못함: {filepath}")

    data = raw.iloc[start:, :11].copy()  # 날짜 포함 11개 열
    # 열 이름 강제 지정
    data.columns = [
        "일자","평균기온","표준편차","최고기온","최저기온",
        "습도","강우량","일사량","결로시간","평균풍속","최대풍속"
    ]

    # 필요한 열만
    data = data[["일자","평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]]

    # 날짜 변환
    data["일자"] = pd.to_datetime(data["일자"], errors="coerce").dt.date

    # 숫자 변환
    for col in ["평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # 지역명 추가
    data["지역명"] = region[-2:] if region else ""

    # 유효한 날짜만 남김
    data = data.dropna(subset=["일자"])

    # 컬럼 순서
    data = data[["지역명","일자","평균기온","최고기온","최저기온","습도","강우량","일사량","결로시간","평균풍속","최대풍속"]]
    return data

def main():
    files = sorted(
        glob.glob(str(Path(FOLDER) / "*.xlsx"))
        + glob.glob(str(Path(FOLDER) / "*.xls"))
    )
    if not files:
        raise FileNotFoundError("폴더에 엑셀 파일이 없습니다")

    dfs = []
    for f in files:
        try:
            dfs.append(read_one(f))
        except Exception as e:
            print(f"건너뜀: {f}  이유: {e}")

    if not dfs:
        raise RuntimeError("읽어들일 수 있는 파일이 없습니다")

    final_df = pd.concat(dfs, ignore_index=True)

    # 지역명과 일자 기준 정렬
    final_df = final_df.sort_values(["지역명","일자"]).reset_index(drop=True)

    # 저장
    final_df.to_excel(OUTFILE, index=False)
    print(f"완료: {OUTFILE}")

if __name__ == "__main__":
    main()

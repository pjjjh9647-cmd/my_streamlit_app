import pandas as pd
from pathlib import Path

FOLDER = Path(r"C:\Users\User\Desktop\과실데이터")
OUTFILE = FOLDER / "과실데이터_통합.xlsx"

def get_cultivar_from_filename(path: Path) -> str:
    # 확장자 제외한 파일명에서 첫 단어(공백/언더바 전까지)를 품종으로 사용
    name = path.stem
    # 필요하다면 '_' 기준 split
    cultivar = name.split("_")[0]
    return cultivar

def load_one(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)  # 첫 행을 열 이름으로 사용한다고 가정
    cultivar = get_cultivar_from_filename(path)
    df.insert(0, "품종", cultivar)   # 맨 앞 열에 추가
    return df

def main():
    files = sorted(list(FOLDER.glob("*.xlsx")) + list(FOLDER.glob("*.xls")))
    if not files:
        raise FileNotFoundError(f"엑셀 파일이 없습니다: {FOLDER}")

    dfs = [load_one(p) for p in files]
    final = pd.concat(dfs, ignore_index=True)

    # 결과 저장
    final.to_excel(OUTFILE, index=False)
    print(f"완료: {OUTFILE}")

if __name__ == "__main__":
    main()

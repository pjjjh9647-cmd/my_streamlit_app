import matplotlib.pyplot as plt
import matplotlib as mpl

# 한글 폰트 설정
mpl.rc('font', family='Malgun Gothic')

# 마이너스(-) 기호 깨짐 방지
mpl.rcParams['axes.unicode_minus'] = False


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# 엑셀 파일 경로 설정
base_path = r"C:\Users\User\Desktop\일\주산지현장연구"
file_fruit = os.path.join(base_path, "과실.xlsx")
file_env = os.path.join(base_path, "환경.xlsx")

# 데이터 불러오기
df_fruit = pd.read_excel(file_fruit)
df_env = pd.read_excel(file_env)

# 병합 (지역 기준)
df = pd.merge(df_fruit, df_env, on="지역", how="left")

# 과실 특성 변수 목록
fruit_vars = ['과중', '종경', '횡경', 'DA-meter', 'Hunter_L', 'Hunter_a', 'Hunter_b',
              '경도', '당도', '산도', '과경길이', '동녹']

# 환경 변수 목록 (자동 인식)
env_vars = [col for col in df.columns if col.startswith(('기온', '풍향', '풍속', '습도', '강수량', '일사량'))]

# 상관관계 분석
corr_matrix = df[fruit_vars + env_vars].corr().loc[fruit_vars, env_vars]

# 히트맵 시각화
plt.figure(figsize=(20, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("과실 특성과 환경 변수 간 Pearson 상관관계")
plt.tight_layout()
plt.show()

# 회귀분석 예시: 당도 ~ 기온_5월 + 일사량_5월
if '기온_5월' in df.columns and '일사량_5월' in df.columns:
    X = df[['기온_5월', '일사량_5월']]
    X = sm.add_constant(X)
    y = df['당도']

    model = sm.OLS(y, X).fit()
    print("\n[회귀분석 결과] 당도 ~ 기온_5월 + 일사량_5월")
    print(model.summary())

    # 산점도 + 추세선
    sns.lmplot(x='일사량_5월', y='당도', data=df, height=5, aspect=1.5)
    plt.title("일사량(5월) vs 당도")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ 회귀분석용 '기온_5월' 또는 '일사량_5월' 열이 없습니다.")

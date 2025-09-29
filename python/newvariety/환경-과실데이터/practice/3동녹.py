import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 파일 경로 설정
env_path = r'C:\Users\User\Desktop\일\주산지현장연구\환경.xlsx'
fruit_path = r'C:\Users\User\Desktop\일\주산지현장연구\과실.xlsx'

# 데이터 불러오기
df_env = pd.read_excel(env_path)
df_fruit = pd.read_excel(fruit_path)

# 병합
df = pd.merge(df_env, df_fruit, on='지역')

# 환경 변수 추출: 기온/습도만 추출
env_cols = [col for col in df.columns if col.startswith('기온') or col.startswith('습도')]
X = df[env_cols]

# 종속변수
y = df['동녹']

# 결측치 제거
data = pd.concat([X, y], axis=1).dropna()
X = data[env_cols]
y = data['동녹']

# 숫자형 변환
X = X.astype(float)
y = y.astype(float)

# 회귀 모델 생성
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# 회귀 결과 출력
print(model.summary())

# 예측값 vs 실제값 그래프
plt.figure(figsize=(6, 4))
sns.regplot(x=model.fittedvalues, y=y, line_kws={'color': 'red'})
plt.xlabel('예측 동녹')
plt.ylabel('실제 동녹')
plt.title('예측값 vs 실제값')
plt.tight_layout()
plt.show()

# 잔차 플롯
residuals = model.resid
plt.figure(figsize=(6, 4))
sns.residplot(x=model.fittedvalues, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.xlabel('예측값')
plt.ylabel('잔차')
plt.title('잔차 플롯')
plt.tight_layout()
plt.show()

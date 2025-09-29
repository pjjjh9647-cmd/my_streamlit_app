import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 엑셀 파일 경로 설정
env_path = r'C:\Users\User\Desktop\일\주산지현장연구\환경.xlsx'
fruit_path = r'C:\Users\User\Desktop\일\주산지현장연구\과실.xlsx'

# 엑셀 불러오기
df_env = pd.read_excel(env_path)
df_fruit = pd.read_excel(fruit_path)

# 병합 (지역 기준)
df = pd.merge(df_env, df_fruit, on='지역')

# 사용 변수 지정
X = df[['기온_4월', '습도_4월']]  # 독립변수
y = df['동녹']  # 종속변수

# 타입 변환 (object → float)
X = X.astype(float)
y = y.astype(float)

# 회귀 분석 모델 구축
X = sm.add_constant(X)  # 상수항 추가
model = sm.OLS(y, X).fit()

# 결과 출력
print(model.summary())

# 잔차 플롯 (선택)
residuals = model.resid
fitted = model.fittedvalues

plt.figure(figsize=(6,4))
sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.xlabel('예측값')
plt.ylabel('잔차')
plt.title('잔차 플롯')
plt.tight_layout()
plt.show()

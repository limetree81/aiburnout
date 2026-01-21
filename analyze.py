#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("ai_workplace_productivity_burnout.csv")
df
# %%
df.columns

['User_ID', 'Day', 'Experience_Level', 'AI_Tool_Usage_%', 'Hours_Coded',
       'Code_Complexity_Score', 'Bugs_Introduced', 'Tasks_Completed',
       'Stress_Level_1_10', 'Current_Burnout_Score', 'Burnout_Risk']

#%%
des = df[['Experience_Level', 'Burnout_Risk']].groupby('Experience_Level').describe()
risk_by_ratio = df[['Experience_Level', 'Burnout_Risk']].groupby('Experience_Level').value_counts()/des[('Burnout_Risk','count')]
# %%
risk_for_level = risk_by_ratio.unstack(level=-1)

#%%
# 1. 상관계수 계산
corr_matrix = df.groupby('Experience_Level').corr(numeric_only=True)

# 2. 히트맵 시각화 설정
plt.figure(figsize=(10, 8)) # 그래프 크기 설정
sns.heatmap(corr_matrix, 
            annot=False,      # 각 칸에 숫자 표시
            fmt=".2f",       # 소수점 자리수 설정
            cmap='coolwarm', # 색상 팔레트 (양의 상관관계는 빨강, 음의 상관관계는 파랑)
            linewidths=0.5,  # 칸 사이의 간격
            square=True)     # 각 칸을 정사각형으로 설정

plt.title('Correlation Matrix Heatmap', fontsize=15)
plt.tight_layout()

# 3. 결과 저장 또는 출력
plt.savefig('correlation_heatmap.png') # 이미지 파일로 저장
plt.show()
# %%
# 1. 설정값 (문턱값)
threshold = 0.5

# 2. groupby corr 수행
corr_matrix = df.groupby('Experience_Level').corr(numeric_only=True)

# 3. 데이터 정제 및 필터링
# stack(): 행렬을 (Level, Feature1, Feature2) 형태의 시리즈로 펼침
# reset_index(): 인덱스를 컬럼으로 변환
relevant_corr = corr_matrix.stack().reset_index()

# 컬럼명 변경 (기본값: level_0, level_1, level_2, 0)
relevant_corr.columns = ['Experience_Level', 'Feature_A', 'Feature_B', 'Correlation']

# 4. 필터링 조건 적용
# - 자기 자신과의 상관관계(1.0) 제외
# - 절댓값이 threshold보다 큰 경우만 추출
filtered_list = relevant_corr[
    (relevant_corr['Feature_A'] != relevant_corr['Feature_B']) & 
    (relevant_corr['Correlation'].abs() > threshold)
]

# Feature_A 이름이 Feature_B보다 사전순으로 앞에 있는 것만 남기면 중복이 제거됨
unique_pairs = filtered_list[filtered_list['Feature_A'] < filtered_list['Feature_B']]

# 5. 결과 확인
print(filtered_list.sort_values(by=['Experience_Level', 'Correlation'], ascending=False))


#%%
# 1. 상관계수 계산 (MultiIndex 형태)
corr_matrix_grouped = df.groupby('Experience_Level').corr(numeric_only=True)

# 2. 멀티인덱스를 컬럼으로 펼치기
stacked_corr = corr_matrix_grouped.stack().reset_index()
stacked_corr.columns = ['Experience_Level', 'Feature_A', 'Feature_B', 'Correlation']

# 3. 자기 자신과의 상관관계(1.0) 및 중복 쌍 제거 (A-B, B-A 중복)
# Feature_A와 Feature_B를 정렬하여 중복 제거용 키 생성
filtered_corr = stacked_corr[stacked_corr['Feature_A'] < stacked_corr['Feature_B']].copy()

# 4. 피벗 테이블 생성: 변수 쌍을 인덱스로, 레벨을 컬럼으로
comparison_table = filtered_corr.pivot(
    index=['Feature_A', 'Feature_B'], 
    columns='Experience_Level', 
    values='Correlation'
)

# 5. 보기 좋게 출력
print("--- [Experience_Level별 상관계수 비교 테이블] ---")
display(comparison_table)

#%%


# 1. 설정값 (문턱값)
threshold = 0.5

# 2. groupby corr 수행
corr_matrix = df.groupby('Experience_Level').corr(numeric_only=True)

# 3. 데이터 정제
relevant_corr = corr_matrix.stack().reset_index()
relevant_corr.columns = ['Experience_Level', 'Feature_A', 'Feature_B', 'Correlation']

# 4. [추가] 레벨 간 분산 계산
# 모든 레벨의 상관관계 데이터를 바탕으로 피처 쌍별 분산을 구합니다.
variance_df = relevant_corr.groupby(['Feature_A', 'Feature_B'])['Correlation'].var().reset_index()
variance_df['Experience_Level'] = 'VARIANCE' # 레벨 이름을 VARIANCE로 설정

# 5. 기존 데이터와 분산 데이터 합치기
# 상관관계 데이터 끝에 분산 행들을 붙입니다.
combined_df = pd.concat([relevant_corr, variance_df], ignore_index=True)

# 6. 필터링 조건 적용
# - 자기 자신과의 상관관계 제외
# - 피처 쌍 중복 제거 (A < B)
# - (선택 사항) 상관관계가 threshold보다 큰 행들 위주로 필터링
#   단, 분산 행은 threshold에 상관없이 보고 싶을 수 있으므로 조건을 조정합니다.
filtered_list = combined_df[combined_df['Feature_A'] < combined_df['Feature_B']]

# threshold 필터링: 원본 상관관계가 기준치 이상인 것만 보거나, 
# 혹은 특정 피처 쌍의 레벨별 상관관계들만 모아서 출력
# 여기서는 원본 logic을 유지하여 필터링합니다.
final_result = filtered_list[
    (filtered_list['Experience_Level'] == 'VARIANCE') | 
    (filtered_list['Correlation'].abs() > threshold)
]

# 7. 결과 정렬 (피처 쌍별로 모아서 보기 위해 Feature_A, Feature_B 순으로 정렬)
final_result = final_result.sort_values(by=['Feature_A', 'Feature_B', 'Experience_Level'])

# 8. 결과 확인
#print(final_result)
final_result.to_csv("corr.csv")

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 시각화 스타일 설정
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans' # 환경에 맞는 폰트 설정

# 2. 분석할 지표 리스트 (컬럼명)
metrics = ['AI_Tool_Usage_%', 'Hours_Coded',
       'Code_Complexity_Score', 'Bugs_Introduced', 'Tasks_Completed',
       'Stress_Level_1_10', 'Current_Burnout_Score']
fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 5 * len(metrics)), sharex=True)

# 3. 각 지표별로 Lineplot 생성
for i, metric in enumerate(metrics):
    sns.lineplot(
        data=df, 
        x='Day', 
        y=metric, 
        hue='Experience_Level', 
        style='Experience_Level', # 선 모양 구분
        markers=True,            # 데이터 포인트 표시
        dashes=False,            # 실선 유지
        ax=axes[i]
    )
    
    axes[i].set_title(f'Change in {metric} over Time by Experience Level', fontsize=14)
    axes[i].set_ylabel(metric)
    axes[i].legend(title='Level', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xlabel('Day')
plt.tight_layout()
plt.show()
# %%
stack_raw = pd.read_csv('survey_results_public.csv')
stack = stack_raw.copy()
stack.describe()

df1 = stack[['AIFrustration', 'JobSat']]
df1 = df1[df1['AIFrustration'].notnull() & df1['JobSat'].notnull()]
df1.plot('AIFrustration', 'JobSat', kind='scatter')

#%%
df1 = stack['AISelect'].copy()
df1 = df1[df1.str.contains('Yes') == True]
df1.value_counts()
df1 = stack[['WorkExp', 'JobSat']]
df1 = df1[df1['WorkExp'].notnull() & df1['JobSat'].notnull()]
df1.plot('WorkExp', 'JobSat', kind='scatter', legend=False)


#%%
#df1 = stack['AISelect'].copy()
#dfai = stack[df1.str.contains('Yes') == True]

# %%
# 2. 경력(WorkExp) 구간 나누기 (취준생/신입과 시니어를 비교하기 위함)
def bin_experience(exp):
    if exp <= 2: return '0-2 years (Junior)'
    elif exp <= 5: return '3-5 years'
    elif exp <= 10: return '6-10 years'
    elif exp <= 20: return '11-20 years'
    else: return '21+ years (Senior)'

stack['ExpGroup'] = stack['WorkExp'].apply(bin_experience)

# 3. 데이터 집계 및 정렬
# AISent의 일반적인 순서: Very favorable > Favorable > Indifferent > Unfavorable > Very unfavorable
sent_order = ['Very favorable', 'Favorable', 'Indifferent', 'Unfavorable', 'Very unfavorable', 'Unsure']
exp_order = ['0-2 years (Junior)', '3-5 years', '6-10 years', '11-20 years', '21+ years (Senior)']

pivot_stack = pd.crosstab(stack['ExpGroup'], stack['AISent'], normalize='index') * 100
pivot_stack = pivot_stack.reindex(index=exp_order, columns=[s for s in sent_order if s in pivot_stack.columns])

# 4. 시각화
plt.figure(figsize=(12, 8))
pivot_stack.plot(kind='barh', stacked=True, colormap='RdYlGn_r', figsize=(12, 7))

plt.title('AI Sentiment by Years of Work Experience', fontsize=15)
plt.xlabel('Percentage (%)')
plt.ylabel('Work Experience Group')
plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig('ai_sent_by_work_exp.png')

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드 (실제 파일명으로 수정 필요)
# df = pd.read_csv('survey_results_public.csv')

# 2. 데이터 전처리 (결측치 제거 및 타입 변환)
df1 = stack.dropna(subset=['WorkExp', 'AIAcc'])
df1['WorkExp'] = pd.to_numeric(df1['WorkExp'], errors='coerce')

# 3. 경력(WorkExp) 구간 나누기 (Junior vs Senior 비교)
def bin_experience(exp):
    if exp <= 2: return '0-2 years (Junior)'
    elif exp <= 5: return '3-5 years'
    elif exp <= 10: return '6-10 years'
    elif exp <= 20: return '11-20 years'
    else: return '21+ years (Senior)'

df1['ExpGroup'] = df1['WorkExp'].apply(bin_experience)

# 4. 신뢰도(AIAcc) 순서 설정 (보통 높은 신뢰도부터 낮은 순서)
# 'Highly trust' > 'Trust' > 'Somewhat trust' > 'Neither trust nor distrust' ... 등
# 실제 데이터에 포함된 값에 맞춰 순서를 정의합니다.
acc_order = ['Highly trust', 'Trust', 'Somewhat trust', 'Neither trust nor distrust', 
             'Somewhat distrust', 'Distrust', 'Highly distrust']
exp_order = ['0-2 years (Junior)', '3-5 years', '6-10 years', '11-20 years', '21+ years (Senior)']

# 5. 교차 집계 및 비율 계산
pivot_df = pd.crosstab(df1['ExpGroup'], df1['AIAcc'], normalize='index') * 100
# 존재하는 컬럼만 순서대로 정렬
existing_acc = [a for a in acc_order if a in pivot_df.columns]
pivot_df = pivot_df.reindex(index=exp_order, columns=existing_acc)

# 6. 시각화 (누적 막대 그래프)
plt.figure(figsize=(14, 8))
pivot_df.plot(kind='barh', stacked=True, colormap='Spectral', figsize=(14, 8))

plt.title('Trust in AI Output Accuracy by Work Experience', fontsize=16, pad=20)
plt.xlabel('Percentage (%)', fontsize=12)
plt.ylabel('Years of Experience', fontsize=12)
plt.legend(title='Trust Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig('ai_accuracy_trust_by_exp.png')

#%%
import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 로드 (파일이 있는 환경에서 실행)
# df = pd.read_csv('survey_results_public.csv')

# 2. 데이터 정제
df1 = stack.dropna(subset=['WorkExp', 'AIComplex'])
df1['WorkExp'] = pd.to_numeric(df1['WorkExp'], errors='coerce')

# 3. 경력 구간 나누기
def bin_exp(x):
    if x <= 2: return '0-2 years (Junior)'
    elif x <= 5: return '3-5 years'
    elif x <= 10: return '6-10 years'
    elif x <= 20: return '11-20 years'
    else: return '21+ years (Senior)'

df1['ExpGroup'] = df1['WorkExp'].apply(bin_exp)

# 4. AIComplex 답변 순서 정의
# 보통 'Very well' > 'Well' > 'Somewhat well' > 'Neither well nor poorly' > 'Poorly' > 'Very poorly'
complex_order = [
    'Very well at handling complex tasks',
    'Good, but not great at handling complex tasks',
    'Neither good or bad at handling complex tasks',
    'Bad at handling complex tasks',
    'Very poor at handling complex tasks',
    "I don't use AI tools for complex tasks / I don't know"
]
exp_order = ['0-2 years (Junior)', '3-5 years', '6-10 years', '11-20 years', '21+ years (Senior)']

# 5. 교차 집계 및 비율 계산
pivot_df1 = pd.crosstab(df1['ExpGroup'], df1['AIComplex'], normalize='index') * 100
existing_cols = [c for c in complex_order if c in pivot_df1.columns]
pivot_df1 = pivot_df1.reindex(index=exp_order, columns=existing_cols)

# 6. 시각화
plt.figure(figsize=(12, 8))
pivot_df1.plot(kind='barh', stacked=True, colormap='RdYlGn', figsize=(12, 7))

plt.title('How well AI handles complex tasks by Work Experience', fontsize=15)
plt.xlabel('Percentage (%)')
plt.ylabel('Years of Experience')
plt.legend(title='AI Performance', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig('ai_complex_by_exp.png')

# %%

stack = stack_raw.copy()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 타입 변환 및 경력 그룹화
stack['WorkExp'] = pd.to_numeric(stack['WorkExp'], errors='coerce')

def bin_exp(x):
    if pd.isna(x): return None
    if x <= 2: return '0-2 yrs (Junior)'
    elif x <= 5: return '3-5 yrs'
    elif x <= 10: return '6-10 yrs'
    elif x <= 20: return '11-20 yrs'
    else: return '21+ yrs (Senior)'

stack['ExpGroup'] = stack['WorkExp'].apply(bin_exp)
exp_order = ['0-2 yrs (Junior)', '3-5 yrs', '6-10 yrs', '11-20 yrs', '21+ yrs (Senior)']

# 2. SOFriction 분석 (순서 정렬 및 필터링)
# 제공해주신 실제 값 기반 순서
fric_map = [
    'More than half the time', 
    'About half of the time', 
    'Less than half of the time', 
    'Rarely, almost never'
]
df_fric = stack[stack['SOFriction'].isin(fric_map)].copy()

# 3. AIFrustration 멀티 선택 데이터 분리 (핵심 가설 지표 추출)
# '디버깅이 더 오래 걸림' 항목이 포함되어 있는지 확인
stack['Frust_Debug_Time'] = stack['AIFrustration'].str.contains('Debugging AI-generated code is more time-consuming', na=False)
# '코드 이해가 어려움' 항목이 포함되어 있는지 확인
stack['Frust_Understanding'] = stack['AIFrustration'].str.contains('It’s hard to understand how or why the code works', na=False)

# --- 수치 표 출력 ---
print("--- [표 1] 경력별 AI 코드 수정/이해 빈도 (SOFriction) ---")
fric_table = pd.crosstab(df_fric['ExpGroup'], df_fric['SOFriction'], normalize='index') * 100
fric_table = fric_table.reindex(index=exp_order, columns=fric_map)
print(fric_table.round(2))

print("\n--- [표 2] 경력별 AI 스트레스 원인 (이해 부족 & 디버깅 시간) ---")
frust_table = stack.groupby('ExpGroup')[['Frust_Debug_Time', 'Frust_Understanding']].mean() * 100
frust_table = frust_table.reindex(exp_order)
print(frust_table.round(2))

# --- 시각화 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))

# 그래프 1: SOFriction (AI 코드 때문에 고생하는 빈도)
fric_table.plot(kind='barh', stacked=True, colormap='Reds', ax=ax1, edgecolor='white')
ax1.set_title('Frequency of Struggling with AI Code (SOFriction)', fontsize=15, pad=15)
ax1.set_xlabel('Percentage of Developers (%)')
ax1.legend(title='Frequency', bbox_to_anchor=(1.05, 1))

# 그래프 2: AIFrustration (구체적인 고충 비교)
frust_table.plot(kind='bar', ax=ax2, color=['#e74c3c', '#3498db'], rot=0)
ax2.set_title('Specific AI Frustrations: Debugging Time & Understanding Gap', fontsize=15, pad=15)
ax2.set_ylabel('Percentage (%)')
ax2.set_xlabel('Experience Group')
ax2.legend(['Debugging is more time-consuming', 'Hard to understand how code works'])
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
# %%
#stack_raw['SOFriction'].value_counts()
str_bin = set()
for str in stack['AIFrustration'].unique():
    if type(str) != float:
        for tok in str.split(';'):
            str_bin.add(tok)

str_bin

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

stack = stack_raw.copy()
# 1. 경력 데이터 수치화 및 그룹화
stack['WorkExp'] = pd.to_numeric(stack['WorkExp'], errors='coerce')

def bin_exp(x):
    if pd.isna(x): return None
    if x <= 2: return '0-2 yrs (Junior)'
    elif x <= 5: return '3-5 yrs'
    elif x <= 10: return '6-10 yrs'
    elif x <= 20: return '11-20 yrs'
    else: return '21+ yrs (Senior)'

stack['ExpGroup'] = stack['WorkExp'].apply(bin_exp)
exp_order = ['0-2 yrs (Junior)', '3-5 yrs', '6-10 yrs', '11-20 yrs', '21+ yrs (Senior)']

# 2. 분석할 핵심 지표 정의 (유저님이 주신 선택지 기반)
# 컬럼명은 데이터에 맞춰 AIFrustration(혹은 해당 값이 들어있는 컬럼)으로 설정하세요.
target_col = 'AIFrustration' 

problems = {
    'Bugs (Almost right)': 'AI solutions that are almost right, but not quite',
    'Debugging Time': 'Debugging AI-generated code is more time-consuming',
    'Understanding Gap': 'It’s hard to understand how or why the code works',
    'Loss of Confidence': 'I’ve become less confident in my own problem-solving'
}

# 각 항목별로 선택 여부(True/False) 컬럼 생성
for label, snippet in problems.items():
    stack[label] = stack[target_col].str.contains(snippet, na=False, regex=False)

# 3. 경력별 발생 비율 계산 (%)
analysis_table = stack.groupby('ExpGroup')[list(problems.keys())].mean() * 100
analysis_table = analysis_table.reindex(exp_order)

# --- 결과 출력 ---
print("--- [가설 검증 표] 경력별 AI 활용 고충 발생 비율 (%) ---")
print(analysis_table.round(2))

# 4. 시각화 (그룹 막대 그래프)
plt.figure(figsize=(14, 8))
analysis_table.plot(kind='bar', figsize=(14, 7), rot=0, width=0.8)

plt.title('AI-Related Challenges by Work Experience', fontsize=16, pad=20)
plt.ylabel('Percentage of Developers (%)', fontsize=12)
plt.xlabel('Experience Group', fontsize=12)
plt.legend(title='Challenge Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig('ai_challenge_analysis.png')
plt.show()
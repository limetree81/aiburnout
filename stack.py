#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
columns = ['YearsCodePro', 'OrgSize', 'Employment', 'AIThreat', 'CompTotal']
# 모든 컬럼을 불러오지 않고 분석에 필요한 것만 선택해서 로드
df = pd.read_csv('survey_results_public.csv', usecols=['OrgSize', 'Employment', 'AIThreat', 'CompTotal'])
df
# %%
df.groupby('OrgSize')['AIThreat'].value_counts()

#%%
df['OrgSize'].unique()
['20 to 99 employees', '500 to 999 employees', nan,
       '10,000 or more employees', 'Less than 20 employees',
       '5,000 to 9,999 employees', '100 to 499 employees',
       '1,000 to 4,999 employees', 'I don’t know',
       'Just me - I am a freelancer, sole proprietor, etc.']
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드 (실제 파일명이 다를 경우 수정 필요)
# df = pd.read_csv('survey_results_public.csv')

# 2. 기업 규모(OrgSize) 순서 정렬 (설문 데이터 특성상 수동 정렬이 깔끔합니다)
org_order = [
    'Just me - I am a freelancer, sole proprietor, etc.', 
    '2 to 9 employees', '10 to 19 employees', '20 to 99 employees', 
    '100 to 499 employees', '500 to 999 employees', 
    '1,000 to 4,999 employees', '5,000 to 9,999 employees', 
    '10,000 or more employees', 'I don’t know'
]

# 3. 데이터 집계 및 비율 계산
pivot_df = df.groupby(['OrgSize', 'AIThreat']).size().unstack(fill_value=0)
pivot_df = pivot_df.reindex(org_order) # 순서 정렬
pivot_perc = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100 # 백분율 변환

# 4. 시각화
plt.figure(figsize=(12, 7))
pivot_perc.plot(kind='barh', stacked=True, cmap='RdYlGn_r', figsize=(12, 8))

plt.title('AI Threat Perception by Organization Size', fontsize=15, pad=20)
plt.xlabel('Percentage (%)')
plt.ylabel('Organization Size')
plt.legend(title='Is AI a threat?', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.savefig('ai_threat_by_org_size.png')

#%%

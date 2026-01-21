#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

stack_raw = pd.read_csv('survey_results_public.csv')
stack = stack_raw.copy()

#%%

# cols = []
# for col in stack.columns:
#     print(stack[col].value_counts())
#     y = input('y/n')
#     if 'y' in y:
#         cols.append(col)

# print(cols)
# %%
cols = ['WorkExp', 'YearsCode', 'OrgSize', \
        #'TechEndorse_1', 'TechEndorse_2', 'TechEndorse_3', 'TechEndorse_4', 'TechEndorse_5', 'TechEndorse_6', 'TechEndorse_7', 'TechEndorse_8', 'TechEndorse_9', 'TechEndorse_13', \
        #'TechOppose_1', 'TechOppose_2', 'TechOppose_3', 'TechOppose_5', 'TechOppose_7', 'TechOppose_9', 'TechOppose_11', 'TechOppose_13', 'TechOppose_16', 'TechOppose_15', \
        'JobSatPoints_1', 'JobSatPoints_2', 'JobSatPoints_3', 'JobSatPoints_4', 'JobSatPoints_5', 'JobSatPoints_6', 'JobSatPoints_7', 'JobSatPoints_8', 'JobSatPoints_9', 'JobSatPoints_10', 'JobSatPoints_11', 'JobSatPoints_13', 'JobSatPoints_14', 'JobSatPoints_15', 'JobSatPoints_16', \
        'AIThreat', 'AISelect', 'AISent', 'AIAcc', 'AIComplex', \
        'AIFrustration', 'AIExplain', 'AIAgents', 'AIAgentChange', 'ConvertedCompYearly', 'JobSat']

stack = stack[cols]

#%%
# 중앙값 매핑 사전 정의
size_map = {
    'Just me - I am a freelancer, sole proprietor, etc.': 1,
    'Less than 20 employees': 10, # 1~19의 중앙값 (약 10)
    '20 to 99 employees': 59.5,   # (20+99)/2
    '100 to 499 employees': 299.5, # (100+499)/2
    '500 to 999 employees': 749.5, # (500+999)/2
    '1,000 to 4,999 employees': 2999.5, # (1000+4999)/2
    '5,000 to 9,999 employees': 7499.5, # (5000+9999)/2
    '10,000 or more employees': 10000    # 하한선인 10000으로 설정 (혹은 분석 목적에 따라 가중치 부여)
}

# 새로운 컬럼 생성
# map() 함수를 사용하면 사전에 없는 값(I don't know 등)은 자동으로 NaN이 됩니다.
stack['OrgSize_'] = stack['OrgSize'].map(size_map)

# %%
stack['AIThreat'].value_counts()
# AIThreat 변환 매핑 사전 정의 (순서 의미 부여)
threat_map = {
    'No': 0,
    'I\'m not sure': 1,
    'Yes': 2
}

stack['AIThreat_'] = stack['AIThreat'].map(threat_map)

#%%
stack['AISelect'].value_counts()
# 사용 빈도를 '월간 사용 횟수'로 변환하는 매핑
select_map = {
    'Yes, I use AI tools daily': 30,
    'Yes, I use AI tools weekly': 4.3,
    'Yes, I use AI tools monthly or infrequently': 1,
    'No, but I plan to soon': 0,
    'No, and I don\'t plan to': 0
}

# AISelect_ 컬럼 생성
stack['AISelect_'] = stack['AISelect'].map(select_map)

# 변환 확인
#print(stack[['AISelect', 'AISelect_']].drop_duplicates())

#%%
stack['AISent'].value_counts()
# 호감도 매핑 (긍정은 +, 부정은 -)
sent_map = {
    'Very favorable': 2,
    'Favorable': 1,
    'Indifferent': 0,
    'Unfavorable': -1,
    'Very unfavorable': -2,
    'Unsure': np.nan  # 분석의 정확도를 위해 null 처리
}

# AISent_ 컬럼 생성
stack['AISent_'] = stack['AISent'].map(sent_map)

# 결과 확인
# print(stack[['AISent', 'AISent_']].value_counts().sort_index())

##AI에 대한 
# %%
stack['AIAcc'].value_counts()
# 신뢰도 매핑 (신뢰는 +, 불신은 -)
acc_map = {
    'Highly trust': 2,
    'Somewhat trust': 1,
    'Neither trust nor distrust': 0,
    'Somewhat distrust': -1,
    'Highly distrust': -2
}

# AIAcc_ 컬럼 생성
stack['AIAcc_'] = stack['AIAcc'].map(acc_map)

# 결과 확인
#print(stack[['AIAcc', 'AIAcc_']].value_counts().sort_index())

#%%
stack['AIComplex'].value_counts()
# 복잡한 작업 처리 능력 매핑
complex_map = {
    'Very well at handling complex tasks': 2,
    'Good, but not great at handling complex tasks': 1,
    'Neither good or bad at handling complex tasks': 0,
    'Bad at handling complex tasks': -1,
    'Very poor at handling complex tasks': -2,
    'I don\'t use AI tools for complex tasks / I don\'t know': np.nan
}

# AIComplex_ 컬럼 생성
stack['AIComplex_'] = stack['AIComplex'].map(complex_map)

# 결과 확인
#print(stack[['AIComplex', 'AIComplex_']].value_counts().sort_index())

#%%
stack['AIFrustration'].value_counts()

# 1. 세미콜론으로 구분된 데이터를 분리하여 원-핫 인코딩(0, 1) 생성
# sep=';'를 기준으로 나누고, 접두사(prefix)를 붙입니다.
frust_dummies = stack['AIFrustration'].str.get_dummies(sep=';')

# 2. 컬럼명 정리 (불필요한 공백 제거 및 형식 맞춤)
# 사용자님이 요청하신 'AIFrust_' 접두사를 붙입니다.
frust_dummies.columns = ['AIFrust_' + col.strip().replace(' ', '_') for col in frust_dummies.columns]

# 3. 원본 AIFrustration이 NaN이었던 행들은 다시 NaN으로 복구
# get_dummies는 기본적으로 NaN을 0으로 변환하므로, 이를 다시 masking 처리합니다.
frust_dummies[stack['AIFrustration'].isna()] = np.nan

# 4. 기존 데이터프레임에 합치기
stack = pd.concat([stack, frust_dummies], axis=1)

# 생성된 컬럼 확인 (예시로 몇 개만 출력)
print(frust_dummies.columns.tolist())
# %%
stack['AIExplain'].value_counts()
def classify_explain(text):
    # NaN(결측치) 처리
    if pd.isna(text):
        return None
    
    # 소문자 변환
    text_lower = str(text).lower()
    
    # 'yes'가 포함된 경우 (긍정)
    if 'yes' in text_lower:
        return True
    # 'no'가 포함된 경우 (부정)
    elif 'no' in text_lower:
        return False
    # 그 외 (판단 불가 또는 무응답)
    else:
        return None

# 새로운 컬럼 생성
stack['AIExplain_'] = stack['AIExplain'].apply(classify_explain)

# 결과 확인 (각 값의 분포 확인)
print(stack['AIExplain_'].value_counts(dropna=False))

#%%
stack['AIAgents'].value_counts()
# 에이전트 사용 빈도 매핑 (월간 사용 횟수 기준)
agents_map = {
    'Yes, I use AI agents at work daily': 30,
    'Yes, I use AI agents at work weekly': 4.3,
    'Yes, I use AI agents at work monthly or infrequently': 1,
    'No, I use AI exclusively in copilot/autocomplete mode': 0,
    'No, but I plan to': 0,
    'No, and I don\'t plan to': 0
}

# AIAgents_ 컬럼 생성
stack['AIAgents_'] = stack['AIAgents'].map(agents_map)

# 결과 확인
print(stack[['AIAgents', 'AIAgents_']].value_counts().sort_index())
#%%
stack['AIAgentChange'].value_counts()
# AI로 인한 변화 정도 매핑
change_map = {
    'Yes, to a great extent': 2,
    'Yes, somewhat': 1,
    'Not at all or minimally': 0,
    'No, but my development work has changed somewhat due to non-AI factors': 0,
    'No, but my development work has significantly changed due to non-AI factors': 0
}

# AIAgentChange_ 컬럼 생성
stack['AIAgentChange_'] = stack['AIAgentChange'].map(change_map)

# 결과 확인
#print(stack[['AIAgentChange', 'AIAgentChange_']].value_counts().sort_index())

#%%

stack.to_csv('stack_survey_corr.csv')
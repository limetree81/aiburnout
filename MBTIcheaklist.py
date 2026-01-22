import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os


# --- [수정] 파일 경로 설정 ---
# 현재 스크립트가 있는 폴더에서 'data'라는 하위 폴더 내의 파일을 찾도록 설정합니다.
FILE_NAME = "survey_results_public_2025.csv"

# 파일이 존재하는지 확인 (없으면 AssertionError 발생)
#assert os.path.exists(FILE_NAME), f"'{FILE_NAME}' 파일이 없습니다. 압축을 해제하여 해당 위치에 파일을 생성해 주세요."


# 1. 페이지 설정 및 디자인
st.set_page_config(page_title="2025 AI-Dev MBTI", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f9fbfd; }
    .stRadio {
        background-color: white;
        padding: 25px 30px;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 25px;
        border: 1px solid #edf2f7;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #2563eb 0%, #06b6d4 100%);
    }
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 700;
        height: 3.5em;
        background-color: #2563eb;
        color: white;
        border: none;
    }
    .res-title {
        text-align: center;
        color: #2563eb;
        font-size: 2.5rem;
        margin-bottom: 5px;
        font-weight: 800;
    }
    .hashtag-text {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 20px;
        font-weight: 500;
    }
    .guide-box {
        background-color: #f0f7ff;
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #2563eb;
        margin: 25px 0;
        line-height: 1.6;
    }
    .badge-container {
        text-align: center;
        margin-bottom: 15px;
    }
    .level-badge {
        padding: 6px 16px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.9rem;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. 데이터 분석 로직
def process_mbti_distribution(file_path_or_buffer):
    try:
        df = pd.read_csv(file_path_or_buffer)
        df['RemoteWork'] = df['RemoteWork'].fillna('Hybrid')
        df['LearnCode'] = df['LearnCode'].fillna('')
        df['AIThreat'] = df['AIThreat'].fillna('Neutral')
        df['WorkExp'] = pd.to_numeric(df['WorkExp'], errors='coerce').fillna(5)

        ei = df['RemoteWork'].apply(lambda x: "I" if "Remote" in str(x) else "E")
        sn = df['LearnCode'].apply(lambda x: "N" if "AI" in str(x) else "S")
        tf = df['AIThreat'].apply(lambda x: "T" if "not concerned" in str(x).lower() else "F")
        jp = df['WorkExp'].apply(lambda x: "J" if x > 8 else "P")
        
        return (ei + sn + tf + jp).tolist()
    except:
        return []

# 세션 상태 초기화
if 'page' not in st.session_state:
    st.session_state.page = 0
if 'scores' not in st.session_state:
    st.session_state.scores = {k: 0 for k in "EISTNFJP"}
if 'external_data' not in st.session_state:
    file_name = 'survey_results_public_2025.csv'
    if os.path.exists(file_name):
        st.session_state.external_data = process_mbti_distribution(file_name)
    else:
        st.session_state.external_data = []

if not st.session_state.external_data:
    st.warning("📊 분석을 위해 'survey_results_public_2025.csv' 파일을 업로드해주세요.")
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=['csv'])
    if uploaded_file:
        st.session_state.external_data = process_mbti_distribution(uploaded_file)
        st.rerun()
    st.stop()

# 3. 질문 및 데이터 정의 (생략 없음)
questions = [
    ("AI 도구가 해결하지 못한 버그를 커뮤니티나 동료에게 질문해 해결하는 편인가요?", "E", "I"),
    ("AI와 대화하며 코딩하는 것보다 팀원과 함께 프로그래밍하는 게 더 즐겁나요?", "E", "I"),
    ("AI 기술 트렌드에 대해 동료들과 이야기 나누는 것을 좋아하시나요?", "E", "I"),
    ("나만 아는 AI 프롬프트 꿀팁을 블로그나 SNS에 공유하고 싶나요?", "E", "I"),
    ("AI가 짠 코드라도 내가 직접 한 줄씩 검증하지 않으면 불안한가요?", "S", "N"),
    ("AI의 혁신적인 제안보다 현재 프로젝트의 안정적인 진행이 더 중요한가요?", "S", "N"),
    ("AI가 생성한 추상적인 아키텍처보다 구체적인 함수 로직 구현이 더 흥미롭나요?", "S", "N"),
    ("AI의 답변이 모호할 때, '일단 실행'보다 공식 문서를 먼저 찾아보나요?", "S", "N"),
    ("AI 도입 여부를 결정할 때 팀원들의 기분보다 비용 대비 효율성이 최우선인가요?", "T", "F"),
    ("AI가 내 코드를 비판할 때, 논리적으로 맞다면 전혀 상처받지 않나요?", "T", "F"),
    ("개발자의 가치는 '감성적 사용자 경험'보다 '코드의 성능과 논리'에 있다고 보나요?", "T", "F"),
    ("AI가 동료의 일자리를 대체할 가능성을 경제적 현상으로 담담하게 보나요?", "T", "F"),
    ("AI 도구를 사용할 때도 미리 정해진 작업계획과 규칙에 맞춰 사용하시나요?", "J", "P"),
    ("AI가 갑자기 새로운 기능을 제안하면, 계획에 없던 일이 생겨서 불편한가요?", "J", "P"),
    ("프로젝트 마감 기한을 지키기 위해 AI의 도움을 빌려 계획대로 끝내는 편인가요?", "J", "P"),
    ("AI가 실시간으로 코드를 추천해주는 것보다 내가 직접 계획한 대로 타이핑하는 게 편한가요?", "J", "P")
]

mbti_types = {
    "ISTJ": "철저한 AI 검수자", "ISFJ": "헌신적인 코드 수호자", "INFJ": "통찰력 있는 비전가", "INTJ": "전략적인 AI 마스터",
    "ISTP": "효율적인 문제 해결사", "ISFP": "예술적인 코드 작가", "INFP": "이상적인 기술 중재자", "INTP": "논리적인 기술 분석가",
    "ESTP": "대담한 기술 모험가", "ESFP": "에너지 넘치는 분위기 메이커", "ENFP": "열정적인 혁신가", "ENTP": "도전적인 기술 변론가",
    "ESTJ": "체계적인 팀 리더", "ESFJ": "따뜻한 기술 전파자", "ENFJ": "정의로운 멘토", "ENTJ": "강력한 기술 통솔자"
}

mbti_keywords = {
    "ISTJ": "#완벽주의 #원칙주의 #안전제일", "ISFJ": "#세심함 #서포트 #꼼꼼한코드", "INFJ": "#통찰력 #미래지향 #가치관중심", "INTJ": "#효율극대화 #논리왕 #전략가",
    "ISTP": "#실용주의 #도구장인 #해결사", "ISFP": "#유연함 #심미안 #감각적코딩", "INFP": "#이상가 #창의성 #성장지향", "INTP": "#지적호기심 #원리파악 #기술탐구",
    "ESTP": "#행동파 #위기대응 #실무중심", "ESFP": "#친화력 #즐거운개발 #에너자이저", "ENFP": "#아이디어 #열정 #유연한사고", "ENTP": "#혁신 #토론 #새로운시도",
    "ESTJ": "#체계적 #관리 #목표달성", "ESFJ": "#조화 #협업 #커뮤니티", "ENFJ": "#멘토링 #영향력 #동반성장", "ENTJ": "#카리스마 #추진력 #시스템설계"
}

mbti_ai_guides = {
    "ISTJ": "AI가 제안한 코드를 기존의 팀 컨벤션과 표준 라이브러리에 맞춰 철저히 검증할 때 가장 빛납니다. 코드의 '창의성'보다는 '정확성'을 잡는 최후의 보루 역할을 수행하세요.",
    "ISFJ": "AI를 활용해 문서화나 주석 작업을 자동화하여 팀원들이 코드를 더 쉽게 이해하도록 돕는 데 집중하세요. 기술을 통해 협업의 온도를 높이는 데 탁월합니다.",
    "INFJ": "AI를 단순히 코드 생성기가 아닌 '복잡한 아키텍처의 타당성'을 검토하는 토론 파트너로 삼으세요. 긴 안목으로 시스템의 부작용을 예측하는 당신의 통찰과 AI가 만나면 무적입니다.",
    "INTJ": "단순 코딩은 AI에게 맡기고, 당신은 고도의 추상화 설계와 전체 시스템의 최적화 전략에 집중하세요. AI를 부하 직원처럼 부리며 기술적 결정을 내리는 '지휘관' 스타일이 제격입니다.",
    "ISTP": "새로운 라이브러리나 도구를 도입할 때 AI를 활용해 빠르게 프로토타입을 만들어보세요. 긴 설명보다는 '어떻게 작동하는지' 결과물을 바로 확인하며 학습할 때 효율이 극대화됩니다.",
    "ISFP": "AI를 활용해 UI/UX의 다양한 시각적 대안을 생성해보세요. 당신의 미적 감각을 코드로 변환하는 과정에서 AI는 아주 훌륭한 '조수'가 되어줄 것입니다.",
    "INFP": "당신의 창의적인 아이디어가 기술적인 한계에 부딪릴 때 AI에게 구현 방법을 물어보세요. 코드라는 도구를 넘어 당신의 가치관을 서비스에 투영하는 데 집중할 수 있게 해줍니다.",
    "INTP": "AI의 답변에서 논리적 모순을 찾아내고, 새로운 기술의 내부 작동 원리를 파악하는 백과사전으로 AI를 쓰세요. 당신의 지적 탐구심을 채워줄 최고의 데이터 소스가 될 것입니다.",
    "ESTP": "현장에서 발생하는 실시간 버그와 긴급 장애 대응에 AI를 적극 활용하세요. 빠르게 해결책을 찾아내고 즉각 실행에 옮기는 당신의 과감함과 AI의 속도는 최고의 조합입니다.",
    "ESFP": "지루한 반복 코딩과 환경 설정은 AI에게 맡기고, 당신은 팀의 사기를 북돋고 사용자의 피드백을 수집하는 등 '사람' 중심의 활동에 더 많은 시간을 할애하세요.",
    "ENFP": "아이디어가 넘쳐 흐를 때 AI를 활용해 그 생각들을 구조화된 초안으로 만드세요. AI를 '브레인스토밍 파트너'로 활용하면 당신의 상상력이 실질적인 결과물로 빠르게 전환됩니다.",
    "ENTP": "AI가 제안한 방식에 대해 '왜?'라고 끊임없이 질문하며 더 나은 대안을 찾아보세요. AI와의 기술적 논쟁은 당신의 논리를 날카롭게 하고 예상치 못한 혁신을 만들어냅니다.",
    "ESTJ": "AI를 업무 프로세스의 자동화와 팀 표준 구축에 투입하세요. 명확한 규칙을 기반으로 AI가 워크플로우를 관리하게 함으로써 프로젝트의 가시성과 목표 달성률을 높일 수 있습니다.",
    "ESFJ": "팀 내 기술 공유 문화를 만드는 데 AI를 활용하세요. 온보딩 가이드를 생성하거나 팀원들의 실수를 부드럽게 지적해주는 리뷰 가이드를 만드는 등 조화를 이루는 데 AI가 큰 도움을 줍니다.",
    "ENFJ": "AI를 활용해 동료들의 성장을 돕는 학습 로드맵이나 피드백 초안을 작성해보세요. 기술이 사람을 향하게 만드는 당신의 리더십을 지원하는 최고의 비서가 될 것입니다.",
    "ENTJ": "대규모 프로젝트의 자원 배분과 기술 스택 선정을 위한 분석 도구로 AI를 사용하세요. 목표를 향해 가장 빠른 길을 찾는 당신의 추진력에 AI의 데이터 분석력을 더해 승리하는 전략을 짜세요."
}

trait_details = {
    "E": "협업과 소통에서 에너지를 얻으며, AI 도구를 팀의 생산성을 높이는 공유 자산으로 활용합니다.",
    "I": "독립적인 작업 환경을 선호하며, AI를 개인의 성능을 극대화하는 파트너로 활용합니다.",
    "S": "구체적인 로직과 안정성을 중시하며, AI 결과물을 현실적인 가이드라인에 맞춰 검증합니다.",
    "N": "미래의 혁신 가능성에 주목하며, AI와 함께 창의적인 대안을 구상하는 것을 즐깁니다.",
    "T": "객관적 사실과 효율성을 우선합니다. AI의 제안을 이성적으로 분석하여 타당성을 따집니다.",
    "F": "팀원 간의 조화와 사용자 경험을 소중히 여기며, 기술이 사람에게 주는 가치에 주목합니다.",
    "J": "체계적인 계획을 선호하며, AI를 통해 워크플로우를 자동화하고 마감 기한을 엄수합니다.",
    "P": "유연하고 개방적인 태도를 지향하며, AI가 제안하는 변화를 적극적으로 수용합니다."
}

# 4. 설문 진행 로직
if st.session_state.page < 4:
    st.markdown("<h1 style='text-align: center;'>🤖 AI-Dev MBTI 검사</h1>", unsafe_allow_html=True)
    st.progress((st.session_state.page + 1) / 4)
    start_idx = st.session_state.page * 4
    end_idx = start_idx + 4
    
    current_responses = {}
    for i in range(start_idx, end_idx):
        q_text, pos, neg = questions[i]
        choice = st.radio(f"Q{i+1}. {q_text}", ["매우 그렇다", "그렇다", "아니다", "전혀 아니다"], horizontal=True, key=f"q_{i}")
        if choice == "매우 그렇다": current_responses[i] = (pos, 2)
        elif choice == "그렇다": current_responses[i] = (pos, 1)
        elif choice == "아니다": current_responses[i] = (neg, 1)
        else: current_responses[i] = (neg, 2)

    if st.button("결과 분석하기" if st.session_state.page == 3 else "다음 단계로"):
        for i in range(start_idx, end_idx):
            t_key, val = current_responses[i]
            st.session_state.scores[t_key] += val
        st.session_state.page += 1
        st.rerun()

# 5. 결과 페이지
else:
    scores = st.session_state.scores
    res = "".join(["E" if scores["E"] >= scores["I"] else "I", "S" if scores["S"] >= scores["N"] else "N", 
                   "T" if scores["T"] >= scores["F"] else "F", "J" if scores["J"] >= scores["P"] else "P"])
    
    st.balloons()

    # --- [추가] 뱃지 로직 ---
    total_raw_score = sum(scores.values())
    if total_raw_score >= 24:
        level, l_color = "AI없인 못살아!", "#ef4444"
    elif total_raw_score >= 18:
        level, l_color = "AI아직 잘 못믿겟는걸?", "#3b82f6"
    else:
        level, l_color = "AI와 아직 어색어색", "#10b981"
    
    # 상단 정보
    st.markdown(f"<div class='res-title'>{res}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='badge-container'><span class='level-badge' style='background-color:{l_color}'>LV. {level}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hashtag-text'>{mbti_keywords.get(res)}</div>", unsafe_allow_html=True)
    
    # 아바타
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        avatar_url = f"https://api.dicebear.com/7.x/avataaars/svg?seed={res}&backgroundColor=b6e3f4"
        st.image(avatar_url, use_container_width=True)
    
    st.markdown(f"<h2 style='text-align: center; margin-top: 10px;'>{mbti_types.get(res)}</h2>", unsafe_allow_html=True)

    # --- [추가] 희귀도 계산 로직 ---
    all_data = st.session_state.external_data + [res]
    type_counts = pd.Series(all_data).value_counts()
    my_type_rank = type_counts.index.get_loc(res) + 1
    my_type_ratio = (type_counts[res] / len(all_data)) * 100
    
    st.markdown(f"""
    <div style='display: flex; justify-content: space-around; background: white; padding: 20px; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 20px;'>
        <div style='text-align: center;'>
            <div style='color: #64748b; font-size: 0.8rem;'>희귀도 순위</div>
            <div style='color: #2563eb; font-size: 1.5rem; font-weight: 800;'>{my_type_rank}위</div>
        </div>
        <div style='border-left: 1px solid #e2e8f0;'></div>
        <div style='text-align: center;'>
            <div style='color: #64748b; font-size: 0.8rem;'>전체 중 비율</div>
            <div style='color: #2563eb; font-size: 1.5rem; font-weight: 800;'>{my_type_ratio:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 협업 가이드
    st.markdown(f"""
    <div class='guide-box'>
        <h4 style='color: #2563eb; margin-top: 0; margin-bottom: 15px;'>✨ {res} 개발자를 위한 AI 활용 전략</h4>
        <p style='color: #1e293b; font-size: 1.05rem;'>{mbti_ai_guides.get(res)}</p>
    </div>
    """, unsafe_allow_html=True)

    # 레이더 차트
    categories = ['E-I', 'S-N', 'T-F', 'J-P']
    user_vals = [
        (scores['E']/(scores['E']+scores['I'])*100) if (scores['E']+scores['I'])>0 else 50,
        (scores['S']/(scores['S']+scores['N'])*100) if (scores['S']+scores['N'])>0 else 50,
        (scores['T']/(scores['T']+scores['F'])*100) if (scores['T']+scores['F'])>0 else 50,
        (scores['J']/(scores['J']+scores['P'])*100) if (scores['J']+scores['P'])>0 else 50
    ]
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=user_vals, theta=categories, fill='toself', 
        fillcolor='rgba(37, 99, 235, 0.3)', line_color='#2563eb'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 100])), 
        margin=dict(l=80, r=80, t=40, b=40), height=450
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # 결과 요약 복사
    share_text = f"🤖 나의 AI-Dev MBTI: [{res}] {mbti_types.get(res)}\n🏆 희귀도: {my_type_rank}위 ({my_type_ratio:.1f}%)\n🎖 레벨: {level}\n#AI_MBTI #개발자테스트"
    if st.button("📋 결과 요약 텍스트 복사하기 (공유용)"):
        st.info(f"아래 텍스트를 복사하세요:\n\n{share_text}")

    st.write("---")
    
    # 지표별 상세 분석
    st.markdown("### 📊 지표별 상세 코멘트")
    for p, n in [("E","I"), ("S","N"), ("T","F"), ("J","P")]:
        p_v, n_v = scores[p], scores[n]
        total = p_v + n_v
        ratio = (p_v / total * 100) if total > 0 else 50
        dominant = p if ratio >= 50 else n
        with st.container():
            c_a, c_b = st.columns([1, 4])
            c_a.metric(f"{p} vs {n}", f"{int(ratio if ratio>=50 else 100-ratio)}%", delta=dominant)
            c_b.info(f"**{dominant} 성향 분석:** {trait_details[dominant]}")
            st.progress(ratio / 100)
            st.write("")

    # 글로벌 분포 (도넛 차트)
    st.write("---")
    dist_df = type_counts.reset_index()
    dist_df.columns = ['유형', '인원']
    
    st.markdown(f"### 🌏 글로벌 개발자 유형 분포 (Total: {len(all_data):,})")
    fig_donut = px.pie(dist_df, values='인원', names='유형', hole=0.6,
                       color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_donut.update_traces(textinfo='percent+label', pull=[0.1 if x == res else 0 for x in dist_df['유형']])
    fig_donut.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.3))
    st.plotly_chart(fig_donut, use_container_width=True)

    if st.button("🔄 테스트 다시 하기"):
        st.session_state.page = 0
        st.session_state.scores = {k: 0 for k in "EISTNFJP"}
        st.rerun()
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from collections import Counter
import textwrap

# -----------------------------------------------------------------------------
# 1. Page Configuration & Aesthetic Style
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI & Burnout Analytics", layout="wide", page_icon="🛡️")

# Set a premium plotting style (Minimalist)
plt.style.use('default') # Reset defaults
sns.set_theme(style="white", context="talk") # Clean white background
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Custom Color Palette (Premium Pastels)
# Junior(Danger/Red-ish), Mid(Neutral/Orange-ish), Senior(Safe/Green-ish)
PREMIUM_PALETTE = {
    'Junior': '#E63946',     # Intense Red
    'Mid-Level': '#F4A261',  # Muted Orange
    'Senior': '#2A9D8F',     # Elegant Teal
    'Refuser': '#BDBDBD',    # Gray
    'Dependent': '#E63946',  # Red
    'Verifier': '#2A9D8F'    # Teal
}

# CSS for Dashboard UI
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 36px;
        font-weight: 700;
        color: #1D3557;
        margin-bottom: 10px;
    }
    .sub-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 18px;
        color: #457B9D;
        margin-bottom: 30px;
    }
    .card {
        background-color: #F1FAEE;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1D3557;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Data Loading & Logic
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_prep_data():
    df_burnout = None
    df_so = None
    
    # Load CSVs
    if os.path.exists("ai_workplace_productivity_burnout.csv"):
        df_burnout = pd.read_csv("ai_workplace_productivity_burnout.csv")
    if os.path.exists("survey_results_public_2025.csv"):
        df_so = pd.read_csv("survey_results_public_2025.csv", low_memory=False)
        
    # --- Preprocessing Burnout Data ---
    if df_burnout is not None:
        # 1. Order Experience Levels
        if 'Experience_Level' in df_burnout.columns:
            cats = ['Junior', 'Mid-Level', 'Senior']
            existing = [c for c in cats if c in df_burnout['Experience_Level'].unique()]
            df_burnout['Experience_Level'] = pd.Categorical(
                df_burnout['Experience_Level'], categories=existing, ordered=True
            )
        
        # 2. Define Junior Types (Refuser / Dependent / Verifier)
        if 'AI_Tool_Usage_%' in df_burnout.columns and 'Bugs_Introduced' in df_burnout.columns:
            bug_median = df_burnout['Bugs_Introduced'].median()
            def get_type(row):
                if row['AI_Tool_Usage_%'] < 20: return 'Refuser'
                elif row['Bugs_Introduced'] > bug_median: return 'Dependent'
                else: return 'Verifier'
            df_burnout['Junior_Type'] = df_burnout.apply(get_type, axis=1)

    # --- Preprocessing SO Data ---
    if df_so is not None:
        if 'YearsCode' in df_so.columns:
            df_so['YearsCode_Num'] = pd.to_numeric(df_so['YearsCode'], errors='coerce').fillna(0)
            df_so['Career_Group'] = pd.cut(
                df_so['YearsCode_Num'], bins=[-1, 3, 7, 100], 
                labels=['Junior', 'Mid-Level', 'Senior']
            )

    return df_burnout, df_so

# -----------------------------------------------------------------------------
# 3. Visualization Helpers
# -----------------------------------------------------------------------------
def plot_trend_line(data, x, y, group_col, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Loop through groups to draw clean lines
    groups = data[group_col].unique()
    for grp in groups:
        if pd.isna(grp): continue
        subset = data[data[group_col] == grp]
        color = PREMIUM_PALETTE.get(grp, '#333333')
        
        # Regression Plot with NO scatter points for cleanliness
        sns.regplot(
            data=subset, x=x, y=y, 
            scatter=False, label=grp, color=color,
            line_kws={'linewidth': 3}, ax=ax
        )
        # Add a very faint fill to show variance if desired, or keep it strict line
        
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(frameon=False, loc='upper left')
    return fig

def plot_bar_comparison(data, x, y, hue, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=data, x=x, y=y, hue=hue, 
        palette=PREMIUM_PALETTE, errorbar=None, ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("")
    ax.set_ylabel(y.replace('_', ' '))
    ax.legend(frameon=False)
    return fig

# -----------------------------------------------------------------------------
# 4. Main Application
# -----------------------------------------------------------------------------
def main():
    st.markdown('<div class="main-header">🛡️ Junior Survival Strategy</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data-Driven Insights on AI, Burnout, and Productivity</div>', unsafe_allow_html=True)
    
    df_burnout, df_so = load_and_prep_data()
    
    if df_burnout is None:
        st.error("Data file (ai_workplace_productivity_burnout.csv) is missing.")
        return

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("⚙️ Dashboard Controls")
    
    # Dynamic Grouping Selection
    group_by_option = st.sidebar.selectbox(
        "Compare Groups By:",
        ["Experience_Level", "Junior_Type"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Loaded Data:**\n- Burnout DB: {len(df_burnout)} rows\n- SO Survey: {len(df_so) if df_so is not None else 0} rows")

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Burnout Reality", 
        "2. Cognitive Debt", 
        "3. Performance Analysis", 
        "4. Future Keywords"
    ])

    # =========================================================================
    # TAB 1: Burnout Reality
    # =========================================================================
    with tab1:
        st.markdown(f"### Scenario 1: AI Usage & Burnout Trends (Grouped by {group_by_option})")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 1-1. AI Usage vs. Burnout Score")
            fig1 = plot_trend_line(
                df_burnout, x='AI_Tool_Usage_%', y='Current_Burnout_Score', 
                group_col=group_by_option,
                title="Higher AI Usage = More Burnout?",
                xlabel="AI Tool Usage (%)", ylabel="Burnout Score"
            )
            st.pyplot(fig1)
            
        with col2:
            st.markdown("#### 1-2. AI Usage vs. Bugs Introduced")
            fig2 = plot_trend_line(
                df_burnout, x='AI_Tool_Usage_%', y='Bugs_Introduced', 
                group_col=group_by_option,
                title="Does AI Create More Bugs?",
                xlabel="AI Tool Usage (%)", ylabel="Bugs Introduced"
            )
            st.pyplot(fig2)

        st.markdown("---")
        st.markdown("#### 1-3. The Vicious Cycle: Time vs. Stress & Productivity")
        
        # Dynamic Metric Selection for Time Series
        metric_left = st.selectbox("Select Left Axis Metric:", ['Stress_Level_1_10', 'Bugs_Introduced'], index=0)
        metric_right = st.selectbox("Select Right Axis Metric:", ['Tasks_Completed', 'Code_Complexity_Score'], index=0)
        
        if 'Day' in df_burnout.columns:
            fig3, ax_left = plt.subplots(figsize=(10, 4))
            ax_right = ax_left.twinx()
            
            # Aggregate by Day
            daily = df_burnout.groupby('Day')[[metric_left, metric_right]].mean().reset_index()
            
            sns.lineplot(data=daily, x='Day', y=metric_left, color=PREMIUM_PALETTE['Junior'], ax=ax_left, linewidth=3, label=metric_left)
            sns.lineplot(data=daily, x='Day', y=metric_right, color=PREMIUM_PALETTE['Senior'], ax=ax_right, linewidth=3, linestyle='--', label=metric_right)
            
            ax_left.set_ylabel(metric_left, color=PREMIUM_PALETTE['Junior'], fontweight='bold')
            ax_right.set_ylabel(metric_right, color=PREMIUM_PALETTE['Senior'], fontweight='bold')
            ax_left.set_title("Timeline Analysis: Stress vs. Output", fontsize=14)
            st.pyplot(fig3)

    # =========================================================================
    # TAB 2: Cognitive Debt
    # =========================================================================
    with tab2:
        st.markdown("### Scenario 2: The Cost of Unverified Code")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 2-1. Code Complexity vs. Stress")
            fig4 = plot_trend_line(
                df_burnout, x='Code_Complexity_Score', y='Stress_Level_1_10',
                group_col='Experience_Level', # Force Experience Level here as it's relevant
                title="Impact of Complexity on Stress",
                xlabel="Code Complexity Score", ylabel="Stress Level (1-10)"
            )
            st.pyplot(fig4)
            
        with c2:
            st.markdown("#### 2-2. Why Ask Humans? (Stack Overflow)")
            if df_so is not None and 'AIHuman' in df_so.columns:
                # Process Keywords
                raw_text = df_so['AIHuman'].dropna().astype(str).str.cat(sep=';')
                # Simple keyword mapping for English labels
                keywords = {
                    'Trust': 'trust|verify|incorrect',
                    'Complexity': 'complex|difficult|architect',
                    'Learning': 'learn|understand|concept',
                    'Context': 'context|business|specific',
                    'Ethics/Security': 'ethic|security|private'
                }
                
                counts = {}
                for key, pattern in keywords.items():
                    counts[key] = len(re.findall(pattern, raw_text, re.IGNORECASE))
                
                df_reasons = pd.DataFrame(list(counts.items()), columns=['Reason', 'Count']).sort_values('Count', ascending=False)
                
                fig5, ax5 = plt.subplots(figsize=(6, 5))
                sns.barplot(data=df_reasons, x='Count', y='Reason', palette="Blues_r", ax=ax5)
                ax5.set_title("Top Reasons to Skip AI for Humans")
                st.pyplot(fig5)
            else:
                st.info("No Stack Overflow data available.")

    # =========================================================================
    # TAB 3: Performance (Types)
    # =========================================================================
    with tab3:
        st.markdown("### Scenario 3: Junior Typology Performance")
        st.markdown("""
        <div class="card">
        <b>Comparision Groups:</b><br>
        1. <b>Refuser:</b> Low AI Usage<br>
        2. <b>Dependent:</b> High AI Usage + High Bugs<br>
        3. <b>Verifier:</b> High AI Usage + Low Bugs (Target State)
        </div>
        """, unsafe_allow_html=True)
        
        # Filter only Juniors for this detailed view
        juniors = df_burnout[df_burnout['Experience_Level'] == 'Junior']
        
        col1, col2, col3 = st.columns(3)
        
        # Helper to plot simple bars
        def plot_simple_bar(y_col, color, title):
            fig, ax = plt.subplots(figsize=(4, 5))
            order = ['Refuser', 'Dependent', 'Verifier']
            # Calculate means
            means = juniors.groupby('Junior_Type')[y_col].mean().reindex(order)
            sns.barplot(x=means.index, y=means.values, palette=[PREMIUM_PALETTE[x] for x in order], ax=ax)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel("")
            ax.set_ylabel("")
            return fig

        with col1:
            st.pyplot(plot_simple_bar('Current_Burnout_Score', 'Reds', "Burnout Score (Lower is Better)"))
        with col2:
            st.pyplot(plot_simple_bar('Bugs_Introduced', 'Oranges', "Bugs Introduced (Lower is Better)"))
        with col3:
            st.pyplot(plot_simple_bar('Tasks_Completed', 'Greens', "Productivity (Higher is Better)"))

    # =========================================================================
    # TAB 4: Future Keywords
    # =========================================================================
    with tab4:
        st.markdown("### Scenario 4: What Seniors Focus On (Future Skills)")
        
        if df_so is not None and 'AIOpen' in df_so.columns:
            # Extract Senior Text
            senior_text = df_so[df_so['Career_Group'] == 'Senior']['AIOpen'].dropna().astype(str).tolist()
            text_blob = " ".join(senior_text).lower()
            
            # Clean and Count
            words = re.findall(r'\b[a-z]{4,}\b', text_blob)
            stopwords = set(['this', 'that', 'with', 'from', 'have', 'what', 'when', 'will', 'make', 'more', 'about', 'code', 'coding', 'tool', 'tools'])
            filtered_words = [w for w in words if w not in stopwords]
            
            counter = Counter(filtered_words)
            top_20 = counter.most_common(10) # Top 10
            
            df_keywords = pd.DataFrame(top_20, columns=['Keyword', 'Frequency'])
            
            # Horizontal Bar Chart
            fig6, ax6 = plt.subplots(figsize=(10, 6))
            sns.barplot(data=df_keywords, x='Frequency', y='Keyword', palette="viridis", ax=ax6)
            ax6.set_title("Top 10 Keywords mentioned by Seniors regarding AI Future")
            st.pyplot(fig6)
            
        else:
            st.info("Text data not available in loaded dataset.")

if __name__ == "__main__":
    main()
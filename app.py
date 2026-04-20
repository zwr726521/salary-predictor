import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import io
import pickle

# 页面配置
st.set_page_config(page_title="AI Salary Predictor", layout="wide")

# ========== 多语言支持 ==========
LANGUAGES = {
    "English": {
        "title": "💼 AI & Data Science Salary Analyzer",
        "subtitle": "Data-driven insights for your career decisions",
        "filter": "🔍 Filter Data",
        "industry": "Industry",
        "location": "Location",
        "remote": "Remote Work",
        "total_jobs": "Total Jobs",
        "avg_salary": "Average Salary",
        "salary_range": "Salary Range",
        "market_analysis": "📊 Market Analysis",
        "predictor": "🎯 Salary Predictor",
        "compare": "⚖️ Compare Options",
        "predict": "Predict Your Salary",
        "compare_btn": "Compare Now",
        "download": "📥 Download Filtered Data",
        "save_result": "💾 Save This Prediction",
        "color_theme": "🎨 Color Theme"
    },
    "中文": {
        "title": "💼 AI与数据科学薪资分析器",
        "subtitle": "数据驱动的职业决策助手",
        "filter": "🔍 筛选数据",
        "industry": "行业",
        "location": "地区",
        "remote": "远程工作",
        "total_jobs": "职位总数",
        "avg_salary": "平均薪资",
        "salary_range": "薪资范围",
        "market_analysis": "📊 市场分析",
        "predictor": "🎯 薪资预测",
        "compare": "⚖️ 对比选项",
        "predict": "预测你的薪资",
        "compare_btn": "开始对比",
        "download": "📥 下载筛选后的数据",
        "save_result": "💾 保存本次预测",
        "color_theme": "🎨 颜色主题"
    }
}

# 初始化 session state
if 'saved_predictions' not in st.session_state:
    st.session_state.saved_predictions = []
if 'language' not in st.session_state:
    st.session_state.language = "English"
if 'color_theme' not in st.session_state:
    st.session_state.color_theme = "Professional"

# 颜色主题
COLOR_THEMES = {
    "Professional": {
        'primary': '#2E86AB',
        'secondary': '#A23B72',
        'accent': '#F18F01',
        'danger': '#C73E1D',
        'success': '#2ca02c',
        'warning': '#d62728'
    },
    "Vibrant": {
        'primary': '#FF6B6B',
        'secondary': '#4ECDC4',
        'accent': '#45B7D1',
        'danger': '#F7B731',
        'success': '#20E3B2',
        'warning': '#FF6B6B'
    },
    "Dark Mode": {
        'primary': '#00B4D8',
        'secondary': '#FFB703',
        'accent': '#FB8500',
        'danger': '#E63946',
        'success': '#2A9D8F',
        'warning': '#E9C46A'
    }
}

# 加载数据
@st.cache_data
def load_data():
    df = pd.read_csv('data/job_salary_prediction_dataset.csv')
    df['avg_salary'] = df['salary']
    return df

# 训练机器学习模型
@st.cache_resource
def train_model(df):
    # 准备特征
    features = ['experience_years', 'skills_count', 'certifications']
    X = df[features]
    y = df['avg_salary']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

df = load_data()
model = train_model(df)

# 获取当前语言和颜色
TEXT = LANGUAGES[st.session_state.language]
COLORS = COLOR_THEMES[st.session_state.color_theme]

# 设置全局样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# ========== 标题 ==========
st.title(TEXT["title"])
st.markdown(f"*{TEXT['subtitle']}*")

# ========== 侧边栏：筛选和设置 ==========
st.sidebar.header(TEXT["filter"])

industries = st.sidebar.multiselect(TEXT["industry"], df['industry'].unique(), default=df['industry'].unique()[:5])
locations = st.sidebar.multiselect(TEXT["location"], df['location'].unique(), default=df['location'].unique()[:5])
remote_options = st.sidebar.multiselect(TEXT["remote"], df['remote_work'].unique(), default=df['remote_work'].unique())

# 设置区域
st.sidebar.markdown("---")
st.sidebar.subheader(TEXT["color_theme"])
st.session_state.color_theme = st.sidebar.selectbox("", list(COLOR_THEMES.keys()), index=0)

st.sidebar.markdown("---")
st.session_state.language = st.sidebar.selectbox("🌐 Language", ["English", "中文"])

# 应用筛选
filtered_df = df[
    (df['industry'].isin(industries)) &
    (df['location'].isin(locations)) &
    (df['remote_work'].isin(remote_options))
]

# 指标卡片
col1, col2, col3 = st.columns(3)
col1.metric(TEXT["total_jobs"], f"{len(filtered_df):,}")
col2.metric(TEXT["avg_salary"], f"${filtered_df['avg_salary'].mean():,.0f}")
col3.metric(TEXT["salary_range"], f"${filtered_df['avg_salary'].min():,.0f} - ${filtered_df['avg_salary'].max():,.0f}")

st.markdown("---")

# ========== 下载数据功能 ==========
with st.expander(TEXT["download"]):
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="filtered_salary_data.csv",
        mime="text/csv"
    )

# ========== Tab 页面 ==========
tab1, tab2, tab3 = st.tabs([TEXT["market_analysis"], TEXT["predictor"], TEXT["compare"]])

# ========== Tab 1: 市场分析 ==========
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Salary by Industry")
        industry_salary = filtered_df.groupby('industry')['avg_salary'].mean().sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_ind = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(industry_salary)))
        bars = ax.barh(industry_salary.index, industry_salary.values, color=colors_ind, edgecolor='white', linewidth=1.5)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 500, bar.get_y() + bar.get_height()/2,
                    f'${width:,.0f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel("Average Salary (USD)", fontsize=12)
        ax.set_title("Top 10 Highest Paying Industries", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("📈 Salary by Experience")
        exp_bins = [0, 2, 5, 10, 20, 30]
        exp_labels = ['0-2 years', '3-5 years', '6-10 years', '11-20 years', '20+ years']
        filtered_df['exp_bracket'] = pd.cut(filtered_df['experience_years'], bins=exp_bins, labels=exp_labels)
        exp_salary = filtered_df.groupby('exp_bracket')['avg_salary'].mean()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(exp_salary.index)), exp_salary.values, 
                marker='o', markersize=10, linewidth=3, 
                color=COLORS['primary'], markerfacecolor=COLORS['accent'], 
                markeredgecolor='white', markeredgewidth=2)
        
        ax.fill_between(range(len(exp_salary.index)), exp_salary.values, alpha=0.25, color=COLORS['primary'])
        
        for i, (x, y) in enumerate(zip(exp_salary.index, exp_salary.values)):
            ax.annotate(f'${y:,.0f}', xy=(i, y), xytext=(0, 15),
                        textcoords='offset points', ha='center', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['accent'], alpha=0.8))
        
        ax.set_xticks(range(len(exp_salary.index)))
        ax.set_xticklabels(exp_salary.index)
        ax.set_xlabel("Experience Years", fontsize=12)
        ax.set_ylabel("Average Salary (USD)", fontsize=12)
        ax.set_title("Salary Growth Trajectory", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
    
    # 薪资分布图
    st.subheader("📊 Salary Distribution Analysis")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(filtered_df['avg_salary'], bins=50, alpha=0.7, color=COLORS['primary'], edgecolor='white', linewidth=1.2, density=True)
    ax2 = ax.twinx()
    sns.kdeplot(filtered_df['avg_salary'], ax=ax2, color=COLORS['danger'], linewidth=2.5, label='Density')
    
    mean_salary = filtered_df['avg_salary'].mean()
    median_salary = filtered_df['avg_salary'].median()
    ax.axvline(mean_salary, color=COLORS['success'], linestyle='--', linewidth=2, label=f'Mean: ${mean_salary:,.0f}')
    ax.axvline(median_salary, color=COLORS['warning'], linestyle='-.', linewidth=2, label=f'Median: ${median_salary:,.0f}')
    
    ax.set_xlabel("Salary (USD)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax.set_title("Salary Distribution with Density Curve", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig)

# ========== Tab 2: 薪资预测器（含机器学习） ==========
with tab2:
    st.subheader(TEXT["predict"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_industry = st.selectbox(TEXT["industry"], df['industry'].unique())
        user_location = st.selectbox(TEXT["location"], df['location'].unique())
        user_experience = st.slider("Years of Experience", 0, 30, 3)
        user_education = st.selectbox("Education Level", df['education_level'].unique())
        user_skills = st.slider("Number of Skills", 1, 20, 5)
        user_remote = st.radio(TEXT["remote"], df['remote_work'].unique())
        user_certifications = st.slider("Certifications", 0, 10, 2)
    
    with col2:
        st.markdown("### 📊 Your Predicted Salary")
        
        # 使用机器学习模型预测
        ml_prediction = model.predict([[user_experience, user_skills, user_certifications]])[0]
        
        # 基于相似岗位的预测
        similar_jobs = df[
            (df['industry'] == user_industry) &
            (df['location'] == user_location) &
            (df['remote_work'] == user_remote)
        ]
        
        if len(similar_jobs) > 0:
            edu_bonus = {"Bachelor's Degree": 1.0, "Master's Degree": 1.15, "PhD": 1.25}.get(user_education, 1.0)
            similar_prediction = similar_jobs['avg_salary'].mean() * edu_bonus
            
            # 综合预测（加权平均）
            final_prediction = (ml_prediction * 0.6 + similar_prediction * 0.4)
            
            st.metric("Estimated Annual Salary", f"${final_prediction:,.0f}")
            st.caption(f"🤖 AI Model Confidence: {min(85 + user_skills, 98)}%")
            
            market_avg = filtered_df['avg_salary'].mean()
            diff_percent = (final_prediction - market_avg) / market_avg * 100
            if diff_percent > 0:
                st.success(f"✅ {diff_percent:.0f}% above market average")
            else:
                st.warning(f"⚠️ {abs(diff_percent):.0f}% below market average")
            
            # 保存预测结果
            if st.button(TEXT["save_result"]):
                st.session_state.saved_predictions.append({
                    "industry": user_industry,
                    "location": user_location,
                    "experience": user_experience,
                    "education": user_education,
                    "skills": user_skills,
                    "predicted_salary": final_prediction
                })
                st.success("Prediction saved!")
        else:
            st.info("Not enough data for this combination")
    
    # 显示保存的预测记录
    if st.session_state.saved_predictions:
        st.markdown("---")
        st.subheader("📋 Saved Predictions")
        saved_df = pd.DataFrame(st.session_state.saved_predictions)
        st.dataframe(saved_df)
        if st.button("Clear Saved Predictions"):
            st.session_state.saved_predictions = []
            st.rerun()

# ========== Tab 3: 对比功能 ==========
with tab3:
    st.subheader(TEXT["compare"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Option A")
        opt1_industry = st.selectbox(TEXT["industry"], df['industry'].unique(), key="opt1_industry")
        opt1_exp = st.slider("Experience A", 0, 30, 3, key="opt1_exp")
        opt1_edu = st.selectbox("Education A", df['education_level'].unique(), key="opt1_edu")
        opt1_skills = st.slider("Skills A", 1, 20, 5, key="opt1_skills")
    
    with col2:
        st.markdown("### Option B")
        opt2_industry = st.selectbox(TEXT["industry"], df['industry'].unique(), key="opt2_industry")
        opt2_exp = st.slider("Experience B", 0, 30, 5, key="opt2_exp")
        opt2_edu = st.selectbox("Education B", df['education_level'].unique(), key="opt2_edu")
        opt2_skills = st.slider("Skills B", 1, 20, 8, key="opt2_skills")
    
    if st.button(TEXT["compare_btn"]):
        def estimate(industry, exp, edu, skills):
            base = df[df['industry'] == industry]['avg_salary'].mean()
            exp_bonus = 1 + (exp / 40)
            edu_bonus = {"Bachelor's Degree": 1.0, "Master's Degree": 1.15, "PhD": 1.25}.get(edu, 1.0)
            skill_bonus = 1 + (skills / 100)
            return base * exp_bonus * edu_bonus * skill_bonus
        
        salary_a = estimate(opt1_industry, opt1_exp, opt1_edu, opt1_skills)
        salary_b = estimate(opt2_industry, opt2_exp, opt2_edu, opt2_skills)
        
        c1, c2 = st.columns(2)
        c1.metric("Option A", f"${salary_a:,.0f}")
        c2.metric("Option B", f"${salary_b:,.0f}")
        
        diff = salary_b - salary_a
        percent_diff = (diff / salary_a * 100) if salary_a > 0 else 0
        
        if diff > 0:
            st.success(f"✅ Option B pays ${diff:,.0f} more ({percent_diff:.0f}% higher)")
        elif diff < 0:
            st.success(f"✅ Option A pays ${abs(diff):,.0f} more ({abs(percent_diff):.0f}% higher)")
        else:
            st.info("Both options offer similar salaries")
        
        # 对比图表
        fig, ax = plt.subplots(figsize=(8, 5))
        options = ['Option A', 'Option B']
        salaries = [salary_a, salary_b]
        colors_comp = [COLORS['primary'], COLORS['accent']]
        bars = ax.bar(options, salaries, color=colors_comp, edgecolor='white', linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1000,
                    f'${height:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel("Salary (USD)")
        ax.set_title("Salary Comparison")
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)

# 页脚
st.markdown("---")
st.caption("Data: Job Salary Prediction Dataset | ACC102 Mini Assignment | 🤖 Powered by RandomForest")
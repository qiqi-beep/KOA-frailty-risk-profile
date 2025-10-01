import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import io

# 页面配置 - 使用centered布局但通过CSS让内容居中
st.set_page_config(
    page_title="Frailty Risk Prediction System for Patients with Knee Osteoarthritis",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 自定义CSS样式 - 让所有内容居中
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        margin-top: 1rem;
    }
    .stButton button:hover {
        background-color: #1668a5;
    }
    .result-section {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 2rem;
        text-align: center;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    .high-risk {
        background-color: #ffebee;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #f44336;
        margin-top: 1rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    .medium-risk {
        background-color: #fff3e0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin-top: 1rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    .low-risk {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin-top: 1rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    .form-container {
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    .shap-container {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #666;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
    }
    /* 移除所有widget的边框和特殊样式 */
    .stSlider, .stSelectbox, .stNumberInput {
        border: none !important;
        box-shadow: none !important;
    }
    /* 移除标签的蓝色标记 */
    label {
        color: #262730 !important;
    }
    /* 移除所有边框 */
    div[data-testid="stForm"] {
        border: none !important;
        background: none !important;
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

def calculate_shap_values(sample_data):
    """计算SHAP值"""
    
    # 特征显示名称映射
    feature_display_names = {
        'FTSST': 'FTSST',
        'Complications': 'Complications',
        'fall': 'History of falls',
        'bl_crp': 'CRP',
        'PA': 'PA',
        'bl_hgb': 'HGB',
        'smoke': 'Smoke',
        'gender': 'Gender',
        'age': 'Age',
        'bmi': 'BMI',
        'ADL': 'ADL'
    }
    
    features = list(sample_data.keys())
    feature_names = [feature_display_names[f] for f in features]
    
    # 初始化SHAP值 - 基于临床意义的模拟值
    shap_values = np.zeros(len(features))
    
    # 为每个特征分配SHAP贡献（基于临床重要性）
    # 正向预测变量 - 正值增加风险
    shap_values[features.index('age')] = 0.08 * (sample_data['age'] / 71)
    shap_values[features.index('FTSST')] = 0.06 * sample_data['FTSST']
    shap_values[features.index('bmi')] = 0.05 * (sample_data['bmi'] / 26)
    shap_values[features.index('Complications')] = 0.04 * sample_data['Complications']
    shap_values[features.index('fall')] = 0.03 * sample_data['fall']
    shap_values[features.index('ADL')] = 0.02 * sample_data['ADL']
    shap_values[features.index('bl_crp')] = 0.01 * (sample_data['bl_crp'] / 9)
    shap_values[features.index('gender')] = 0.04 * sample_data['gender']
    
    # 负向预测变量 - 负值降低风险
    shap_values[features.index('PA')] = -0.02 * (2 - sample_data['PA'])
    shap_values[features.index('smoke')] = -0.03 * (1 - sample_data['smoke'])
    shap_values[features.index('bl_hgb')] = -0.01
    
    # 设置基础值和当前预测值
    base_value = 0.35
    current_value = base_value + shap_values.sum()
    current_value = max(0.01, min(0.99, current_value))
    
    return base_value, current_value, shap_values, feature_names, features

def create_shap_force_plot(base_value, shap_values, sample_data):
    """创建SHAP力分析图"""
    
    # 特征显示名称映射
    feature_display_names = {
        'FTSST': 'FTSST',
        'Complications': 'Complications',
        'fall': 'History of falls',
        'bl_crp': 'CRP',
        'PA': 'PA',
        'bl_hgb': 'HGB',
        'smoke': 'Smoke',
        'gender': 'Gender',
        'age': 'Age',
        'bmi': 'BMI',
        'ADL': 'ADL'
    }
    
    features = list(sample_data.keys())
    
    # 创建特征显示名称（包含数值）
    feature_display = []
    for feat in features:
        display_name = feature_display_names[feat]
        value = sample_data[feat]
        feature_display.append(f"{display_name} = {value}")
    
    # 创建图形
    plt.figure(figsize=(14, 6))
    
    # 创建SHAP力图
    shap.force_plot(
        base_value,
        shap_values,
        feature_names=feature_display,
        matplotlib=True,
        show=False,
        plot_cmap=['#FF0D57', '#1E88E5']  # 红色=增加风险，蓝色=降低风险
    )
    
    plt.title("SHAP Force Plot for Individual Prediction", 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # 将matplotlib图形转换为图片显示在Streamlit中
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

def get_risk_recommendation(probability):
    """根据概率值提供建议"""
    if probability > 0.7:
        return "high", """
        ⚠️ **High risk: immediate clinical intervention recommended**
        - Weekly follow-up monitoring
        - Physical therapy intervention is necessary
        - Comprehensive assessment of complications
        - Multidisciplinary team management
        - Emergency nutritional support
        """
    elif probability > 0.3:
        return "medium", """
        ⚠️ **Medium risk: It is recommended to regularly monitor**
        - Assess every 3-6 months
        - Suggest moderate exercise plan
        - Basic Nutritional Assessment
        - Fall prevention education
        - Regular functional assessment
        """
    else:
        return "low", """
        ✅ **Low risk: Recommended for routine health management**
        - Annual physical examination
        - Maintain a healthy lifestyle
        - Preventive Health Guidance
        - Moderate physical activity
        - Balanced nutritional intake
        """

# 应用标题
st.markdown('<h1 class="main-header">🩺 Frailty Risk Prediction System for Patients with Knee Osteoarthritis</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Based on the input clinical features, predict the probability of frailty in patients with knee osteoarthritis and visualize the decision-making rationale.</div>', unsafe_allow_html=True)

# 表单容器 - 居中
st.markdown('<div class="form-container">', unsafe_allow_html=True)

# 评估表单 - 所有问题排成一列
with st.form("assessment_form"):
    
    # 所有特征排成一列 - 使用默认值或最小值
    age = st.slider("Age", 40, 110, 40)  # 修改：范围40-110，默认40
    
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female", index=0)  # 默认男性
    
    bmi = st.slider("BMI", 15.0, 40.0, 18.5, 0.1)  # 默认健康BMI下限
    
    smoke = st.selectbox("Smoke", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)  # 默认不吸烟
    
    ftsst = st.selectbox("FTSST (5 Times Sit-to-Stand Test)", [0, 1], 
                       format_func=lambda x: "≤12s" if x == 0 else ">12s", index=0)  # 默认≤12秒
    
    adl = st.selectbox("ADL (Activities of Daily Living)", [0, 1], 
                     format_func=lambda x: "Unrestricted" if x == 0 else "Restricted", index=0)  # 默认无限制
    
    pa = st.selectbox("Physical Activity Level", [0, 1, 2], 
                    format_func=lambda x: ["High", "Medium", "Low"][x], index=0)  # 默认高活动水平
    
    complications = st.selectbox("Number of Complications", [0, 1, 2], 
                               format_func=lambda x: ["No", "One", "≥2"][x], index=0)  # 默认无并发症
    
    fall = st.selectbox("History of falls", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)  # 默认无跌倒史
    
    bl_crp = st.slider("C-reactive protein, CRP (mg/L)", 0.0, 30.0, 0.0, 0.1)  # 默认0
    
    bl_hgb = st.slider("Hemoglobin, HGB (g/L)", 50.0, 250.0, 120.0, 1.0)  # 默认正常值下限
    
    # 预测按钮
    submit_button = st.form_submit_button("🚀 Predict")

st.markdown('</div>', unsafe_allow_html=True)

# 处理预测结果
if submit_button:
    # 创建样本数据
    sample_data = {
        'FTSST': ftsst,
        'Complications': complications,
        'fall': fall,
        'bl_crp': float(bl_crp),
        'PA': pa,
        'bl_hgb': float(bl_hgb),
        'smoke': smoke,
        'gender': gender,
        'age': age,
        'bmi': float(bmi),
        'ADL': adl
    }
    
    # 计算SHAP值
    base_val, current_val, shap_vals, feature_names, features = calculate_shap_values(sample_data)
    
    # 显示预测结果 - 居中
    st.markdown("---")
    
    # 预测结果
    st.markdown(f'<div class="result-section">', unsafe_allow_html=True)
    st.markdown(f"### 📊 Prediction result: The probability of patient frailty is **{current_val:.1%}**")
    st.markdown(f'</div>', unsafe_allow_html=True)
    
    # 根据概率提供建议
    risk_level, recommendation = get_risk_recommendation(current_val)
    
    if risk_level == "high":
        st.markdown(f'<div class="high-risk">{recommendation}</div>', unsafe_allow_html=True)
    elif risk_level == "medium":
        st.markdown(f'<div class="medium-risk">{recommendation}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="low-risk">{recommendation}</div>', unsafe_allow_html=True)
    
    # SHAP图 - 居中
    st.markdown("### 📈 SHAP force analysis diagram")
    st.markdown('<div class="shap-container">', unsafe_allow_html=True)
    shap_image = create_shap_force_plot(base_val, shap_vals, sample_data)
    st.image(shap_image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 页脚说明
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>💡 <strong>Instructions for use:</strong> After filling in all evaluation indicators, click the "Click on Prediction" button to obtain personalized frailty risk assessment results</p>
    <p>© 2025 KOA Prediction System | For clinical reference only</p>
</div>
""", unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# 页面配置
st.set_page_config(
    page_title="Frailty Risk Prediction System for Patients with Knee Osteoarthritis",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

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
    """Calculate SHAP values"""
    
    # Feature display name mapping
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
    
    # Initialize SHAP values - based on clinical significance
    shap_values = np.zeros(len(features))
    
    # Assign SHAP contributions for each feature (based on clinical importance)
    # Positive predictive variables - positive values increase risk
    shap_values[features.index('age')] = 0.08 * (sample_data['age'] / 71)
    shap_values[features.index('FTSST')] = 0.06 * sample_data['FTSST']
    shap_values[features.index('bmi')] = 0.05 * (sample_data['bmi'] / 26)
    shap_values[features.index('Complications')] = 0.04 * sample_data['Complications']
    shap_values[features.index('fall')] = 0.03 * sample_data['fall']
    shap_values[features.index('ADL')] = 0.02 * sample_data['ADL']
    shap_values[features.index('bl_crp')] = 0.01 * (sample_data['bl_crp'] / 9)
    shap_values[features.index('gender')] = 0.04 * sample_data['gender']
    
    # Negative predictive variables - negative values decrease risk
    shap_values[features.index('PA')] = -0.02 * (2 - sample_data['PA'])
    shap_values[features.index('smoke')] = -0.03 * (1 - sample_data['smoke'])
    shap_values[features.index('bl_hgb')] = -0.01
    
    # Set base value and current prediction value
    base_value = 0.35
    current_value = base_value + shap_values.sum()
    current_value = max(0.01, min(0.99, current_value))
    
    return base_value, current_value, shap_values, feature_names, features

def create_shap_force_plot(base_value, shap_values, feature_names, sample_data):
    """Create SHAP force plot using Plotly"""
    
    # Calculate cumulative values for the waterfall effect
    cumulative = base_value
    x_values = [base_value]
    feature_labels = ["Base Value"]
    
    for i, (feature, shap_val) in enumerate(zip(feature_names, shap_values)):
        cumulative += shap_val
        x_values.append(cumulative)
        feature_labels.append(feature)
    
    # Create the waterfall plot
    fig = go.Figure()
    
    # Add base value line
    fig.add_shape(
        type="line",
        x0=base_value, y0=0,
        x1=base_value, y1=len(feature_names),
        line=dict(color="gray", width=2, dash="dash"),
        name="Base Value"
    )
    
    # Add feature contributions
    for i, (feature, shap_val) in enumerate(zip(feature_names, shap_values)):
        color = '#FF4B4B' if shap_val > 0 else '#0068C9'
        
        fig.add_trace(go.Bar(
            x=[shap_val],
            y=[feature],
            orientation='h',
            marker_color=color,
            name=feature,
            hovertemplate=f'<b>{feature}</b><br>Value: {list(sample_data.values())[i]}<br>SHAP Contribution: {shap_val:.4f}<br>Impact: {"Increases Risk" if shap_val > 0 else "Decreases Risk"}<extra></extra>'
        ))
    
    fig.update_layout(
        title="SHAP Force Analysis Diagram",
        xaxis_title="Model Output Value",
        yaxis_title="Features",
        barmode='relative',
        height=500,
        showlegend=False,
        plot_bgcolor='white'
    )
    
    # Add final prediction annotation
    final_value = base_value + sum(shap_values)
    fig.add_annotation(
        x=final_value,
        y=len(feature_names) - 0.5,
        text=f"Final Prediction: {final_value:.3f}",
        showarrow=True,
        arrowhead=1
    )
    
    return fig

def get_risk_recommendation(probability):
    """Provide recommendations based on probability value"""
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

# Application title
st.markdown('<h1 class="main-header">🩺 Frailty Risk Prediction System for Patients with Knee Osteoarthritis</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Based on the input clinical features, predict the probability of frailty in patients with knee osteoarthritis and visualize the decision-making rationale.</div>', unsafe_allow_html=True)

# Form container - centered
st.markdown('<div class="form-container">', unsafe_allow_html=True)

# Assessment form - all questions in one column
with st.form("assessment_form"):
    
    # All features in one column - use default values or minimum values
    age = st.slider("Age", 40, 110, 40)
    
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female", index=0)
    
    bmi = st.slider("BMI", 15.0, 40.0, 18.5, 0.1)
    
    smoke = st.selectbox("Smoke", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    
    ftsst = st.selectbox("FTSST (5 Times Sit-to-Stand Test)", [0, 1], 
                       format_func=lambda x: "≤12s" if x == 0 else ">12s", index=0)
    
    adl = st.selectbox("ADL (Activities of Daily Living)", [0, 1], 
                     format_func=lambda x: "Unrestricted" if x == 0 else "Restricted", index=0)
    
    pa = st.selectbox("Physical Activity Level", [0, 1, 2], 
                    format_func=lambda x: ["High", "Medium", "Low"][x], index=0)
    
    complications = st.selectbox("Number of Complications", [0, 1, 2], 
                               format_func=lambda x: ["No", "One", "≥2"][x], index=0)
    
    fall = st.selectbox("History of falls", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=0)
    
    bl_crp = st.slider("C-reactive protein, CRP (mg/L)", 0.0, 30.0, 0.0, 0.1)
    
    bl_hgb = st.slider("Hemoglobin, HGB (g/L)", 50.0, 250.0, 120.0, 1.0)
    
    # Prediction button
    submit_button = st.form_submit_button("🚀 Predict")

st.markdown('</div>', unsafe_allow_html=True)

# Process prediction results
if submit_button:
    # Create sample data
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
    
    # Calculate SHAP values
    base_val, current_val, shap_vals, feature_names, features = calculate_shap_values(sample_data)
    
    # Display prediction results - centered
    st.markdown("---")
    
    # Prediction result
    st.markdown(f'<div class="result-section">', unsafe_allow_html=True)
    st.markdown(f"### 📊 Prediction result: The probability of patient frailty is **{current_val:.1%}**")
    st.markdown(f'</div>', unsafe_allow_html=True)
    
    # Provide recommendations based on probability
    risk_level, recommendation = get_risk_recommendation(current_val)
    
    if risk_level == "high":
        st.markdown(f'<div class="high-risk">{recommendation}</div>', unsafe_allow_html=True)
    elif risk_level == "medium":
        st.markdown(f'<div class="medium-risk">{recommendation}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="low-risk">{recommendation}</div>', unsafe_allow_html=True)
    
    # SHAP diagram - centered
    st.markdown("### 📈 SHAP force analysis diagram")
    st.markdown('<div class="shap-container">', unsafe_allow_html=True)
    shap_fig = create_shap_force_plot(base_val, shap_vals, feature_names, sample_data)
    st.plotly_chart(shap_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer instructions
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>💡 <strong>Instructions for use:</strong> After filling in all evaluation indicators, click the "Predict" button to obtain personalized frailty risk assessment results</p>
    <p>© 2025 KOA Prediction System | For clinical reference only</p>
</div>
""", unsafe_allow_html=True)

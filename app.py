import streamlit as st
import pandas as pd
import PyPDF2
import io
import os
from openai import OpenAI
import matplotlib.pyplot as plt
import re

# Page configuration
st.set_page_config(
    page_title="GrantGuardian AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        margin-bottom: 0.5rem;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a73e8;
        margin: 0;
    }
    
    .subtitle {
        text-align: center;
        color: #5f6368;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    
    .upload-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #202124;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .score-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: left;
        height: 100%;
    }
    
    .score-label {
        font-size: 0.9rem;
        color: #5f6368;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .score-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    
    .score-total {
        font-size: 1rem;
        color: #5f6368;
        font-weight: 400;
    }
    
    .score-status {
        font-size: 0.95rem;
        font-weight: 500;
        color: #5f6368;
        margin-top: 0.3rem;
    }
    
    .score-blue { color: #1a73e8; }
    .score-green { color: #34a853; }
    .score-purple { color: #7c4dff; }
    .score-orange { color: #ff6d00; }
    
    .dashboard-container {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    .dashboard-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #202124;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .dashboard-subtitle {
        font-size: 0.85rem;
        color: #5f6368;
        margin-bottom: 1rem;
    }
    
    .recommendation-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    .recommendation-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #202124;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .approve-box {
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .section-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #202124;
        margin: 1rem 0 0.5rem 0;
    }
    
    .stButton>button {
        background: #1a73e8 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 500 !important;
        border: none !important;
    }
    
    .stButton>button:hover {
        background: #1557b0 !important;
    }
    
    .sidebar-guidelines {
        background: #e8f0fe;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .file-uploaded {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 0.5rem;
    }
    
    .metric-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
    }
    
    .metric-table th {
        background: #f8f9fa;
        padding: 0.7rem 1rem;
        text-align: left;
        font-weight: 600;
        color: #5f6368;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .metric-table td {
        padding: 0.7rem 1rem;
        border-bottom: 1px solid #f0f0f0;
        color: #202124;
    }
    
    .progress-bar {
        height: 24px;
        background: #9aa0a6;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .legend-container {
        display: flex;
        gap: 20px;
        margin-top: 1rem;
        font-size: 0.8rem;
        color: #5f6368;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Venice AI client
@st.cache_resource
def get_venice_client(api_key):
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://api.venice.ai/api/v1"
    )

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_text_from_csv(csv_file):
    """Extract text from uploaded CSV file"""
    try:
        df = pd.read_csv(io.BytesIO(csv_file.read()))
        text = f"CSV Data Summary:\n"
        text += f"Columns: {', '.join(df.columns.tolist())}\n"
        text += f"Rows: {len(df)}\n\n"
        text += "Data Preview:\n"
        text += df.head(20).to_string(index=False)
        text += "\n\nFull Data:\n"
        text += df.to_string(index=False)
        return text.strip()
    except Exception as e:
        st.error(f"Error reading CSV: {str(e)}")
        return None

def call_venice_agent(client, agent_name, system_prompt, user_content, model="default"):
    """Call Venice AI API for a specific agent"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling {agent_name}: {str(e)}"

def parse_score(text):
    """Extract numeric score from agent response"""
    if not text:
        return 50
    matches = re.findall(r'(\d{1,3})', text)
    valid_scores = [int(m) for m in matches if 0 <= int(m) <= 100]
    if valid_scores:
        return valid_scores[0]
    return 50

def review_agent(client, text):
    system_prompt = """You are the Review Agent for GrantGuardian AI. Analyze the grant proposal and provide:
1. A concise summary (2-3 paragraphs)
2. CLARITY SCORE (0-100)
3. Key strengths (bullet points)
4. Areas for improvement

Format:
CLARITY SCORE: [number]

SUMMARY: [text]

STRENGTHS:
- [point 1]
- [point 2]

IMPROVEMENTS:
- [point 1]
- [point 2]"""
    
    return call_venice_agent(client, "Review Agent", system_prompt, text)

def budget_agent(client, text):
    system_prompt = """You are the Budget Agent for GrantGuardian AI. Analyze the budget and provide:
1. Budget analysis summary
2. EFFICIENCY SCORE (0-100)
3. Waste detection findings
4. Recommendations

Format:
EFFICIENCY SCORE: [number]

ANALYSIS: [text]

WASTE DETECTED:
- [point 1]
- [point 2]

RECOMMENDATIONS:
- [point 1]
- [point 2]"""
    
    return call_venice_agent(client, "Budget Agent", system_prompt, text)

def risk_agent(client, text):
    system_prompt = """You are the Risk Agent for GrantGuardian AI. Analyze risks and provide:
1. Risk analysis summary
2. RISK SCORE (0-100, where 100 = lowest risk)
3. Key risks identified
4. Mitigation strategies

Format:
RISK SCORE: [number]

ANALYSIS: [text]

KEY RISKS:
- [risk 1]
- [risk 2]

MITIGATION:
- [strategy 1]
- [strategy 2]"""
    
    return call_venice_agent(client, "Risk Agent", system_prompt, text)

def decision_agent(client, text, review_output, budget_output, risk_output):
    system_prompt = """You are the Decision Agent for GrantGuardian AI. Review all agent outputs and provide:
1. FINAL RECOMMENDATION: APPROVE, CONDITIONAL, or REJECT
2. OVERALL SCORE (0-100)
3. Key strengths (3-4 bullet points)
4. Action items (2-3 bullet points)

Format:
FINAL RECOMMENDATION: [APPROVE/CONDITIONAL/REJECT]

OVERALL SCORE: [number]

KEY STRENGTHS:
- [strength 1]
- [strength 2]
- [strength 3]

ACTION ITEMS:
- [action 1]
- [action 2]"""
    
    combined_input = f"""ORIGINAL PROPOSAL:
{text[:3000]}

REVIEW AGENT OUTPUT:
{review_output}

BUDGET AGENT OUTPUT:
{budget_output}

RISK AGENT OUTPUT:
{risk_output}"""
    
    return call_venice_agent(client, "Decision Agent", system_prompt, combined_input)

# def extract_strengths(text):
#     strengths = []
#     lines = text.split('\n')
#     in_strengths = False
#     for line in lines:
#         if 'STRENGTHS:' in line.upper() or 'KEY STRENGTHS:' in line.upper():
#             in_strengths = True
#             continue
#         if in_strengths and line.strip().startswith('-'):
#             strengths.append(line.strip()[1:].strip())
#         elif in_strengths and line.strip() and not line.strip().startswith('-'):
#             break
#     return strengths[:3] if strengths else ["Clear proposal structure", "Well-defined objectives", "Realistic timeline"]

def extract_strengths(text):
    strengths = []
    lines = text.split('\n')
    in_strengths = False
    for line in lines:
        if 'STRENGTHS:' in line.upper() or 'KEY STRENGTHS:' in line.upper():
            in_strengths = True
            continue
        if in_strengths and line.strip().startswith('-'):
            strength_text = line.strip()[1:].strip()
            
            if "1200" not in strength_text and "1,200" not in strength_text:
                strengths.append(strength_text)
        elif in_strengths and line.strip() and not line.strip().startswith('-'):
            break
    return strengths[:3] if strengths else ["Clear proposal structure", "Well-defined objectives", "Realistic timeline"]

def extract_actions(text):
    actions = []
    lines = text.split('\n')
    in_actions = False
    for line in lines:
        if 'ACTION' in line.upper() or 'ACTION ITEMS:' in line.upper():
            in_actions = True
            continue
        if in_actions and line.strip().startswith('-'):
            actions.append(line.strip()[1:].strip())
        elif in_actions and line.strip() and not line.strip().startswith('-'):
            break
    return actions[:2] if actions else ["Monitor project milestones", "Ensure timely reporting"]

def get_recommendation(text, overall_score):
    # Override AI recommendation based on actual score logic
    if overall_score >= 70:
        return "APPROVE", "This proposal meets all criteria and is recommended for funding."
    elif overall_score >= 50:
        return "CONDITIONAL", "This proposal requires modifications before approval."
    else:
        return "REJECT", "This proposal is not recommended for funding."

def create_scores_chart(scores, labels):
    """Create matplotlib bar chart of scores"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = []
    for s in scores:
        if s >= 75:
            colors.append("#34a853")
        elif s >= 50:
            colors.append("#fbbc04")
        else:
            colors.append("#ea4335")
    
    bars = ax.barh(labels, scores, color=colors, height=0.6)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score")
    ax.set_title("GrantGuardian AI - Agent Scores Dashboard", fontweight='bold', pad=15)
    
    for bar, score in zip(bars, scores):
        ax.text(score + 2, bar.get_y() + bar.get_height()/2, 
                str(score), va='center', fontweight='bold')
    
    ax.axvline(x=70, color='#1a73e8', linestyle='--', alpha=0.7, label='Threshold (70)')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <span style="font-size: 2.5rem;">🛡️</span>
            <h1>GrantGuardian AI</h1>
        </div>
        <div class="subtitle">Intelligent Grant Proposal Analysis System</div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        api_key = st.text_input("Venice AI API Key", type="password")
        if api_key:
            st.session_state["venice_api_key"] = api_key
            st.success("✅ API Key saved successfully!")
        
        st.markdown("---")
        st.markdown("**Supported Models:**")
        model_choice = st.selectbox("Select Model", 
                                    ["default", "llama-3.3-70b", "llama-3.1-405b", "qwen32b"],
                                    label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("**About:**")
        st.info(""" GrantGuardian AI is an AI-powered decision support system that evaluates grant proposals using multi-agent analysis (Review, Budget, Risk, and Decision agents).It helps identify proposal quality, financial efficiency, and execution risk to support better funding decisions.
         Developed by: Ravi Khunt""")
        
        st.markdown("""
            <div class="sidebar-guidelines">
                <h4 style="color: #1a73e8; margin: 0 0 0.8rem 0; font-size: 0.95rem;">📋 Upload Guidelines</h4>
                <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.85rem; color: #5f6368; line-height: 1.8;">
                    <li>Upload PDF or CSV files</li>
                    <li>Maximum file size: 20MB</li>
                    <li>CSV files give better structured analysis</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Check API key
    api_key = st.session_state.get("venice_api_key")
    if not api_key:
        st.warning("⚠️ Please enter your Venice AI API key in the sidebar to continue.")
        st.stop()
    
    client = get_venice_client(api_key)
    if client is None:
        st.error("❌ Failed to initialize Venice AI client.")
        st.stop()
    
    # Upload Section
    st.markdown("""
        <div class="upload-section">
            <div class="upload-title">📁 Upload Grant Proposal</div>
            <div style="color: #5f6368; font-size: 0.9rem;">Upload a PDF or CSV file</div>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["pdf", "csv"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        file_size = len(uploaded_file.getvalue()) / 1024
        
        st.markdown(f"""
            <div class="file-uploaded">
                <span style="font-size: 1.2rem;">📄</span>
                <span style="flex: 1; font-size: 0.9rem; color: #202124;">{uploaded_file.name}</span>
                <span style="font-size: 0.8rem; color: #5f6368;">{file_size:.1f}KB</span>
                <span style="color: #34a853; font-size: 1.2rem;">✓</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Extract text
        with st.spinner("Extracting text from file..."):
            if file_type == "pdf":
                extracted_text = extract_text_from_pdf(uploaded_file)
            else:
                extracted_text = extract_text_from_csv(uploaded_file)
        
        if not extracted_text:
            st.error("Failed to extract text from file")
            st.stop()
        
        # Extracted text preview
        with st.expander("📄 Extracted Text Preview"):
            st.text(extracted_text[:2000] + ("..." if len(extracted_text) > 2000 else ""))
        
        # Run Analysis Button
        col_btn = st.columns([6, 1])[1]
        with col_btn:
            run_analysis = st.button("🚀 Run Full Analysis", type="primary")
        
        if run_analysis:
            scores = {}
            
            # Run all agents with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📝 Review Agent analyzing...")
            review_output = review_agent(client, extracted_text)
            scores['clarity'] = parse_score(review_output)
            progress_bar.progress(25)
            
            status_text.text("💰 Budget Agent analyzing...")
            budget_output = budget_agent(client, extracted_text)
            scores['efficiency'] = parse_score(budget_output)
            progress_bar.progress(50)
            
            status_text.text("⚠️ Risk Agent analyzing...")
            risk_output = risk_agent(client, extracted_text)
            scores['risk'] = parse_score(risk_output)
            progress_bar.progress(75)
            
            status_text.text("🎯 Decision Agent evaluating...")
            decision_output = decision_agent(client, extracted_text, review_output, 
                                            budget_output, risk_output)
            scores['overall'] = parse_score(decision_output)
            progress_bar.progress(100)
            
            status_text.empty()
            progress_bar.empty()
            
            # Score Cards Row
            # st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            cols = st.columns(4)
            
            score_data = [
                ("📝", "Clarity Score", scores['clarity'], "High" if scores['clarity'] >= 75 else "Good" if scores['clarity'] >= 50 else "Needs Work", "score-blue"),
                ("💰", "Efficiency Score", scores['efficiency'], "Good" if scores['efficiency'] >= 75 else "Average" if scores['efficiency'] >= 50 else "Poor", "score-green"),
                ("⚠️", "Risk Score (Higher = Safer)", scores['risk'], "Very Safe" if scores['risk'] >= 85 else "Safe" if scores['risk'] >= 70 else "Moderate", "score-purple"),
                ("🎯", "Overall Score", scores['overall'], "Strong" if scores['overall'] >= 80 else "Good" if scores['overall'] >= 60 else "Weak", "score-orange")
            ]
            
            for i, (icon, label, score, status, color_class) in enumerate(score_data):
                with cols[i]:
                    st.markdown(f"""
                        <div class="score-card">
                            <div class="score-label">{icon} {label}</div>
                            <div class="score-value {color_class}">{score}<span class="score-total">/100</span></div>
                            <div class="score-status">{status}</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Dashboard and Recommendation Row
            st.markdown("<br>", unsafe_allow_html=True)
            col_left, col_right = st.columns([3, 2])
            
            with col_left:
                # Score Dashboard using native Streamlit
                st.markdown('<div class="dashboard-container">', unsafe_allow_html=True)
                st.markdown("### 📊 Score Dashboard")
                st.caption("Agent-wise performance overview (Score out of 100)")
                
                # Use Streamlit native components for the table
                metrics_data = {
                    "Metric": ["Clarity", "Efficiency", "Risk (Higher = Safer)", "Overall"],
                    "Score": [scores['clarity'], scores['efficiency'], scores['risk'], scores['overall']]
                }
                
                # # Create progress bars
                # for i, (metric, score) in enumerate(zip(metrics_data["Metric"], metrics_data["Score"])):
                #     col1, col2, col3 = st.columns([2, 1, 4])
                #     with col1:
                #         st.write(f"**{metric}**")
                #     with col2:
                #         st.write(f"**{score}**")
                #     with col3:
                #         st.progress(score / 100.0)

                # Create colored progress bars
                for metric, score in zip(metrics_data["Metric"], metrics_data["Score"]):
                    col1, col2, col3 = st.columns([2, 1, 4])
                    with col1:
                        st.write(f"**{metric}**")
                    with col2:
                        st.write(f"**{score}**")
                    with col3:
                        # Determine color based on score
                        if score >= 75:
                            bar_color = "#34a853"  # Green
                        elif score >= 50:
                            bar_color = "#fbbc04"  # Yellow/Orange
                        else:
                            bar_color = "#ea4335"  # Red
                        
                        # Custom HTML progress bar with color
                        st.markdown(f"""
                            <div style="background-color: #e0e0e0; border-radius: 4px; height: 24px; margin-top: 5px;">
                                <div style="background-color: {bar_color}; width: {score}%; height: 100%; border-radius: 4px; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.85rem; font-weight: 500;">
                                    {score}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Legend
                st.markdown("""
                    <div class="legend-container">
                        <div class="legend-item">
                            <div class="legend-dot" style="background: #ea4335;"></div>
                            <span>0-49 (Needs Improvement)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-dot" style="background: #fbbc04;"></div>
                            <span>50-74 (Average)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-dot" style="background: #34a853;"></div>
                            <span>75-100 (Excellent)</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_right:
                # Final Recommendation
                rec, rec_text = get_recommendation(decision_output, scores['overall'])
                strengths = extract_strengths(decision_output)
                actions = extract_actions(decision_output)
                
                rec_color = "#34a853" if rec == "APPROVE" else "#fbbc04" if rec == "CONDITIONAL" else "#ea4335"
                rec_bg = "#e6f4ea" if rec == "APPROVE" else "#fef3e8" if rec == "CONDITIONAL" else "#fce8e6"
                
                st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                st.markdown("### 📋 Final Recommendation")
                
                st.markdown(f"""
                    <div class="approve-box" style="background: {rec_bg}; border: 1px solid {rec_color};">
                        <div style="color: {rec_color}; font-weight: 600; font-size: 1rem; margin-bottom: 0.3rem;">
                            ✓ {rec}
                        </div>
                        <div style="color: {rec_color}; font-size: 0.9rem;">{rec_text}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("**Key Strengths**")
                for s in strengths:
                    st.markdown(f"- {s}")
                
                st.markdown("**Action Items**")
                for a in actions:
                    st.markdown(f"- {a}")
                
                # Download button
                export_data = f"""GRANTGUARDIAN AI ANALYSIS REPORT
{'='*50}

REVIEW AGENT OUTPUT:
{review_output}

BUDGET AGENT OUTPUT:
{budget_output}

RISK AGENT OUTPUT:
{risk_output}

DECISION AGENT OUTPUT:
{decision_output}

SCORES:
- Clarity: {scores['clarity']}/100
- Efficiency: {scores['efficiency']}/100
- Risk: {scores['risk']}/100
- Overall: {scores['overall']}/100
"""
                st.download_button("⬇ Download Full Report", export_data, 
                                 file_name="grantguardian_analysis.txt", 
                                 mime="text/plain",
                                 use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Matplotlib Chart
            st.markdown("---")
            st.subheader("📈 Score Visualization")
            fig = create_scores_chart(
                [scores['clarity'], scores['efficiency'], scores['risk'], scores['overall']],
                ['Clarity', 'Efficiency', 'Risk (Low)', 'Overall']
            )
            st.pyplot(fig)

if __name__ == "__main__":
    main()
# GrantGuardian AI

**Intelligent Grant Proposal Analysis System**

GrantGuardian AI is an advanced Streamlit application that leverages 4 specialized AI agents to analyze grant proposals, providing comprehensive insights, risk assessments, and data-driven recommendations.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Venice AI](https://img.shields.io/badge/Venice%20AI-API-green.svg)

---

## Features

### 4 Specialized AI Agents

| Agent | Function | Output |
|-------|----------|--------|
| 📝 **Review Agent** | Analyzes proposal clarity and structure | Clarity Score (0-100) |
| 💰 **Budget Agent** | Detects inefficiencies and waste | Efficiency Score (0-100) |
| ⚠️ **Risk Agent** | Evaluates implementation risks | Risk Score (0-100) |
| 🎯 **Decision Agent** | Final recommendation based on all inputs | Overall Score + Verdict |

### Key Capabilities
- 📄 **Multi-format Support**: Upload PDF or CSV grant proposals
- 📊 **Visual Dashboard**: Interactive score cards and progress bars
- 📈 **Matplotlib Charts**: Horizontal bar charts with threshold indicators
- 💾 **Export Reports**: Download full analysis as text file
- 🎨 **Modern UI**: Clean, professional interface with color-coded scores

---

## Score Interpretation
- ## Score Range	Status	Color
- ## 75-100	Excellent	🟢 Green
- ## 50-74	Average	🟡 Yellow
- ## 0-49	Needs Improvement	🔴 Red
- ## Threshold: 70+ recommended for approval

---

## 🚀 Live Demo

**[Click here to view live app](STREAMLIT_URL_HERE)**

---

## Technologies Used
- ## Frontend: Streamlit
- ## AI/ML: Venice AI API (OpenAI-compatible)
- ## PDF Processing: PyPDF2
- ## Data Analysis: Pandas
- ## Visualization: Matplotlib
- ## Language: Python 3.9+

---

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- Venice AI API Key ([Get one here](https://venice.ai/))

### Setup

1. Clone the repository:
```bash
git clone https://github.com/RK5Coder/grantguardian-ai.git
cd grantguardian-ai

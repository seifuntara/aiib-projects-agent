# 🚀 AIIB Projects Search Agent
This dashboard uses **LLM-Powered** search to answer questions about projects financed by the **Asian Infrastructure Investment Bank (AIIB)**.
Check out delpoyed link on [Streamlit](https://aiib-projects-agent.streamlit.app/)

---

## Architecture
* 📊 Project data loaded into context.
* 🤖 Gemini 2.5 Flash with system instructions
* 💡 General knowledge integration

## 🛠 Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Intelligence:** [Google Gemini 2.5 Flash](https://ai.google.dev/)
* **Data Processing:** [Pandas](https://pandas.pydata.org/)
* **Language:** Python 3.10+

---

## 📊 Data Source
Data are periodically sourced from publicly available AIIB project disclosures.
[Learn more about AIIB](https://www.aiib.org/en/projects/list/index.html)

---

## ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/aiib-dashboard.git](https://github.com/your-username/aiib-dashboard.git)
    cd aiib-dashboard
    ```

2.  **Install dependencies:**
    ```bash
    pip install streamlit pandas google-generativeai tabulate
    ```

3.  **Configure Secrets:**
    Create a `.streamlit/secrets.toml` file in your project root:
    ```toml
    GOOGLE_API_KEY = "your_gemini_api_key_here"
    ```

4.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

---
**Disclaimer:** *This is an AI-powered tool. Always verify financial totals against official AIIB documentation.*

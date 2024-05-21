import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from pandasai.llm.openai import OpenAI
from pandasai.responses.streamlit_response import StreamlitResponse
import matplotlib.pyplot as plt
import plotly.express as px
from pandasai import Agent
from pandasai.skills import skill

# Set plot option for streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set maximum row, column size for uploaded files
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# Load environment variables
load_dotenv()

# Check open_ai_api_key in st.secrets
OPEN_AI_API_KEY = st.secrets["OpenAI_API_KEY"]

if OPEN_AI_API_KEY is None or OPEN_AI_API_KEY == "":
    st.error("open_ai_api_key 未设置为环境变量")
else:
    st.success("Open AI API 密钥已设置")

# Set title for Streamlit UI
st.title("数据智能分析")
st.info("上传文件并开始提问")

# Set formats of files allowed to upload
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

# Define a function for file formats
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"不支持的文件格式: {ext}")
        return None

# Cumulative sum for the column
@skill
def cum_sum(dataframe, col):
    return dataframe[col].cumsum()

# Upload a file from UI
uploaded_file = st.file_uploader("请上传数据文件", type=list(file_formats.keys()))

# Check the uploaded file whether empty or not
if uploaded_file:
    dataframe = load_data(uploaded_file)
    if dataframe is not None and not dataframe.empty:
        st.write(dataframe.head(3))

        # Give a general description of data
        if st.sidebar.button("统计"):
            st.write(dataframe.describe())
        if st.sidebar.button("信息"):
            df_info = dataframe.info()
            st.write(df_info)
        if st.sidebar.button("平均值"):
            df_mean = dataframe.mean()
            st.write(df_mean)
        if st.sidebar.button("中位数"):
            df_med = dataframe.median()
            st.write(df_med)
        if st.sidebar.button("样本"):
            df_sample = dataframe.sample()
            st.write(df_sample)
        if st.sidebar.button("相关性"):
            df_corr = dataframe.corr()
            st.write(df_corr)
        if st.sidebar.button("删除所有重复项"):
            df_drop_all = dataframe.drop_duplicates()
            st.write(df_drop_all)
        if st.sidebar.button("删除所有空值"):
            df_drop_null = dataframe.dropna(how='all')
            st.write(df_drop_null)

        # Define preferred LLM
        llm = OpenAI(api_token=OPEN_AI_API_KEY, model="gpt-3.5-turbo")

        # Create agent
        agent = Agent([dataframe],
                      config={
                          "save_logs": True,
                          "verbose": True,
                          "enforce_privacy": True,
                          "enable_cache": True,
                          "use_error_correction_framework": True,
                          "max_retries": 3,
                          "open_charts": True,
                          "save_charts": False,
                          "save_charts_path": "exports/charts",
                          "custom_whitelisted_dependencies": [],
                          "llm": llm,
                          "saved_dfs": [],
                          "response_parser": StreamlitResponse
                      },
                      memory_size=10)

        # Add agent skills
        agent.add_skills(cum_sum)

        # Get question from user
        prompt = st.text_area("请输入您的问题:")
        st.info("输入您的问题，如果要获取图表，可以指定图表类型")

        # Generate answer for the question
        if st.button("生成回复"):
            if not dataframe.empty:
                with st.spinner("计算中..."):
                    try:
                        response = agent.chat(prompt)
                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            "Generated Response",
                            "Generated Chart",
                            "Explanation",
                            "Generated Code",
                            "Clarification Questions"
                        ])
                        with tab1:
                            st.write(response)

                        with tab2:
                            try:
                                df_numeric = dataframe.select_dtypes(include=['number'])
                                if 'pie' in prompt.lower():
                                    fig = px.pie(df_numeric)
                                elif 'bar' in prompt.lower():
                                    fig = px.bar(df_numeric)
                                elif 'bubble' in prompt.lower():
                                    fig = px.scatter(df_numeric)
                                elif 'dot' in prompt.lower():
                                    fig = px.scatter(df_numeric)
                                elif 'time series' in prompt.lower():
                                    fig = px.line(df_numeric)
                                else:
                                    fig = px.histogram(df_numeric)
                                st.plotly_chart(fig)
                            except Exception as e:
                                st.error(f"An error occurred while generating the chart: {e}")

                        with tab3:
                            st.write(agent.explain())

                        with tab4:
                            st.write(f"生成代码 :")
                            st.write(agent.last_code_executed)

                        with tab5:
                            try:
                                clarification_questions = agent.clarification_questions(response)
                                st.write("\n".join(clarification_questions))
                            except Exception as e:
                                st.error(f"An error occurred while getting clarification questions: {e}")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("空文件无法生成回复")
    else:
        st.write("给定的数据为空，请上传完整文件以提问")
else:
    st.warning("请上传数据文件")

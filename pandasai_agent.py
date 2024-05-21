import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from pandasai.llm.openai import OpenAI
from pandasai.responses.streamlit_response import StreamlitResponse
import matplotlib.pyplot as plt
import plotly.express as px
from langchain.evaluation import load_evaluator
from pandasai import Agent
from pandasai.skills import skill

#set plot option for streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)

#set maximum row,column size for uploaded files
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# load environment variables
load_dotenv()

# check open_ai_api_key in env, if it is not defined as a variable
#it can be added manually in the code below
st.write("OPEN_AI_API_KEY",st.secrets["OpenAI_API_KEY"])

OPEN_AI_API_KEY = st.secrets["OpenAI_API_KEY"]

#if OPEN_AI_API_KEY is None or OPEN_AI_API_KEY == "":
 #   st.error("open_ai_api_key 未设置为环境变量")
#else:
 #   st.success("Open AI API 密钥已设置")

# set tittle for Streamlit UI
st.title("数据智能分析")
st.info("上传文件并开始提问")

#set formats of files allowed to upload
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}

#define a function for file formats
#check the file format among the list above
def load_data(uploaded_file):
   # ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    #if file format does not match give message
    else:
        st.error(f"不支持的文件格式: {ext}")
        return None

#cumulative sum for the column
def cum_sum(dataframe,col):
    return dataframe(col).cumsum()


#upload a file from ui
uploaded_files = st.file_uploader("请上传数据文件",
                                 type =list(file_formats.keys()),)# accept_multiple_files=True,

#check the uploaded file whether empty or not
if uploaded_files:
    dataframe = load_data(uploaded_files)
    if dataframe.empty:
        #if file is empty give a message
        st.write("给定的数据为空，请上传完整文件以提问")
    # if file is full, list first 3 rows
    else:
        st.write(dataframe.head(3))
    # give a general description of data
    if st.sidebar.button("统计"):
        if dataframe.empty:
            st.write("空文件无法进行描述")
    # if uploaded file is full, return description of data
        else:
            df_desc = dataframe.describe()
            st.write(df_desc)
    if st.sidebar.button("信息"):
        df_info=dataframe.info()
        st.write(df_info)
    if st.sidebar.button("平均值"):
        df_mean=dataframe.mean()
        st.write(df_mean)
    if st.sidebar.button("中位数"):
        df_med=dataframe.median()
        st.write(df_med)
    if st.sidebar.button("样本"):
        df_sample=dataframe.sample()
        st.write(df_sample)
    if st.sidebar.button("相关性"):
        df_corr = dataframe.corr()
        st.write(df_corr)
    if st.sidebar.button("删除所有重复项"):
        df_drop_all=dataframe.drop_duplicates()
        st.write(df_drop_all)
    if st.sidebar.button("删除所有空值"):
        df_drop_null =dataframe.dropna(how='all')
        st.write(df_drop_null)

# define preffered llm
    llm = OpenAI(api_token=OPEN_AI_API_KEY,model="gpt-3.5-turbo-1106")

# create agent

    agent = Agent([dataframe],
              config={"save_logs": True,
                                      "verbose": True,
                                      "enforce_privacy": True,
                                      "enable_cache": True,
                                      "use_error_correction_framework": True,
                                      "max_retries": 3,
                                      #"custom_prompts": {},
                                      "open_charts": True,
                                      "save_charts": False,
                                      "save_charts_path": "exports/charts",
                                      "custom_whitelisted_dependencies": [],
                                      "llm": llm,
                                      #"llm_options": null,
                                      "saved_dfs": [],
                                     "response_parser": StreamlitResponse
                                      #  share a custom sample head to the LLM  "custom_head": head_df

                                 }
                                    , memory_size=10
                              )

    #adding agent skilss

    agent.add_skills(cum_sum)


    # get question from user
    prompt =st.text_area("请输入您的问题:")
    st.info("输入您的问题，如果要获取图表，可以指定图表类型")



#get the request to generate an asnwer for the question
# check uploaded file, if the file is empty give a message
# show the model's response as an answer

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
                    if 'pie' in prompt.lower():
                        fig = px.pie(dataframe)
                    elif 'bar' in prompt.lower():
                        fig = px.bar(dataframe)
                    elif 'bubble' in prompt.lower():
                        fig = px.scatter(dataframe)
                    elif 'dot' in prompt.lower():
                        fig = px.scatter(dataframe)
                    elif 'time series' in prompt.lower():
                        fig = px.line(dataframe)
                    else:
                        fig = px.histogram(dataframe)
                    st.plotly_chart(fig)

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








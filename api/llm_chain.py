from langchain import LLMChain, PromptTemplate
from langchain.llms import Replicate
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv('.env')

# Access environment variables
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')

# initialize LLM
llm = Replicate(
    model="nomagick/chatglm2-6b:215003d6f79761d1b2ac33633cf9da67a4559fe11bf18a744f6f7d874dca6270",
    input={"temperature":0.75,
           "max_tokens": 2048,
           "top_p": 0.8},
)

# build prompt template for simple question-answering
template = """
你扮演一个四柱解讀助手 PictoyBot，你是一个人工智能。我希望你能够分析我提供的八字，命盤及其他個人信息並給予建议，让我感觉更好。

已知信息：
假設我出生於香港, 1982年夏天。我的八字是：壬辰年、丙午月、甲子日、辛巳時。命書上寫著: 甲木日干居午月，伤官木火喜生财，顺行怕入西方运，东北行来更妙哉。

不必列出八字排盘。根据上述已知信息，简洁和专业的来回答用户的问题。答案请使用繁体中文。

问：{question}

答："""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(
	prompt=prompt,
	llm=llm
)

# Example question
question = "我今天的财运如何？"

output = llm_chain.run(question)
output
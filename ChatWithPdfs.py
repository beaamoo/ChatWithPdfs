#chat with pdfs app
#v1

#import libraries
import platform
import os
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

#getting the Open AI Key stored as an environment variable
OPENAI_API_KEY=os.environ['OPENAI_API_KEY']

#setting the report name
report="Tesla_Annual_Report_2023_Jan31.pdf"

class bcolors:
    GREEN = '\033[92m'
    ENDCOLOR = '\033[0m'

#load, convert to text and split pdf into pages
pages = PyPDFLoader(report).load_and_split()

#chunking/splitting each of the pages into overlapping sections
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len)
sections = text_splitter.split_documents(pages)

#create FAISS index
faiss_index = FAISS.from_documents(sections, OpenAIEmbeddings(OPENAI_API_KEY))

#define chain
retriever = faiss_index.as_retriever()
memory = ConversationBufferMemory(
    memory_key='chat_history', 
    return_messages=True, 
    output_key='answer')

chain = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(),
    retriever=retriever,
    memory=memory)


#collect user input and simulate chat with pdf
if platform.system() == "Windows":
    eof_key = "<Ctrl+Z>"
else:
    eof_key = "<Ctrl+D>"

print(f'Lets talk with the'+report+'. What would you like to know? Or press {eof_key} to exit.')

while True:
    try:
        user_input = input('Q:')
        print(f"{bcolors.GREEN} A: {chain({'question': user_input})['answer'].strip()}{bcolors.ENDCOLOR}")
    except EOFError:
        break
    except KeyboardInterrupt:
        break

print("Bye")

#test the chat
print(chain({'question': 'What was Tesla total revenues and net income?'}))
#print(chain({'question': 'Sum these values?'}))
#print(chain({'question': 'What was the main risk factors for Tesla?'}))

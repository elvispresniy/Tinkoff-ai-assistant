from config import PINECONE_API_KEY, OPENAI_API_KEY
import messages

from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

import pinecone

# LLM
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

# Embeddings initialization
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Vectorstore initialization
pinecone.init(      
	api_key=PINECONE_API_KEY,
	environment='asia-southeast1-gcp-free'      
)      
index = pinecone.Index('tinkoffai')

vectorstore = Pinecone.from_existing_index("tinkoffai", embedding=embeddings)

def pipeline(query, chat_history):

    # Написать обобщение всего диалога
    summary_args = {
        'summary': chat_history,
        'new_lines': query
    }
    summary_prompt = messages.templates['memory_template'].format(**summary_args)
    print(f'\nsummary_prompt:\n\n{summary_prompt}')

    summarization = llm.predict(summary_prompt)
    print(f'\nsummarization:\n\n{summarization}')

    # Собрать запрос для модели и получить на него ответ
    documents_raw = vectorstore.similarity_search(query)
    documents = '\n\n'.join([document.page_content for document in documents_raw])

    conversation_args = {
        'documents': documents,
        'history': summarization,
        'input': query
    }
    conversation_prompt = messages.templates['conversation_prompt'].format(**conversation_args)
    print(f'\nconversation_prompt:\n\n{conversation_prompt}')

    result = llm.predict(conversation_prompt)
    print(f'\nresult:\n\n{result}')

    # Сохранить новую историю диалога
    new_chat_history = f'''{summarization}
    Клиент: {query}
    Бот-ассистент: {result}'''

    print(f'\nnew_chat_history:\n\n{new_chat_history}')

    return result, new_chat_history
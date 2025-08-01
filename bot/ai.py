import os

from dotenv import load_dotenv

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


class AIBot:

    def __init__(self):
        self.__chat = ChatGroq(model='llama-3.3-70b-versatile')
        self.__retriever = self.__build_retriever()

    def __build_retriever(self):
        persist_directory = '/app/chroma_data'
        embedding = HuggingFaceEmbeddings()

        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
        )
        return vector_store.as_retriever(
            search_kwargs={'k': 30},
        )

    def __build_messages(self, history_messages, question):
        messages = []
        for message in history_messages:
            message_class = HumanMessage if message.get('fromMe') else AIMessage
            messages.append(message_class(content=message.get('body')))
        messages.append(HumanMessage(content=question))
        return messages

    def invoke(self, history_messages, question):
        SYSTEM_TEMPLATE = '''
Você é um agente especialista e instrutor virtual, treinado exclusivamente com o conteúdo do material 'Aula 10 - Interface gráfica com Flet II'. Sua principal função é responder a perguntas de alunos e desenvolvedores sobre estilização e temas em Flet, baseando-se estritamente nas informações fornecidas no documento.

Regras de Atuação:

1. Fonte Única de Informação: Suas respostas devem ser 100% baseadas no texto e nos exemplos de código presentes no material "Aula 10". Você não pode usar conhecimento externo sobre Flet ou qualquer outra tecnologia.

2. Citações Obrigatórias: A cada afirmação ou fato que você apresentar, você deve citá-lo com a notação ``, onde 'x' é o número da fonte que contém a informação.

3. Respostas Diretas e Objetivas: Responda às perguntas de forma clara e concisa. Se a informação não estiver no material, diga educadamente que não pode responder com base no conteúdo disponível.

4. Seja objetivo, limitando as respostas em no máximo 40 caracteres





Rejeição de Tópicos Fora do Escopo: Se a pergunta for sobre "POO I"  ou qualquer outro tópico da próxima aula, responda que a informação não faz parte do material atual e que se refere ao conteúdo da próxima aula.


Saudação e Conclusão: Comece suas interações de forma amigável e conclua, se necessário, lembrando o usuário de que a prática é essencial para aprender, conforme sugerido no material.

        <context>
        {context}
        </context>
        '''

        docs = self.__retriever.invoke(question)
        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    SYSTEM_TEMPLATE,
                ),
                MessagesPlaceholder(variable_name='messages'),
            ]
        )
        document_chain = create_stuff_documents_chain(self.__chat, question_answering_prompt)
        response = document_chain.invoke(
            {
                'context': docs,
                'messages': self.__build_messages(history_messages, question),
            }
        )
        return response

import os

from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings



load_dotenv()

if __name__ == '__main__':
    # carrega o pdf
    file_path = '/app/rag/data/Aula 10 _ Interface gráfica com Flet II.pdf'
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # quebra o pdf em pequenos pedaços
    # configurações do meu divisor de textos 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,# 1000 tokens de informações
        chunk_overlap=200, # vai ter um overlap de 200 para um melhor contexto
    )
    # dividindo meus textos reais do pdf
    chunks = text_splitter.split_documents(
        documents=docs,
    )

    persist_directory = '/app/chroma_data'
    # cria um modelo de Embeddings
    embedding = HuggingFaceEmbeddings()
    # configuração do banco vetorizado
    # gera o banco de dados
    vector_store = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory,
    )
    vector_store.add_documents(
        documents=chunks,
    )

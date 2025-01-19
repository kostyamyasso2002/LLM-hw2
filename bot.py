import os

from langchain_community.llms import VLLM
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
import chromadb
from uuid import uuid4
from langchain.llms import OpenAI

import logging

from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

OPENAI_API_KEY = None
TELEGRAM_BOT_TOKEN = None

def create_vectorstore(documents, persist_directory: str = "chroma_db"):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key = OPENAI_API_KEY
    )

    if not os.path.exists(persist_directory):
        persistent_client = chromadb.PersistentClient()
        collection = persistent_client.get_or_create_collection("collection_name")

        vector_store_from_client = Chroma(
            client=persistent_client,
            collection_name="collection_name",
            embedding_function=embeddings,
        )

        uuids = [str(uuid4()) for _ in range(len(documents))]

        vector_store_from_client.add_documents(documents=documents, ids=uuids)
    else:
        vector_store_from_client = Chroma(persist_directory="./chroma", embedding_function=embeddings)
    return vector_store_from_client


def create_chain(folder_path: str):
    loader = TextLoader(folder_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = splitter.split_documents(documents)

    vectorstore = create_vectorstore(split_docs)

    llm_op = OpenAI(temperature=0, openai_api_key = OPENAI_API_KEY)
    system_prompt = (
        "Use the given context to answer the question based on the book 'Modal Logic' by Patrick Blackburn, Maarten de Rijke and Yde Venema. "
        "If you don't know the answer, say you don't know. "
        "Use four sentence maximum and keep the answer concise. "
        "Context: {context}"
        "Question: {input}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm_op, prompt)
    chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)
    return chain


folder_path = "modlog.txt"
qa_pipeline = create_chain(folder_path)

def get_ans(question):
    response = qa_pipeline.invoke({"input":question})
    return response['answer']

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_html(
        f"Привет, {user.mention_html()}!\nОтправьте ваше сообщение, касающееся модальной логики.",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Отправьте ваше сообщение, касающееся модальной логики.")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(get_ans(update.message.text))


def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

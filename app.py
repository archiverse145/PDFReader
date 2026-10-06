import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

from htmlTemplates import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            text += page.extract_text() or ""

    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)

    return chunks


def get_vector_store(chunks):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl"
    )

    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vectorstore


def get_conversation_chain(vectorstore):

    llm = ChatOpenAI(
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain


def handle_userinput(user_question):

    response = st.session_state.conversation.invoke(
        {"question": user_question}
    )

    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):

        if i % 2 == 0:

            st.write(
                user_template.replace(
                    "{{MSG}}",
                    message.content
                ),
                unsafe_allow_html=True
            )

        else:

            st.write(
                bot_template.replace(
                    "{{MSG}}",
                    message.content
                ),
                unsafe_allow_html=True
            )


def main():

    load_dotenv()

    st.set_page_config(
        page_title="Chat with Multiple PDFs",
        page_icon="📚"
    )

    # Load CSS
    st.write(css, unsafe_allow_html=True)

    st.header("Chat with Multiple PDFs 📚")

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # User question
    user_question = st.text_input(
        "Ask a question about your documents:"
    )

    if user_question:

        if st.session_state.conversation is not None:
            handle_userinput(user_question)

        else:
            st.warning(
                "Please upload and process a PDF first."
            )

    # Sidebar
    with st.sidebar:

        st.subheader("Your documents")

        pdf_docs = st.file_uploader(
            "Upload here and then click on Process:",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process"):

            if pdf_docs:

                with st.spinner("Processing"):

                    # Get PDF text
                    raw_text = get_pdf_text(pdf_docs)

                    # Get text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Create vector store
                    vectorstore = get_vector_store(text_chunks)

                    # Create conversation chain
                    st.session_state.conversation = (
                        get_conversation_chain(vectorstore)
                    )

                    st.success("Processing complete!")

            else:
                st.warning(
                    "Please upload at least one PDF."
                )


if __name__ == "__main__":
    main()
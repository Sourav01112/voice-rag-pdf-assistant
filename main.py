import ollama
import os
import datetime
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
from dotenv import load_dotenv


class RAGSystem:
    def __init__(
        self, model_name: str = "llama3.2", vector_db_path: str = "./db/vector_db"
    ):
        self.model_name = model_name
        self.vector_db_path = vector_db_path
        self.vector_db = None
        self.chain = None
        load_dotenv()

    def load_pdf_files(self, data_directory: str = "./data") -> List[Document]:
        pdf_files = [f for f in os.listdir(data_directory) if f.endswith(".pdf")]
        all_pages = []

        for pdf_file in pdf_files:
            file_path = os.path.join(data_directory, pdf_file)
            print(f"Processing PDF file: {pdf_file}")

            loader = PDFPlumberLoader(file_path=file_path)
            pages = loader.load_and_split()
            print(f"Pages loaded: {len(pages)}")

            all_pages.extend(pages)

            # Optional: Generate summary for each PDF
            self._summarize_pdf(pdf_file, pages[0].page_content if pages else "")

        return all_pages

    def _summarize_pdf(self, filename: str, text: str) -> None:
        prompt = f"""
        You are an AI assistant that helps with summarizing PDF documents.
        
        Here is the content of the PDF file '{filename}':
        
        {text}
        
        Please summarize the content of this document in a few sentences.
        """

        try:
            response = ollama.generate(model=self.model_name, prompt=prompt)
            summary = response.get("response", "")
            print(f"Summary generated for '{filename}'")
        except Exception as e:
            print(f"Error summarizing '{filename}': {str(e)}")

    def split_text_into_chunks(
        self, pages: List[Document], chunk_size: int = 1200, chunk_overlap: int = 300
    ) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        text_chunks = []
        for page in pages:
            chunks = text_splitter.split_text(page.page_content)
            text_chunks.extend(chunks)

        print(f"Created {len(text_chunks)} text chunks")
        return text_chunks

    def add_metadata_to_chunks(
        self,
        chunks: List[str],
        doc_title: str = "American Data and Future",
        author: str = "US Tech",
    ) -> List[Dict[str, Any]]:
        metadata_chunks = []
        for chunk in chunks:
            metadata = {
                "title": doc_title,
                "author": author,
                "date": str(datetime.date.today()),
            }
            metadata_chunks.append({"text": chunk, "metadata": metadata})

        return metadata_chunks

    def setup_vector_database(self, metadata_chunks: List[Dict[str, Any]]) -> None:
        docs = [
            Document(page_content=chunk["text"], metadata=chunk["metadata"])
            for chunk in metadata_chunks
        ]

        fastembedding = FastEmbedEmbeddings()

        self.vector_db = Chroma.from_documents(
            documents=docs,
            embedding=fastembedding,
            persist_directory=self.vector_db_path,
            collection_name="docs-local-rag",
        )
        print("Vector database created and populated")

    def setup_retrieval_chain(self) -> None:
        if not self.vector_db:
            raise ValueError(
                "Vector database not initialized. Call setup_vector_database first."
            )

        llm = ChatOllama(model=self.model_name)

        query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI assistant. Given a user question, generate five alternative phrasings 
            that could retrieve relevant documents from a vector database. The variations should reflect 
            different possible interpretations, rewordings, or perspectives to overcome the limitations of 
            distance-based similarity search. Output each alternative on a new line.
    
            Original question: {question}""",
        )

        retriever = MultiQueryRetriever.from_llm(
            self.vector_db.as_retriever(), llm, prompt=query_prompt
        )

        rag_template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(rag_template)

        self.chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("Retrieval chain setup complete")

    def query(self, question: str) -> str:
        if not self.chain:
            raise ValueError(
                "Retrieval chain not initialized. Call setup_retrieval_chain first."
            )

        response = self.chain.invoke(question)
        return response

    def text_to_speech(self, text: str, voice_model: str = "eleven_turbo_v2") -> None:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            print("ElevenLabs API key not found in environment variables")
            return

        try:
            client = ElevenLabs(api_key=api_key)
            audio_stream = client.generate(text=text, model=voice_model, stream=True)
            stream(audio_stream)
        except Exception as e:
            print(f"Error in text-to-speech: {str(e)}")


def main():
    rag = RAGSystem()

    print("Loading PDF files...")
    pages = rag.load_pdf_files("./data")

    print("Splitting text into chunks...")
    chunks = rag.split_text_into_chunks(pages)

    print("Adding metadata...")
    metadata_chunks = rag.add_metadata_to_chunks(chunks)

    print("Setting up vector database...")
    rag.setup_vector_database(metadata_chunks)

    print("Setting up retrieval chain...")
    rag.setup_retrieval_chain()

    question = "Does the document mention any specific technologies?"
    print(f"\nQuery: {question}")

    response = rag.query(question)
    print(f"Response: {response}")

    # Convert to speech
    print("Converting to speech...")
    rag.text_to_speech(response)


if __name__ == "__main__":
    main()

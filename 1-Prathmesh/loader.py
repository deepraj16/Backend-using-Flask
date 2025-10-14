
from langchain_community.document_loaders import TextLoader ,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf_document(file_path):
    try:
        print(f"Loading document: {file_path}")
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError("Only PDF and TXT files are supported")
        documents = loader.load()
        print(f"{len(documents)} pages loaded")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "।", ".", "!", "?", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks")
        
        return split_docs
        
    except Exception as e:
        print(f"Document loading error: {e}")
        raise

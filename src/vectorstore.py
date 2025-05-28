from typing import Union #allow accept either Chroma or FAISS
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

#Stores documents as embeddings and retrieves them based on similarity

class VectorDB:
    def __init__(self,
                documents = None, #documents stored in the vector database
                vector_db: Union[Chroma, FAISS] = Chroma, #type of vector database
                embedding = HuggingFaceEmbeddings(), #turn text → vectors
                ) -> None :
        self.vector_db = vector_db
        self.embedding = embedding
        self.db = self._build_db (documents)
    
    def _build_db(self, documents): #converts documents into embeddings and stores in DB
        """from_doc:standard LangChain method for creating a vector store from a list of documents"""
        db = self.vector_db.from_documents(documents=documents,
                                            embedding=self.embedding)
        return db

    def get_retriever (self,
                        search_type: str = "similarity",
                        search_kwargs: dict = {"k": 10} #top 10 similar chunks
                        ):
        retriever = self.db.as_retriever (search_type=search_type,
                                        search_kwargs=search_kwargs) #Converts the vector DB into a Retriever 
        return retriever
import re #text pattern matching
from langchain import hub #pull prompt template from hub
from langchain_core.runnables import RunnablePassthrough #pass input without modification to the next step
from langchain_core.output_parsers import StrOutputParser #clean up the answer from LLM

"""Build the RAG chain that queries the LLM using retrieved chunks.."""

class Str_OutputParser(StrOutputParser): #create class A that inherits from class B
    def __init__(self) -> None:
        super().__init__() #run the constructor of the parent class
    def parse (self, text: str) -> str: 
        return self.extract_answer(text) #def a method that extracts the answer from the text
    
    def extract_answer(self,
                        text_response: str,
                        pattern: str = r"Answer:\s*(.*)" #find "Answer:" in the LLM response
                        ) -> str:
        match = re.search(pattern, text_response, re. DOTALL) #multiline answers are captured
        if match:
            answer_text = match.group(1).strip()
            return answer_text #returns only the answer part (Cleans the LLM’s raw response)
        else:
            return text_response



class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm #pass the LLM to the class
        self.prompt = hub.pull ("rlm/rag-prompt") #pull the prompt template 
        self.str_parser = Str_OutputParser() #create the custom output parser for extracting answers

    def get_chain (self, retriever):
        input_data = {
            "context": retriever | self.format_docs, #pass documents to formats them into a string
            "question": RunnablePassthrough()
        }
        """
        input data -> prompt template fills in {context} and {question}, LLM generates text, and parser extracts the answer."""
        rag_chain = (
            input_data 
            | self.prompt 
            | self.llm
            | self.str_parser
        )
        return rag_chain
        
    def format_docs(self, docs) :
        #Print each retrieved chunk with its source and page info for debugging/inspection
        print("\nChunks Used for Answer (source, page):")
        for i, doc in enumerate(docs): #loop through each chunk
            source = doc.metadata.get("source", "unknown") #look up the name of pdf file, return unknown if it can't find
            page = doc.metadata.get("page", "unknown") #find the page number
            print(f"\n--- Chunk {i+1} ---") #print which chunk it is
            print(f"Source: {source}, Page: {page}")
            print(doc.page_content[:500])  # print first 500 chars for readability
        return "\n\n".join(doc.page_content for doc in docs) #join all chunks into a single string, separated by two newlines
    



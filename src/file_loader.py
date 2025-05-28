#%%
#load PDF files, preprocess text and split into chunks

from typing import Union, List, Literal 
import glob 
from tqdm import tqdm #progress bar to see files have been loaded
import multiprocessing #run on multiple CPU, speeds up loading PDFs
from langchain_community.document_loaders import PyPDFLoader #parses PDF files into list of documents
from langchain_text_splitters import RecursiveCharacterTextSplitter #splits text into smaller chunks

def remove_non_utf8_characters (text):  #Removes characters not basic English letters or symbols
    return ''. join (char for char in text if ord(char) < 128) #basic English characters

def load_pdf (pdf_file):  #Reads PDF using LangChain
    docs = PyPDFLoader (pdf_file, extract_images=True). load ()
    for doc in docs:
        doc.page_content = remove_non_utf8_characters (doc.page_content) #clean each page
    return docs

def get_num_cpu ():
    """Return the number of available CPU cores to set parallel processing"""
    return multiprocessing.cpu_count() 


class BaseLoader:
    def __init__ (self) -> None: #return None, just store number of CPU
        self.num_processes = get_num_cpu()
    #make placeholder to be overridden by child classes
    def __call__(self, files: List[str], **kwargs): #kwargs: collect additional parameters not strict with any parameters
        pass #kwargs helps future optional parameters can be added without modifying every subclass


class PDFLoader(BaseLoader):
    """Loads and cleans PDF documents using multiprocessing."""
    def __init__(self) -> None: #Inherits from BaseLoader
        super().__init__()

    def __call__(self, pdf_files: List[str], **kwargs): #loads PDFs with multiprocessing
        num_processes = min(self.num_processes, kwargs["workers"]) #Uses minimum of actual CPU cores to avoid overload
        with multiprocessing.Pool(processes=num_processes) as pool: #Creates a pool of processes to load files in parallel
            doc_loaded = []
            total_files = len(pdf_files)
            with tqdm(total=total_files, desc="Loading PDFs", unit="file") as pbar: #show progress bar
                for result in pool.imap_unordered(load_pdf, pdf_files): #load files in any order (faster)
                    doc_loaded.extend(result) #merge all results into one list
                    pbar.update(1) #update progress bar
        return doc_loaded
    


class TextSplitter: #splitting text into smaller parts
    def __init__(self, 
                separators: List[str] = ['\n\n', '\n', ' ', ''],
                chunk_size: int = 300,
                chunk_overlap: int = 0
                ) -> None:
        self.splitter = RecursiveCharacterTextSplitter (
            separators=separators, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
        )

    def __call__(self, documents): #returns a list of smaller chunks
        return self.splitter.split_documents(documents)


class Loader:
    def __init__(self,
                file_type: str = Literal["pdf"], #only pdf supported
                split_kwargs: dict = {
                    "chunk_size": 300,
                    "chunk_overlap": 0}
                ) -> None :
        assert file_type in ["pdf"], "file_type must be pdf"
        self.file_type = file_type
        if file_type == "pdf":
            self.doc_loader = PDFLoader ()
        else :
            raise ValueError("file_type must be pdf")
        self.doc_spltter = TextSplitter (**split_kwargs) #unpack the dictionary above of split_kwargs

    def load(self, pdf_files: Union[str, List[str]], workers: int = 1): #Set number of workers for parallel processing
        """ set workers = 1 to load files sequentially (default) to avoid crash, > 1 to load in parallel """
        if isinstance(pdf_files, str): #check if pdf_files is a string, then convert to list
            pdf_files = [pdf_files]
        pdf_files = pdf_files[:5] # Limit to 5 files for testing, remove later
        doc_loaded = self.doc_loader (pdf_files, workers=workers) #change number of workers if want to load in parallel
        doc_split = self.doc_spltter (doc_loaded) #split the loaded documents into chunks
        return doc_split

    def load_dir (self, dir_path: str, workers: int = 1):
        if self.file_type == "pdf":
            files = glob.glob (f"{dir_path}/*.pdf") #get all pdf files match the pattern
            assert len(files) > 0, f"No {self.file_type} files found in { dir_path}" #check if files list is not empty
        else:
            raise ValueError ("file_type must be pdf")
        return self.load(files, workers=workers)
# %%

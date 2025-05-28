#%%
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

# Import model
def get_hf_llm(model_name: str = "microsoft/phi-1_5",
               max_new_token: int = 256, #Max tokens the model can generate in a response
               **kwargs): #pass any additional options to the pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name) #load & turn text into tokens
    
    model = AutoModelForCausalLM.from_pretrained(
        """Load the model and predict the next token based on the previous tokens""",
        model_name,
        device_map="auto",            
        low_cpu_mem_usage=True        
    )

    model_pipeline = pipeline(
        """Create a pipeline: input token -> model -> decode ouput """,
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id, #Adding padding tokens so all inputs have same length
        device_map="auto"
    )

    llm = HuggingFacePipeline( #Wraps the Hugging Face pipeline inside a LangChain
        pipeline=model_pipeline,
        model_kwargs=kwargs #Passes any extra parameters
    )

    return llm
# %%

import re
"""Utility functions for processing text responses from LLMs can reused in different parts of the project."""
def extract_answer (text_response: str, #raw response from LLM
                    pattern: str = r"Answer:\s*(.*)" #find "Answer:" in the LLM response
                    ) -> str:
    match = re.search (pattern, text_response)
    if match:
        answer_text = match.group(1).strip() #extract only answer part from the LLM response
        return answer_text
    else:
        return "Answer not found."
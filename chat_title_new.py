from kgpedia import KGPediaModel

def get_chat_name(user_message: str, assistant_response: str) -> str:
    llm= KGPediaModel().get_model()
    query = f"User: {user_message}\nAssistant: {assistant_response}"
    prompt = f"""Summarize the user and assistant conversation in a few words. Give weightage to user query
    Conversation: {query}
    Instructions:
    - The title should be concise and capture the essence of the user query
    - The title should not be too long
    - Make sure that is more relevant to the query and not the response because chat title is based on user query
    - If the user query is a simple greeting or a general query, you can return "General Conversation" as the title
    - Do not return any other string, except the title strictly. It is a serious matter, to keep the title clean and professional.
    """
    title = llm.complete(prompt)
    chat_title = title.text.strip("\n")
    return chat_title
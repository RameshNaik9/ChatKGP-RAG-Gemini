from fastapi import FastAPI, HTTPException, Request
from llama_index.core.bridge.pydantic import BaseModel
from chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from typing import Optional
from template import Template
from chat_title import get_chat_name
from kgpchatroom import KGPChatroomModel
from tags import get_tags
from question_recommendations import question_recommendations

import tracemalloc  
tracemalloc.start()

import os
import psutil
import uvicorn

import time

from llama_index.postprocessor.colbert_rerank import ColbertRerank

colbert_reranker = ColbertRerank(
    top_n=3,
    model="colbert-ir/colbertv2.0",
    tokenizer="colbert-ir/colbertv2.0",
    keep_retrieval_score=True,
)

from fastapi.middleware.cors import CORSMiddleware
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS configuration to allow requests from specific origins
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

template = Template.get_template()
LLM = KGPChatroomModel().get_model()

port = os.environ["PORT"]

chat_sessions = {}

class ChatRequest(BaseModel):
    conversation_id: str
    user_message: str
    chat_profile: str


class ChatResponse(BaseModel):
    conversation_id: str
    assistant_response: str
    chat_title: Optional[str] = None 
    tags_list: Optional[list] = None
    questions_list: Optional[list] = None
    # retrieved_sources: Optional[list] = None
    # retrieved_content: Optional[list] = None

import nest_asyncio
nest_asyncio.apply()

async def get_chat_engine(conversation_id: str, chat_profile: str) -> ContextChatEngine:
    # Initialize the session and title status if it doesn't exist
    if conversation_id not in chat_sessions:
        memory = ChatMemoryBuffer.from_defaults(token_limit=40000)
        fusionretriever = KGPChatroomModel().get_fusion_retriever(chat_profile=chat_profile)
        chat_engine = ContextChatEngine.from_defaults(retriever=fusionretriever, 
                                                      memory=memory, 
                                                      system_prompt=template,
                                                      node_postprocessors=[colbert_reranker]
                                                      )
        chat_sessions[conversation_id] = {"engine": chat_engine,"title_generated": False}  # Initialize title status to False
        
    return chat_sessions[conversation_id]["engine"]


@app.get("/")
def health_check():
    return {"message": "FastAPI Chat Assistant is running!",
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent}

from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend

@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend())
    
@app.post("/chat/{conversation_id}", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        user_message = request.user_message
        conversation_id = request.conversation_id
        chat_profile = request.chat_profile
        
        # Use await with get_chat_engine
        chat_engine = await get_chat_engine(conversation_id, chat_profile)
        
        # Use the async version of chat if available
        response = chat_engine.chat(user_message)

        # Generate title only if it hasn't been generated yet
        title = None
        if not chat_sessions[conversation_id]['title_generated']:
            title = get_chat_name(user_message, response)
            # title = chat_title(user_message, response)
            chat_sessions[conversation_id]['title_generated'] = True  # Set to True after generating title

        history = chat_engine.chat_history
        tags_list, _ = get_tags(history,LLM)
        questions_list, _ = question_recommendations(history,LLM) 

        # retrieved_nodes = KGPChatroomModel().get_retriever(chat_profile=chat_profile).retrieve(user_message)
        # sources=[]
        # information=[]
        # for node in retrieved_nodes:
        #     sources.append(node.metadata)
        #     information.append(node.text)
        # Create response object
        response_data = ChatResponse(
            conversation_id=conversation_id,
            assistant_response=str(response),  # Remove newline characters
            chat_title=title,  # Return title only if it was generated
            tags_list = tags_list,
            questions_list = questions_list,
            # retrieved_sources=sources,
            # retrieved_content = information
        )

        return response_data

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_delete/{conversation_id}")
async def delete_chat(conversation_id: str):
    try:
        # Check if the conversation ID exists in chat_sessions
        if conversation_id in chat_sessions:
            del chat_sessions[conversation_id]  # Delete the entire session
            return {"message": f"The conversation {conversation_id} has been deleted successfully ♻️"}
        else:
            raise KeyError  # Raise an error if conversation ID does not exist
    except KeyError:
        logger.warning(f"Attempt to reset non-existing conversation ID: {conversation_id}")
        raise HTTPException(status_code=404, detail=f"Conversation ID {conversation_id} does not exist or has already been deleted 🗑️")
    except Exception as e:
        logger.error(f"Unexpected error in delete_chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chats/delete_all")
def master_reset():
    try:
        if chat_sessions:
            chat_sessions.clear()  # Clear all chat sessions
            return {"message": "All chat conversations have been deleted successfully! 😈"}
        else:
            raise HTTPException(status_code=404, detail="No active chat conversations to delete 😕")
    except Exception as e:
        logger.error(f"Unexpected error in master_reset endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(port), reload=False)

# To run the app:
# Use: uvicorn main:app --reload
# startup command: uvicorn main:app --host 0.0.0.0 --port 8000
# python main.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from typing import Optional

#Custom functions
from chat_engine_new import ContextChatEngine
from template_new import Template
from chat_title_new import get_chat_name
from kgpedia import KGPediaModel
from tags import get_tags
from question_recommendations import question_recommendations

import tracemalloc
import os
import psutil
import uvicorn
import time
import logging
import nest_asyncio

# Initialize tracing for memory usage
tracemalloc.start()

# Apply nest_asyncio for nested loops
nest_asyncio.apply()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# CORS middleware to allow requests from specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load template and language model
template = Template.get_template()
LLM = KGPediaModel().get_model()

# Retrieve port from environment variables
port = os.environ["PORT"]

# Chat sessions storage
chat_sessions = {}

# ColBERT Reranker setup
colbert_reranker = ColbertRerank(top_n=3,model="colbert-ir/colbertv2.0",tokenizer="colbert-ir/colbertv2.0",keep_retrieval_score=True)

# Pydantic models for request and response
class ChatRequest(BaseModel):
    conversation_id: str
    user_message: str
    chat_profile: str


class ChatResponse(BaseModel):
    conversation_id: str
    assistant_response: str
    # chat_title: Optional[str] = None
    # tags_list: Optional[list] = None
    # questions_list: Optional[list] = None


# Health check endpoint
@app.get("/")
def health_check():
    return {
        "message": "FastAPI Chat Assistant is running!",
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
    }


# Helper function to get or create chat engine
async def get_chat_engine(conversation_id: str, chat_profile: str) -> ContextChatEngine:
    if conversation_id not in chat_sessions:
        memory = ChatMemoryBuffer.from_defaults(token_limit=40000)
        fusion_retriever = KGPediaModel().get_fusion_retriever(chat_profile=chat_profile)
        chat_engine = ContextChatEngine.from_defaults(
            retriever=fusion_retriever,
            memory=memory,
            system_prompt=template,
            node_postprocessors=[colbert_reranker],
        )
        chat_sessions[conversation_id] = {"engine": chat_engine, "title_generated": False}
    return chat_sessions[conversation_id]["engine"]


# Chat endpoint
@app.post("/chat/{conversation_id}", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        user_message = request.user_message
        conversation_id = request.conversation_id
        chat_profile = request.chat_profile

        # Get chat engine
        chat_engine = await get_chat_engine(conversation_id, chat_profile)

        # Generate response
        response = chat_engine.chat(user_message)

        # # Generate chat title if not already generated
        # title = None
        # if not chat_sessions[conversation_id]['title_generated']:
        #     title = get_chat_name(user_message, response)
        #     chat_sessions[conversation_id]['title_generated'] = True

        # Retrieve history, tags, and recommendations
        # history = chat_engine.chat_history
        # tags_list, _ = get_tags(history, LLM)
        # questions_list, _ = question_recommendations(history, LLM)

        # Create response object
        response_data = ChatResponse(
            conversation_id=conversation_id,
            assistant_response=str(response),
            # chat_title=title,
            # tags_list=tags_list,
            # questions_list=questions_list,
        )

        return response_data

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Delete specific chat endpoint
@app.post("/chat_delete/{conversation_id}")
async def delete_chat(conversation_id: str):
    try:
        if conversation_id in chat_sessions:
            del chat_sessions[conversation_id]
            return {"message": f"The conversation {conversation_id} has been deleted successfully ♻️"}
        else:
            raise KeyError
    except KeyError:
        logger.warning(f"Attempt to delete non-existing conversation ID: {conversation_id}")
        raise HTTPException(
            status_code=404, detail=f"Conversation ID {conversation_id} does not exist or has already been deleted 🗑️"
        )
    except Exception as e:
        logger.error(f"Unexpected error in delete_chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Delete all chats endpoint
@app.delete("/chats/delete_all")
def master_reset():
    try:
        if chat_sessions:
            chat_sessions.clear()
            return {"message": "All chat conversations have been deleted successfully! 😈"}
        else:
            raise HTTPException(status_code=404, detail="No active chat conversations to delete 😕")
    except Exception as e:
        logger.error(f"Unexpected error in master_reset endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Middleware to add process time header
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Run the app
if __name__ == "__main__":
    uvicorn.run("new_main:app", host="0.0.0.0", port=int(port), reload=False)
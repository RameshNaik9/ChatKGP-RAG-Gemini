import os
import time
import docx2txt
import threading
from enum import Enum
from typing import Any, Dict, Optional, Union, List
from pinecone import Pinecone, ServerlessSpec
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.readers.json import JSONReader
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.prompts import PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.chat_engine.types import AgentChatResponse, StreamingAgentChatResponse
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.schema import NodeWithScore
from llama_index.core.bridge.pydantic import BaseModel, Field, field_serializer
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    MessageRole,
)
from llama_index.core.base.response.schema import (
    StreamingResponse,
    AsyncStreamingResponse,
)
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
    BaseChatEngine,
    StreamingAgentChatResponse,
    ToolOutput,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import BaseMemory, ChatMemoryBuffer
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.chat_engine.utils import (
    # get_prefix_messages_with_context,
    get_response_synthesizer
)

pc = Pinecone(api_key="a217323a-0dbc-4051-9068-f3a8e5a05ee6")
pinecone_index = pc.Index("kgpedia")

Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004",api_key="AIzaSyDTuz-H3XT0tR03slA9kq8_nE5Yyj8gf64") # set the embedding model, add models/ before the model name
Settings.llm = Gemini(model_name="models/gemini-1.5-flash-002",temperature=1,api_key="AIzaSyDTuz-H3XT0tR03slA9kq8_nE5Yyj8gf64")

def get_prefix_messages_with_context(
    context_template: PromptTemplate,
    system_prompt: str,
    prefix_messages: List[ChatMessage],
    chat_history: List[ChatMessage],
    llm_metadata_system_role: MessageRole,
) -> List[ChatMessage]:
    context_str_w_sys_prompt = context_template + system_prompt.strip()
    return [
        ChatMessage(content=context_str_w_sys_prompt, role=llm_metadata_system_role),
        *prefix_messages,
        *chat_history,
        ChatMessage(content="{query_str}", role=MessageRole.USER),
    ]

DEFAULT_CONTEXT_TEMPLATE = (
    "Use the context information below to assist the user."
    "\n--------------------\n"
    "{context_str}"
    "\n--------------------\n"
)

DEFAULT_REFINE_TEMPLATE = (
    "Using the context below, refine the following existing answer using the provided context to assist the user.\n"
    "If the context isn't helpful, just repeat the existing answer and nothing more.\n"
    "\n--------------------\n"
    "{context_msg}"
    "\n--------------------\n"
    "Existing Answer:\n"
    "{existing_answer}"
    "\n--------------------\n"
)


class ContextChatEngine(BaseChatEngine):
    """
    Context Chat Engine.

    Uses a retriever to retrieve a context, set the context in the system prompt,
    and then uses an LLM to generate a response, for a fluid chat experience.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: LLM,
        memory: BaseMemory,
        prefix_messages: List[ChatMessage],
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        context_template: Optional[str] = None,
        context_refine_template: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._memory = memory
        self._prefix_messages = prefix_messages
        self._node_postprocessors = node_postprocessors or []
        self._context_template = context_template or DEFAULT_CONTEXT_TEMPLATE
        self._context_refine_template = (
            context_refine_template or DEFAULT_REFINE_TEMPLATE
        )

        self.callback_manager = callback_manager or CallbackManager([])
        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = self.callback_manager

    @classmethod
    def from_defaults(
        cls,
        retriever: BaseRetriever,
        chat_history: Optional[List[ChatMessage]] = None,
        memory: Optional[BaseMemory] = None,
        system_prompt: Optional[str] = None,
        prefix_messages: Optional[List[ChatMessage]] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        context_template: Optional[str] = None,
        context_refine_template: Optional[str] = None,
        llm: Optional[LLM] = None,
        **kwargs: Any,
    ) -> "ContextChatEngine":
        """Initialize a ContextChatEngine from default parameters."""
        llm = llm or Settings.llm

        chat_history = chat_history or []
        memory = memory or ChatMemoryBuffer.from_defaults(
            chat_history=chat_history, token_limit=llm.metadata.context_window - 256
        )

        if system_prompt is not None:
            if prefix_messages is not None:
                raise ValueError(
                    "Cannot specify both system_prompt and prefix_messages"
                )
            prefix_messages = [
                ChatMessage(content=system_prompt, role=llm.metadata.system_role)
            ]

        prefix_messages = prefix_messages or []
        node_postprocessors = node_postprocessors or []

        return cls(
            retriever,
            llm=llm,
            memory=memory,
            prefix_messages=prefix_messages,
            node_postprocessors=node_postprocessors,
            callback_manager=Settings.callback_manager,
            context_template=context_template,
            context_refine_template=context_refine_template,
        )

    def _get_nodes(self, message: str) -> List[NodeWithScore]:
        """Generate context information from a message."""
        nodes = self._retriever.retrieve(message)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )

        return nodes

    async def _aget_nodes(self, message: str) -> List[NodeWithScore]:
        """Generate context information from a message."""
        nodes = await self._retriever.aretrieve(message)
        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(
                nodes, query_bundle=QueryBundle(message)
            )

        return nodes

    def _get_response_synthesizer(
        self, chat_history: List[ChatMessage], streaming: bool = False
    ) -> CompactAndRefine:
        # Pull the system prompt from the prefix messages
        system_prompt = ""
        prefix_messages = self._prefix_messages
        if (
            len(self._prefix_messages) != 0
            and self._prefix_messages[0].role == MessageRole.SYSTEM
        ):
            system_prompt = str(self._prefix_messages[0].content)
            prefix_messages = self._prefix_messages[1:]

        # Get the messages for the QA and refine prompts
        qa_messages = get_prefix_messages_with_context(
            self._context_template,
            system_prompt,
            prefix_messages,
            chat_history,
            self._llm.metadata.system_role,
        )
        refine_messages = get_prefix_messages_with_context(
            self._context_refine_template,
            system_prompt,
            prefix_messages,
            chat_history,
            self._llm.metadata.system_role,
        )

        # Get the response synthesizer
        return get_response_synthesizer(
            self._llm, self.callback_manager, qa_messages, refine_messages, streaming
        )

    def _create_chat_message_with_timestamp(
    self, content: str, role: MessageRole
) -> ChatMessage:
        return ChatMessage(
        content=content,
        role=role,
        additional_kwargs={"timestamp": int(time.time())}
    )

    @trace_method("chat")
    def chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks: Optional[List[NodeWithScore]] = None,
    ) -> AgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        # Record the Unix timestamp for when the user sends the message
        user_timestamp = int(time.time())  # Unix timestamp for user message

        # get nodes and postprocess them
        nodes = self._get_nodes(message)
        if len(nodes) == 0 and prev_chunks is not None:
            nodes = prev_chunks

        # Get the response synthesizer with dynamic prompts
        chat_history = self._memory.get(input=message)
        synthesizer = self._get_response_synthesizer(chat_history)

        # Generate the assistant's response
        response = synthesizer.synthesize(message, nodes)

        # Record the Unix timestamp for when the assistant responds
        assistant_timestamp = int(time.time())  # Unix timestamp for assistant response

        # Create the user message with user_timestamp in additional_kwargs
        user_message = ChatMessage(
            content=message,
            role=MessageRole.USER,
            additional_kwargs={"user_timestamp": user_timestamp}
        )

        # Create the assistant message with assistant_timestamp in additional_kwargs
        ai_message = ChatMessage(
            content=str(response),
            role=MessageRole.ASSISTANT,
            additional_kwargs={"assistant_timestamp": assistant_timestamp}
        )

        # Store messages in memory
        self._memory.put(user_message)
        self._memory.put(ai_message)

        # Return the response, wrapped in AgentChatResponse
        return AgentChatResponse(
            response=str(response),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(nodes),
                    raw_input={"message": message},
                    raw_output=nodes,
                )
            ],
            source_nodes=nodes,
        )

    @trace_method("chat")
    def stream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks: Optional[List[NodeWithScore]] = None,
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        # Record the Unix timestamp for when the user sends the message
        user_timestamp = int(time.time())  # Unix timestamp for user message

        # get nodes and postprocess them
        nodes = self._get_nodes(message)
        if len(nodes) == 0 and prev_chunks is not None:
            nodes = prev_chunks

        # Get the response synthesizer with dynamic prompts
        chat_history = self._memory.get(input=message)
        synthesizer = self._get_response_synthesizer(chat_history, streaming=True)

        response = synthesizer.synthesize(message, nodes)
        assert isinstance(response, StreamingResponse)

        # Create the user message with user_timestamp in additional_kwargs
        user_message = ChatMessage(
            content=message,
            role=MessageRole.USER,
            additional_kwargs={"user_timestamp": user_timestamp}
        )

        # Store the user message in memory before streaming starts
        self._memory.put(user_message)

        def wrapped_gen(response: StreamingResponse) -> ChatResponseGen:
            full_response = ""
            assistant_timestamp = None
            for token in response.response_gen:
                full_response += token

                # Record the Unix timestamp when the first token is returned
                if assistant_timestamp is None:
                    assistant_timestamp = int(time.time())

                yield ChatResponse(
                    message=ChatMessage(
                        content=full_response, role=MessageRole.ASSISTANT
                    ),
                    delta=token,
                )

            # Create the assistant message with assistant_timestamp in additional_kwargs
            ai_message = ChatMessage(
                content=full_response,
                role=MessageRole.ASSISTANT,
                additional_kwargs={"assistant_timestamp": assistant_timestamp}
            )

            # Store the assistant message in memory
            self._memory.put(ai_message)

        return StreamingAgentChatResponse(
            chat_stream=wrapped_gen(response),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(nodes),
                    raw_input={"message": message},
                    raw_output=nodes,
                )
            ],
            source_nodes=nodes,
            is_writing_to_memory=False,
        )


    @trace_method("chat")
    async def achat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks: Optional[List[NodeWithScore]] = None,
    ) -> AgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)

        # get nodes and postprocess them
        nodes = await self._aget_nodes(message)
        if len(nodes) == 0 and prev_chunks is not None:
            nodes = prev_chunks

        # Get the response synthesizer with dynamic prompts
        chat_history = self._memory.get(
            input=message,
        )
        synthesizer = self._get_response_synthesizer(chat_history)

        response = await synthesizer.asynthesize(message, nodes)
        user_message = ChatMessage(content=message, role=MessageRole.USER)
        ai_message = ChatMessage(content=str(response), role=MessageRole.ASSISTANT)

        await self._memory.aput(user_message)
        await self._memory.aput(ai_message)

        return AgentChatResponse(
            response=str(response),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(nodes),
                    raw_input={"message": message},
                    raw_output=nodes,
                )
            ],
            source_nodes=nodes,
        )

    @trace_method("chat")
    async def astream_chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        prev_chunks: Optional[List[NodeWithScore]] = None,
    ) -> StreamingAgentChatResponse:
        if chat_history is not None:
            self._memory.set(chat_history)
        # get nodes and postprocess them
        nodes = await self._aget_nodes(message)
        if len(nodes) == 0 and prev_chunks is not None:
            nodes = prev_chunks

        # Get the response synthesizer with dynamic prompts
        chat_history = self._memory.get(
            input=message,
        )
        synthesizer = self._get_response_synthesizer(chat_history, streaming=True)

        response = await synthesizer.asynthesize(message, nodes)
        assert isinstance(response, AsyncStreamingResponse)

        async def wrapped_gen(response: AsyncStreamingResponse) -> ChatResponseAsyncGen:
            full_response = ""
            async for token in response.async_response_gen():
                full_response += token
                yield ChatResponse(
                    message=ChatMessage(
                        content=full_response, role=MessageRole.ASSISTANT
                    ),
                    delta=token,
                )

            user_message = ChatMessage(content=message, role=MessageRole.USER)
            ai_message = ChatMessage(content=full_response, role=MessageRole.ASSISTANT)
            await self._memory.aput(user_message)
            await self._memory.aput(ai_message)

        return StreamingAgentChatResponse(
            achat_stream=wrapped_gen(response),
            sources=[
                ToolOutput(
                    tool_name="retriever",
                    content=str(nodes),
                    raw_input={"message": message},
                    raw_output=nodes,
                )
            ],
            source_nodes=nodes,
            is_writing_to_memory=False,
        )

    def reset(self) -> None:
        self._memory.reset()

    @property
    def chat_history(self) -> List[ChatMessage]:
        """Get chat history with Unix timestamps for user and assistant."""

        history = self._memory.get_all()

        # Ensure that each message has the correct additional_kwargs with Unix timestamp
        updated_history = []
        for message in history:
            if message.role == MessageRole.USER:
                # Check if user_timestamp exists, if not add it
                if "user_timestamp" not in message.additional_kwargs:
                    message.additional_kwargs["user_timestamp"] = int(time.time())
            elif message.role == MessageRole.ASSISTANT:
                # Check if assistant_timestamp exists, if not add it
                if "assistant_timestamp" not in message.additional_kwargs:
                    message.additional_kwargs["assistant_timestamp"] = int(time.time())

            updated_history.append(message)

        return updated_history
    
storage_context = StorageContext.from_defaults(persist_dir="pinecone index\Career", vector_store=PineconeVectorStore(pinecone_index=pinecone_index))
pc_index = load_index_from_storage(storage_context)

from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

bm25_retriever = BM25Retriever.from_defaults(
    index=pc_index,
    similarity_top_k=5,  # Increased from 1
    stemmer=Stemmer.Stemmer('english'),
    language='english',
    verbose=True
)

vector_retriever = pc_index.as_retriever(similarity_top_k=8)  # Increased from 5

import asyncio
from enum import Enum
from typing import Dict, List, Optional, Tuple, cast

from llama_index.core.async_utils import run_async_tasks
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.llms.utils import LLMType, resolve_llm
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import IndexNode, NodeWithScore, QueryBundle
from llama_index.core.settings import Settings

QUERY_GEN_PROMPT = (
    "You are a helpful assistant for campus placements at IIT Kharagpur that generates multiple search queries based on a "
    "single input query. Remember that the following query would be asked by a student in the context of clarifying any doubt. Keep the genre limited to placements, interviews, general advices. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)


class FUSION_MODES(str, Enum):
    """Enum for different fusion modes."""
    RECIPROCAL_RANK = "reciprocal_rerank" 
    RELATIVE_SCORE = "relative_score" 
    DIST_BASED_SCORE = "dist_based_score" 
    SIMPLE = "simple" 


class QueryFusionRetriever(BaseRetriever):
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        llm: Optional[LLMType] = None,
        query_gen_prompt: Optional[str] = None,
        mode: FUSION_MODES = FUSION_MODES.SIMPLE,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        num_queries: int = 4,
        use_async: bool = True,
        verbose: bool = False,
        callback_manager: Optional[CallbackManager] = None,
        objects: Optional[List[IndexNode]] = None,
        object_map: Optional[dict] = None,
        retriever_weights: Optional[List[float]] = None,
    ) -> None:
        self.num_queries = num_queries
        self.query_gen_prompt = query_gen_prompt or QUERY_GEN_PROMPT
        self.similarity_top_k = similarity_top_k
        self.mode = mode
        self.use_async = use_async

        self._retrievers = retrievers
        if retriever_weights is None:
            self._retriever_weights = [1.0 / len(retrievers)] * len(retrievers)
        else:
        
            total_weight = sum(retriever_weights)
            self._retriever_weights = [w / total_weight for w in retriever_weights]
        self._llm = (
            resolve_llm(llm, callback_manager=callback_manager) if llm else Settings.llm
        )
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            objects=objects,
            verbose=verbose,
        )

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"query_gen_prompt": PromptTemplate(self.query_gen_prompt)}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "query_gen_prompt" in prompts:
            self.query_gen_prompt = cast(
                PromptTemplate, prompts["query_gen_prompt"]
            ).template

    def _get_queries(self, original_query: str) -> List[QueryBundle]:
        prompt_str = self.query_gen_prompt.format(
            num_queries=self.num_queries - 1,
            query=original_query,
        )
        response = self._llm.complete(prompt_str)

    
        queries = response.text.split("\n")
        queries = [q.strip() for q in queries if q.strip()]
        if self._verbose:
            queries_str = "\n".join(queries)
            print(f"Generated queries:\n{queries_str}")

    
        return [QueryBundle(q) for q in queries[: self.num_queries - 1]]

    def _reciprocal_rerank_fusion(
        self, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        k = 60.0
        fused_scores = {}
        hash_to_node = {}

        for nodes_with_scores in results.values():
            for rank, node_with_score in enumerate(
                sorted(nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True)
            ):
                hash = node_with_score.node.hash
                hash_to_node[hash] = node_with_score
                if hash not in fused_scores:
                    fused_scores[hash] = 0.0
                fused_scores[hash] += 1.0 / (rank + k)

        reranked_results = dict(
            sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        )

        reranked_nodes: List[NodeWithScore] = []
        for hash, score in reranked_results.items():
            reranked_nodes.append(hash_to_node[hash])
            reranked_nodes[-1].score = score

        return reranked_nodes

    def _relative_score_fusion(
        self,
        results: Dict[Tuple[str, int], List[NodeWithScore]],
        dist_based: Optional[bool] = False,
    ) -> List[NodeWithScore]:
        """Apply relative score fusion."""
        min_max_scores = {}
        for query_tuple, nodes_with_scores in results.items():
            if not nodes_with_scores:
                min_max_scores[query_tuple] = (0.0, 0.0)
                continue
            scores = [
                node_with_score.score or 0.0 for node_with_score in nodes_with_scores
            ]
            if dist_based:
                mean_score = sum(scores) / len(scores)
                std_dev = (
                    sum((x - mean_score) ** 2 for x in scores) / len(scores)
                ) ** 0.5
                min_score = mean_score - 3 * std_dev
                max_score = mean_score + 3 * std_dev
            else:
                min_score = min(scores)
                max_score = max(scores)
            min_max_scores[query_tuple] = (min_score, max_score)

        for query_tuple, nodes_with_scores in results.items():
            for node_with_score in nodes_with_scores:
                min_score, max_score = min_max_scores[query_tuple]
                if max_score == min_score:
                    node_with_score.score = 1.0 if max_score > 0 else 0.0
                else:
                    node_with_score.score = (node_with_score.score - min_score) / (
                        max_score - min_score
                    )
                retriever_idx = query_tuple[1]
                existing_score = node_with_score.score or 0.0
                node_with_score.score = (
                    existing_score * self._retriever_weights[retriever_idx]
                )
                node_with_score.score /= self.num_queries

        all_nodes: Dict[str, NodeWithScore] = {}

        for nodes_with_scores in results.values():
            for node_with_score in nodes_with_scores:
                hash = node_with_score.node.hash
                if hash in all_nodes:
                    cur_score = all_nodes[hash].score or 0.0
                    all_nodes[hash].score = cur_score + (node_with_score.score or 0.0)
                else:
                    all_nodes[hash] = node_with_score

        return sorted(all_nodes.values(), key=lambda x: x.score or 0.0, reverse=True)

    def _simple_fusion(
        self, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        """Apply simple fusion."""
        all_nodes: Dict[str, NodeWithScore] = {}
        for nodes_with_scores in results.values():
            for node_with_score in nodes_with_scores:
                hash = node_with_score.node.hash
                if hash in all_nodes:
                    max_score = max(
                        node_with_score.score or 0.0, all_nodes[hash].score or 0.0
                    )
                    all_nodes[hash].score = max_score
                else:
                    all_nodes[hash] = node_with_score

        return sorted(all_nodes.values(), key=lambda x: x.score or 0.0, reverse=True)

    def _run_nested_async_queries(
        self, queries: List[QueryBundle]
    ) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        tasks, task_queries = [], []
        for query in queries:
            for i, retriever in enumerate(self._retrievers):
                tasks.append(retriever.aretrieve(query))
                task_queries.append((query.query_str, i))

        task_results = run_async_tasks(tasks)

        results = {}
        for query_tuple, query_result in zip(task_queries, task_results):
            results[query_tuple] = query_result

        return results

    async def _run_async_queries(
        self, queries: List[QueryBundle]
    ) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        tasks, task_queries = [], []
        for query in queries:
            for i, retriever in enumerate(self._retrievers):
                tasks.append(retriever.aretrieve(query))
                task_queries.append((query.query_str, i))

        task_results = await asyncio.gather(*tasks)

        results = {}
        for query_tuple, query_result in zip(task_queries, task_results):
            results[query_tuple] = query_result

        return results

    def _run_sync_queries(
        self, queries: List[QueryBundle]
    ) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        results = {}
        for query in queries:
            for i, retriever in enumerate(self._retrievers):
                results[(query.query_str, i)] = retriever.retrieve(query)

        return results

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        if self.use_async:
            results = self._run_nested_async_queries(queries)
        else:
            results = self._run_sync_queries(queries)

        if self.mode == FUSION_MODES.RECIPROCAL_RANK:
            return self._reciprocal_rerank_fusion(results)[: self.similarity_top_k]
        elif self.mode == FUSION_MODES.RELATIVE_SCORE:
            return self._relative_score_fusion(results)[: self.similarity_top_k]
        elif self.mode == FUSION_MODES.DIST_BASED_SCORE:
            return self._relative_score_fusion(results, dist_based=True)[
                : self.similarity_top_k
            ]
        elif self.mode == FUSION_MODES.SIMPLE:
            return self._simple_fusion(results)[: self.similarity_top_k]
        else:
            raise ValueError(f"Invalid fusion mode: {self.mode}")

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        queries: List[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        results = await self._run_async_queries(queries)

        if self.mode == FUSION_MODES.RECIPROCAL_RANK:
            return self._reciprocal_rerank_fusion(results)[: self.similarity_top_k]
        elif self.mode == FUSION_MODES.RELATIVE_SCORE:
            return self._relative_score_fusion(results)[: self.similarity_top_k]
        elif self.mode == FUSION_MODES.DIST_BASED_SCORE:
            return self._relative_score_fusion(results, dist_based=True)[
                : self.similarity_top_k
            ]
        elif self.mode == FUSION_MODES.SIMPLE:
            return self._simple_fusion(results)[: self.similarity_top_k]
        else:
            raise ValueError(f"Invalid fusion mode: {self.mode}")
        

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=3, 
    mode="reciprocal_rerank",
    use_async=False, 
    verbose=True
)

template = """
You are a final-semester senior at IIT Kharagpur, experienced in navigating the career-related processes, including internships and placements. Your task is to provide precise, knowledgeable, and helpful responses to students regarding career guidance. You specialize in internships, placements, profile preparation, and addressing student queries about the CDC (Career Development Centre) at IIT Kharagpur.

The Career Development Centre (CDC) at IIT Kharagpur assists students in securing internships and placements across a variety of domains such as Software Engineering, Data Science, Product Management, Finance, FMCG, and Consulting. These processes are conducted in two phases: Phase 1 (Autumn Semester, usually starting in December for placements) and Phase 2 (Spring Semester).

Your responses should:
- Be **clear, concise, and actionable**, offering insights based on your personal experience or general knowledge at IIT Kharagpur.
- Maintain a **helpful senior-like tone**, as if you're giving advice to a peer.
- Stick to **career-related queries**: internships, placements, profile-building, and CDC processes. If the query is irrelevant, politely steer the conversation back to the relevant topic.
- If the query involves the **Phase 1 and Phase 2 placement/internship process**, provide guidance based on the typical timeline, preparation strategies, and selection procedures.
- Refer to common experiences or knowledge from the extensive documentation of IIT Kharagpur students and alumni, as needed, to support your answers.
- In case of queries about **profile building**, offer tips on how students can improve their resumes and applications, highlighting skills, projects, and extracurricular activities relevant to different domains (software, product management, data science, etc.).

Always remain **student-centric**, prioritizing actionable, relevant advice for those who are about to undergo or are currently preparing for internships or placements.

**Contextual Reminders**:
- Answering queries related to **placement processes, CDC guidelines, or preparation strategies** for IIT Kharagpur students is your top priority.
- Answer within the scope of career-related topics (internships, placements, profile preparation, CDC, etc.). If the question deviates from this, politely guide the conversation back to the intended focus.
- If no relevant information is found in the retrieved content, answer truthfully or acknowledge the absence of specific information.
"""
memory =  ChatMemoryBuffer.from_defaults(token_limit=40000)
chat_engine = ContextChatEngine.from_defaults(retriever=retriever,memory=memory,system_prompt=template)

def chat_with_engine(chat_engine):
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        response = chat_engine.stream_chat(user_input)
        print("Chatbot: ", end="")
        for token in response.response_gen:
            print(token, end="")
        print()
        
chat_with_engine(chat_engine)
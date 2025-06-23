import os
import asyncio
import logging
import logging.config
import aiohttp
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import logger, set_verbose_debug
import tiktoken
# WORKING_DIR = "./dickens"
WORKING_DIR = "./Massage_10135"
# API配置
# DEEPSEEK_API_BASE = "https://api.deepseek.com/chat/completions"
DEEPSEEK_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
QWEN_EMBEDDING_MODEL = "text-embedding-v4"
# DEEPSEEK_CHAT_MODEL = "deepseek-reasoner"
DEEPSEEK_CHAT_MODEL = "deepseek-r1"

def split_text_into_chunks(text: str, max_tokens=200, overlap=50) -> list[str]:
    """将长文本按最大 token 分块并加重叠"""

    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = tokenizer.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap  # 有重叠
    return chunks

class QwenEmbedding:
    def __init__(self):
        # self.api_key ='sk-c1f2de78c13a455b806cf32648e36e25'
        self.api_key = 'sk-36930e681f094274964ffe6c51d62078'
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
            
    async def embed(self, texts: list[str]) -> np.ndarray:
        """自动分批请求嵌入"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async with.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        all_embeddings = []

        BATCH_SIZE = 10
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            payload = {
                "model": QWEN_EMBEDDING_MODEL,
                "input": batch
            }

            async with self.session.post(QWEN_API_BASE, headers=headers, json=payload) as response:
                if response.status != 200:
                    error = await response.text()
                    raise RuntimeError(f"DeepSeek API error: {error}")
                data = await response.json()
                batch_embeddings = [item["embedding"] for item in data["data"]]
                all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

class DeepSeekCompletion:
    """DeepSeek大模型接口"""
    def __init__(self):
        # self.api_key ='sk-c1f2de78c13a455b806cf32648e36e25'
        self.api_key = 'sk-36930e681f094274964ffe6c51d62078'
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
            
    async def complete(self, prompt: str, **kwargs) -> str:
        """获取模型补全结果"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async with.")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": DEEPSEEK_CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000)
        }
        
        async with self.session.post(
            f"{DEEPSEEK_API_BASE}",
            headers=headers,
            json=payload
        ) as response:
            if response.status != 200:
                error = await response.text()
                raise RuntimeError(f"DeepSeek API error: {error}")
                
            data = await response.json()
            return data["choices"][0]["message"]["content"]


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_demo.log"))

    print(f"\nLightRAG demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def Qwen_embed(texts: list[str]) -> np.ndarray:
    async with QwenEmbedding() as embedder:
        return await embedder.embed(texts)
Qwen_embed.embedding_dim = 1024

async def deepseek_complete(prompt: str, **kwargs) -> str:
    for _ in range(10):
        try:
            async with DeepSeekCompletion() as completer:
                return await completer.complete(prompt, **kwargs)
        except Exception as e:
            print(f"[Retry] DeepSeek Error: {e}")
            await asyncio.sleep(1)
    raise RuntimeError("DeepSeek failed after 3 retries.")

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=Qwen_embed,  
        llm_model_func=deepseek_complete,  
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

async def main():
    try:
        # 清理旧数据
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]
        for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleting old file:: {file_path}")

        # 初始化 RAG
        rag = await initialize_rag()
        await rag.aclear_cache()
        # 读取书籍文本
        with open("./book_10135.txt", "r", encoding="utf-8") as f:
            content = f.read()

        # 分块处理：chunk size = 500 tokens, overlap = 50
        chunks = split_text_into_chunks(content, max_tokens=500, overlap=50)
        print(f"Total chunks to insert: {len(chunks)}")

        # 每批最多嵌入 10 条，分批调用
        BATCH_SIZE = 10
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            await rag.ainsert(batch)
            print(f">> Inserted chunk batch {i // BATCH_SIZE + 1}")

    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")

import asyncio
import numpy as np
import re
from lightrag import LightRAG
from lightrag.base import QueryParam
from custom_rag_processor import QwenEmbedding, DeepSeekCompletion

class MassageAcupointRAG:
    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.async_embed.embedding_dim = 1024
        self.rag = LightRAG(
            working_dir=working_dir,
            embedding_func=self.async_embed,
            llm_model_func=self.async_complete
        )

    @staticmethod
    async def async_embed(texts: list[str]) -> np.ndarray:
        async with QwenEmbedding() as embedder:
            return await embedder.embed(texts)

    @staticmethod
    async def async_complete(prompt: str, **kwargs) -> str:
        async with DeepSeekCompletion() as completer:
            return await completer.complete(prompt, **kwargs)

    @staticmethod
    def extract_acupoint_list(text: str) -> list[str]:
        """
        从 LLM 返回文本中提取穴位名称列表（形如 ['肩井穴', ...]）
        """
        pattern_list = re.search(r"\[([^\]]+?)\]", text)
        if pattern_list:
            inner_text = pattern_list.group(1)
            return re.findall(r"'([^']+?)'", inner_text)
        return []

    async def initialize(self):
        await self.rag.initialize_storages()

    async def query_acupoints(self, user_query: str) -> tuple[str, list[str]]:
        """
        执行 RAG 查询并提取穴位列表

        :param user_query: 用户输入的自然语言请求
        :return: (完整LLM回复, 穴位列表)
        """
        param = QueryParam(mode='naive', only_need_context=False)
        response = await self.rag.aquery(user_query, param)
        acupoints = self.extract_acupoint_list(str(response))
        return str(response), acupoints
    async def shutdown(self):
        await self.rag.finalize_storages()
    
if __name__ == "__main__":
    async def main():
        rag_client = MassageAcupointRAG(
            working_dir="C:/Users/ZIWEI/Documents/work/向量化/CoreRAG/Massage_10216"
        )
        try:
            await rag_client.initialize()

            query = (
                "我的大腿有些酸痛，请给出一些分布于腿的重点按摩穴位。"
                "在回答的最后，将重点穴位罗列为方便python脚本读取的list形式['XX穴','XX穴', ...]"
            )

            response, acupoints = await rag_client.query_acupoints(query)
            print(response)
            print(acupoints)

        finally:
            # 手动清理资源
            await rag_client.shutdown()
    asyncio.run(main())
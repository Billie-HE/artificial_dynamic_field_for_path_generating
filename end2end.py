from scripts.sorter import sorter
import numpy as np
from scripts.planner import planner
from CoreRAG.MassageAcupointRAG import MassageAcupointRAG
import asyncio

class autoGenerator():
    def __init__(self):
        self.sorter = sorter()
        self.myQuery = (
                "我的肩颈部有些酸痛，请给出一些分布于肩颈部的重点按摩穴位。"
                "在回答的最后，将重点穴位罗列为方便python脚本读取的list形式['XX穴','XX穴', ...]"
            )
        self.acupoints = []
        self.response = []
    async def query(self):
        self.rag_client = MassageAcupointRAG(
            working_dir="C:/Users/ZIWEI/Documents/work/向量化/CoreRAG/Massage_10216"
        )

        await self.rag_client.initialize()

        response, acupoints = await self.rag_client.query_acupoints(self.myQuery)

        print("\n=== RAG 回答 ===")
        print(response)
        print("\n=== 提取的穴位列表 ===")
        print(acupoints)

        self.acupoints = acupoints
        self.response = response

    def generate(self):
        asyncio.run(self.query())
        path_sequence = self.sorter.sort_acupoints(self.acupoints)
        print(f"path sequence:{path_sequence}")
        self.planner = planner(self.sorter.acupoints_metadata,path_sequence,total_steps=80,mode='line')
        full_path = self.planner.path_generator()
        for i in range(len(path_sequence) - 1):
            name_start = path_sequence[i]
            name_goal = path_sequence[i + 1]
            start = self.sorter.acupoints_metadata[name_start]["pos"]
            goal = self.sorter.acupoints_metadata[name_goal]["pos"]

            print(f"规划路径: {name_start} {start} ➝ {name_goal} {goal}")
        print("完整按摩路径点:", full_path)
        self.rag_client.rag = None # 清理内存
        self.rag_client = None

if __name__ == '__main__':
    myGenerator = autoGenerator()
    myGenerator.generate()
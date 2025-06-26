from scripts.sorter import sorter
import numpy as np
from scripts.planner import planner
from CoreRAG.MassageAcupointRAG import MassageAcupointRAG
import asyncio

class autoGenerator():
    def __init__(self):
        self.myQuery = (
                "我的背部有些酸痛，请给出一些分布于背部的重点按摩穴位。"
                "在回答的最后，将重点穴位罗列为方便python脚本读取的list形式['XX穴','XX穴', ...]"
            )
        self.acupoints = []
        self.response = []
        self.body_part = 'back' # 默认部位在back          需要对接接口1
        self.sorter = sorter(body_part=self.body_part)
        self.massage_head = None#                        需要对接接口2
        self.massage_name = None#                        需要对接接口3
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
        self.planner = planner(self.sorter.acupoints_metadata,path_sequence,time_scale = 2.0)
        full_path = self.planner.path_generator()
        queue = self.planner.replanning()
        json_form = self.planner.convert_queue_to_task_plan(queue,self.body_part,self.massage_head,self.massage_name)
        self.rag_client.rag = None # 清理内存
        self.rag_client = None
        return json_form

if __name__ == '__main__':
    myGenerator = autoGenerator()
    json_form = myGenerator.generate()
    print(json_form)
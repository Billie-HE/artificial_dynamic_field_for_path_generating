try:from scripts.APF_global_demo import GaussianSchedule,GaussianPathSchedule,TimedGaussianSchedule,FieldScheduler,Agent,Visualizer
except:from APF_global_demo import GaussianSchedule,GaussianPathSchedule,TimedGaussianSchedule,FieldScheduler,Agent,Visualizer
import numpy as np
try:from scripts.sorter import sorter
except:from sorter import sorter
import matplotlib.pyplot as plt
import json
import time
import os
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文黑体（Windows/Linux）
plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号
class planner():
    def __init__(self,acupoints_dict:dict,sorted_path,time_scale = 1.0):
        self.scale = 5
        self.mid_x = 4
        self.raw_path = sorted_path
        self.acupoints_dict = acupoints_dict
        self.acupoints_dict_scaled = {
            name:{
                **info, # **info 内容和原来一样，但把Pos项乘以一个缩放系数
                "pos":tuple(np.array(info["pos"]) * self.scale)
            } for name, info in self.acupoints_dict.items()
        }
        self.path = []
        self.time_scale = time_scale
        self.vis = Visualizer((8 * self.scale,26 * self.scale),[])
        self.jump_pos = None
        self.jump_pos_name = None

    def _make_scheduler(self, start, goal, dist):
        start_duration = dist * 0.45
        end_duration = dist * 0.45
        return FieldScheduler([
            # 起点附近，早期激活、后期衰减
            TimedGaussianSchedule(
                center=start,
                height_fn=lambda t: -8 * (1 - (t - 0) / start_duration),
                sigma=1.5,
                t_start=0,
                t_end=start_duration,
                hold_final=True
            ),
            # 终点附近，后期激活、前期为0
            TimedGaussianSchedule(
                center=goal,
                height_fn=lambda t: -8 * ((t - end_duration) / end_duration),
                sigma=1.5,
                t_start=dist - end_duration,
                t_end=dist,
                hold_final=True
            ),
            # 中间常驻路径引导势场
            GaussianPathSchedule(
                start=start,
                end=goal,
                height_fn=lambda t:-4,
                sigma=2.0
            )
        ])
    
    def path_generator(self):
        self.path.clear()
        i = 0

        # --- 判断特殊情况：起始点是左侧唯一按摩点，右侧有对应对称点 ---
        if len(self.raw_path) >= 2:
            start_pos = self.acupoints_dict_scaled[self.raw_path[0]]['pos']
            left_coords = [
                self.acupoints_dict_scaled[name]['pos']
                for name in self.raw_path
                if self.acupoints_dict_scaled[name]['pos'][0] < self.mid_x * self.scale
            ]
            right_coords = [
                self.acupoints_dict_scaled[name]['pos']
                for name in self.raw_path
                if self.acupoints_dict_scaled[name]['pos'][0] >= self.mid_x * self.scale
            ]

            # 判断左侧是否只有一个点
            if len(left_coords) == 1 and len(right_coords) >= 1:
                # 判断起始点是否是左侧这个唯一点，且距离右侧至少有一个点对应
                max_x_left = left_coords[0][0]
                if np.allclose(start_pos, left_coords[0], atol=3.0):
                    print("检测到左侧只有一个点，右侧有对应对称点，采用特殊路径顺序")

                    # 生成路径：先左侧点停留若干（假设重复几次），再去右侧点停留
                    left_point = left_coords[0]
                    right_point = right_coords[0]  # 这里可以改成更合适的点，比如对称点或第一个右侧点

                    path = []
                    # 左侧点按10次（可调整）
                    path.extend([left_point] * 10)
                    # 右侧点按10次
                    path.extend([right_point] * 10)

                    self.path = path
                    return self.path  # 直接返回特殊路径，跳过默认流程
        # 回到正常流程
        while i < len(self.raw_path) - 1:
            name_start = self.raw_path[i]
            name_goal = self.raw_path[i + 1]
            start = self.acupoints_dict_scaled[name_start]["pos"]
            goal = self.acupoints_dict_scaled[name_goal]["pos"]

            # --- 生成场和agent ---
            start_array = np.array(start)
            goal_array = np.array(goal)
            dist = np.linalg.norm(goal_array-start_array)
            field_scheduler = self._make_scheduler(start, goal, dist)
            agent = Agent(start)

            # --- 执行路径引导 ---
            t = 0
            while True:
                Z = field_scheduler.get_field(t)
                agent.step(Z)
                if self.vis:
                    self.vis.show(Z=Z, agent_pos=agent.pos, t=t, start=start, goal=goal)

                self.path.append(tuple(agent.pos))

                if np.linalg.norm(agent.pos - np.array(goal)) <= 1.0 and t > dist*self.time_scale:
                    break

                t += 1

            # --- 最终可视化目标点（可选） ---
            if self.vis:
                self.vis.show(Z=Z, agent_pos=goal, t=t + 1, start=start, goal=goal)

            # --- 判断是否跳跃 ---
            final_pos = agent.pos

            left_coords = [
                self.acupoints_dict_scaled[name]["pos"]
                for name in self.raw_path
                if self.acupoints_dict_scaled[name]["pos"][0] < self.mid_x * self.scale
            ]

            if left_coords:
                max_x = max(p[0] for p in left_coords)
                candidates = [p for p in left_coords if p[0] == max_x]
                target_jump_pos = min(candidates, key=lambda p: p[1])
                self.jump_pos = target_jump_pos # 将跳跃位置添加入类
                print(f"flags:{target_jump_pos}")
                print(f"final_pos:{final_pos}")
                if np.allclose(final_pos, target_jump_pos, atol=3.0):
                    print(f"跳跃触发 @ {name_goal} → {self.raw_path[i + 2] if i + 2 < len(self.raw_path) else '终点'}")
                    i += 2  # 跳过一个点
                else:
                    i += 1
            else:
                i += 1

        return self.path

    
    def replanning(self):
        """
        将 self.path 中的坐标映射为穴位名 + 连续停留次数的队列（动态显示 + 竖屏）
        """
        # 异常处理
        if not self.path:
            print("无路径可规划，跳过replanning")
            return []
        name_pos_list = [(name, np.array(info['pos'])) for name, info in self.acupoints_dict_scaled.items()]

        # === 映射路径点为最近穴位名 === #
        mapped_names = []
        for pt in self.path:
            pt = np.array(pt)
            nearest_name = min(name_pos_list, key=lambda item: np.linalg.norm(item[1] - pt))[0]
            mapped_names.append(nearest_name)
            if np.array_equal(pt,self.jump_pos):
                self.jump_pos_name = nearest_name

        # === 统计连续按压次数 === #
        result_queue = []
        prev_name = mapped_names[0]
        count = 1
        for name in mapped_names[1:]:
            if name == prev_name:
                count += 1
            else:
                result_queue.append((prev_name, count))
                prev_name = name
                count = 1
        result_queue.append((prev_name, count))

        # === 动态可视化 === #
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 12))  # 竖屏显示

        grid_w, grid_h = map(int, self.vis.grid_size)
        ax.set_xlim(0, grid_w)
        ax.set_ylim(grid_h, 0)
        ax.invert_yaxis()

        for idx, (name, count) in enumerate(result_queue):
            pos = self.acupoints_dict_scaled[name]['pos']
            ax.plot(pos[0], pos[1], 'ro')
            ax.text(pos[0], pos[1], f"{idx+1}.{name} x{count}", fontsize=8, color='blue')

            if idx > 0:
                prev_pos = self.acupoints_dict_scaled[result_queue[idx - 1][0]]['pos']
                ax.plot([prev_pos[0], pos[0]], [prev_pos[1], pos[1]], 'g--')

            ax.set_title(f"第 {idx+1} 个穴位：{name} x{count}")
            plt.pause(0.5)  # 每步停留 0.5 秒

        plt.ioff()
        plt.show()
        print("连续穴位按压序列:")
        for item in result_queue:
            print(item)
        return result_queue
    def convert_queue_to_task_plan(self,queue,body_part,massage_head,massage_name):
        timestamp = int(time.time())
        if not massage_name:
            massage_name = f"{body_part}-{massage_head}-{timestamp}"
        if not massage_head:
            massage_head = f"finger"  # 默认值
        if not body_part:
            body_part = f"AI诊疗按摩部位"

        task_plan = []
        for i in range(len(queue)):
            name, stay_time = queue[i]
            # 添加停留任务 AA、BB、CC...
            task_plan.append({
                "start_point": name,
                "end_point": name,
                "time": stay_time,
                "path": "point"
            })

            # 添加移动任务 AB、BC、CD...，除了最后一个点不需要转移
            if i < len(queue) - 1:
                next_name = queue[i + 1][0]
                task_plan.append({
                    "start_point": name,
                    "end_point": next_name,
                    "time": None,
                    "path": "line"
                })
        # --- 构造完整项目结构 ---
        project_data = {
            f"{massage_name}": {
                "body_part": body_part,
                "can_delete": True,
                "choose_task": massage_head,
                "introduction": f"{body_part}-AI诊断理疗方案",
                "task_plan": task_plan
            }
        }
        # 打印格式化的 JSON 字符串
        print(json.dumps(project_data, ensure_ascii=False, indent=4))
        self.save_project_json(project_data)
        return project_data
    def save_project_json(self,project_data: dict, save_dir: str = "saved_projects", file_prefix: str = "massage_project"):
        """
        将 massage 项目字典保存为本地 JSON 文件
        :param project_data: 要保存的项目数据（dict）
        :param save_dir: 保存目录（默认 "saved_projects"）
        :param file_prefix: 文件名前缀（默认 "massage_project"）
        :return: 保存的完整文件路径
        """
        # 创建保存目录（如不存在）
        os.makedirs(save_dir, exist_ok=True)

        # 构造文件名：前缀 + 时间戳 + .json
        timestamp = int(time.time())
        file_name = f"{file_prefix}_{timestamp}.json"
        file_path = os.path.join(save_dir, file_name)

        # 写入 JSON 文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, ensure_ascii=False, indent=4)

        print(f"项目已保存到：{file_path}")
        return file_path
if __name__ == "__main__":
    test_text = ['风池穴', '肩井穴', '天柱穴', '大椎穴', '肩髃穴', '天宗穴']
    mySorter = sorter()
    sorted_path = mySorter.sort_acupoints(test_text)
    print(sorted_path)
    myPlanner = planner(mySorter.acupoints_metadata,sorted_path,80,'line')
    myPlanner.path_generator()
    myPlanner.replanning()
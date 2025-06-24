from scripts.APF_global_demo import GaussianSchedule,GaussianPathSchedule,TimedGaussianSchedule,FieldScheduler,Agent,Visualizer
import numpy as np
from scripts.sorter import sorter
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文黑体（Windows/Linux）
plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号
class planner():
    def __init__(self,acupoints_dict:dict,sorted_path,total_steps=50,mode='point'):
        self.scale = 5
        self.mid_x = 4
        self.raw_path = sorted_path
        self.Gaussian_mode = ['line','point']
        self.acupoints_dict = acupoints_dict
        self.acupoints_dict_scaled = {
            name:{
                **info, # **info 内容和原来一样，但把Pos项乘以一个缩放系数
                "pos":tuple(np.array(info["pos"]) * self.scale)
            } for name, info in self.acupoints_dict.items()
        }
        self.default_mode = mode
        self.path = []
        self.total_steps = total_steps
        self.vis = Visualizer((8 * self.scale,26 * self.scale),[])

    def _make_scheduler(self, mode, start, goal):
        if mode == 'point':
            schedule_list = [
                TimedGaussianSchedule(
                    center=start, 
                    height_fn=lambda t: -5 * (1 - (t / 20)), 
                    sigma=1.5, 
                    t_start=0, 
                    t_end=15, 
                    hold_final=True),
                TimedGaussianSchedule(
                    center=goal, 
                    height_fn=lambda t: -5 * (t / 20), 
                    sigma=1.5, 
                    t_start=25, 
                    t_end=40, 
                    hold_final=True),
            ]
        elif mode == 'line':
            schedule_list = [
                TimedGaussianSchedule(
                    center=start,
                    height_fn=lambda t: -5 * (1 - t / 30),
                    sigma=1.5,
                    t_start=0,
                    t_end=30,
                    hold_final=True
                ),
                TimedGaussianSchedule(
                    center=goal,
                    height_fn=lambda t: -5 * ((t - 50) / 30),
                    sigma=1.5,
                    t_start=50,
                    t_end=80,
                    hold_final=True
                ),
                GaussianPathSchedule(
                    start=start,
                    end=goal,
                    height_fn=lambda t: -4,
                    sigma=1.5
                )
            ]
        else:
            raise ValueError(f"未知模式: {mode}")
        return FieldScheduler(schedule_list, base_height=10)
    
    def path_generator(self):
        self.path.clear()
        i = 0
        while i < len(self.raw_path) - 1:
            name_start = self.raw_path[i]
            name_goal = self.raw_path[i + 1]
            start = self.acupoints_dict_scaled[name_start]["pos"]
            goal = self.acupoints_dict_scaled[name_goal]["pos"]

            # --- 判断距离模式 ---
            dist = np.linalg.norm(np.array(goal) - np.array(start))
            mode = 'line' if dist >= 0.0 else 'point'

            print(f"[{i}] {name_start} -> {name_goal} | mode={mode} | dist={dist:.2f}")

            # --- 生成场和agent ---
            field_scheduler = self._make_scheduler(mode, start, goal)
            agent = Agent(start)

            # --- 执行路径引导 ---
            for t in range(self.total_steps + 1):
                Z = field_scheduler.get_field(t)
                agent.step(Z)
                if self.vis:
                    self.vis.show(Z=Z, agent_pos=agent.pos, t=t, start=start, goal=goal)  # 动态展示
            # --- 强制回到目标点 ---
            agent.pos = np.array(goal)
            agent.path.append(tuple(goal))  # 添加回路径中
            if self.vis: # 可视化最后一个点
                self.vis.show(Z=Z, agent_pos=goal, t=t+1, start=start, goal=goal)
            # --- 记录路径 ---
            self.path.extend(agent.path[1:])  # 跳过起点

            # --- 判断是否跳跃 ---
            final_pos = agent.pos

            # 只取左半区域 (x < 4) 的点
            left_coords = [self.acupoints_dict_scaled[name]["pos"] for name in self.raw_path if self.acupoints_dict_scaled[name]["pos"][0] < self.mid_x * self.scale]
            # 若 left_coords 非空，则找 x 最大、在这些点中 y 最小的点
            if left_coords:
                # 找 x 的最大值
                max_x = max(p[0] for p in left_coords)
                # 在 x 最大的点中，找 y 最小的
                candidates = [p for p in left_coords if p[0] == max_x]
                target_jump_pos = min(candidates, key=lambda p: p[1])
                print(f"flags:{target_jump_pos}")
                print(f"final_pos:{final_pos}")
                if np.allclose(final_pos, target_jump_pos, atol=3.0):
                    print(f"跳跃触发 @ {name_goal} → {self.raw_path[i+2] if i+2 < len(self.raw_path) else '终点'}")
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
        name_pos_list = [(name, np.array(info['pos'])) for name, info in self.acupoints_dict_scaled.items()]

        # === 映射路径点为最近穴位名 === #
        mapped_names = []
        for pt in self.path:
            pt = np.array(pt)
            nearest_name = min(name_pos_list, key=lambda item: np.linalg.norm(item[1] - pt))[0]
            mapped_names.append(nearest_name)

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


if __name__ == "__main__":
    test_text = "1. 肩外俞穴:缓解肩颈僵硬2. 天宗穴:放松背部肌肉3. 肺俞穴:增强肺功能"
    mySorter = sorter()
    sorted_path = mySorter.sort_acupoints(test_text)
    print(sorted_path)
    myPlanner = planner(mySorter.acupoins_metadata,sorted_path,80,'line')
    myPlanner.path_generator()
    myPlanner.replanning()
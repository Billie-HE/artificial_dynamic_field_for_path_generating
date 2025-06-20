from APF_global_demo import GaussianSchedule,GaussianPathSchedule,TimedGaussianSchedule,FieldScheduler,Agent,Visualizer
import numpy as np
from sorter import sorter
class planner():
    def __init__(self,acupoints_dict:dict,sorted_path,total_steps=50,mode='point'):
        self.raw_path = sorted_path
        self.Gaussian_mode = ['line','point']
        self.acupoints_dict = acupoints_dict
        self.default_mode = mode
        self.path = []
        self.total_steps = total_steps
        self.vis = Visualizer((8,26),[])


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
                    height_fn=lambda t: -5 * (1 - (t - 0) / 20),
                    sigma=1.5,
                    t_start=0,
                    t_end=20,
                    hold_final=True
                ),
                TimedGaussianSchedule(
                    center=goal,
                    height_fn=lambda t: -5 * ((t - 30) / 20),
                    sigma=1.5,
                    t_start=30,
                    t_end=50,
                    hold_final=True
                ),
                GaussianPathSchedule(
                    start=start,
                    end=goal,
                    height_fn=lambda t: -5,
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
            start = self.acupoints_dict[name_start]["pos"]
            goal = self.acupoints_dict[name_goal]["pos"]

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

            # --- 记录路径 ---
            self.path.extend(agent.path[1:])  # 跳过起点

            # --- 判断是否跳跃 ---
            final_pos = agent.pos

            # 只取左半区域 (x < 4) 的点
            left_coords = [self.acupoints_dict[name]["pos"] for name in self.raw_path if self.acupoints_dict[name]["pos"][0] < 4]

            # 若 left_coords 非空，则找 x 最大、在这些点中 y 最小的点
            if left_coords:
                # 找 x 的最大值
                max_x = max(p[0] for p in left_coords)
                # 在 x 最大的点中，找 y 最小的
                candidates = [p for p in left_coords if p[0] == max_x]
                target_jump_pos = min(candidates, key=lambda p: p[1])

                if tuple(final_pos) == tuple(target_jump_pos):
                    print(f"跳跃触发 @ {name_goal} → {self.raw_path[i+2] if i+2 < len(self.raw_path) else '终点'}")
                    i += 2  # 跳过一个点
                else:
                    i += 1
            else:
                i += 1

        return self.path
if __name__ == "__main__":
    test_text = "1. 肩外俞穴:缓解肩颈僵硬2. 天宗穴:放松背部肌肉3. 肺俞穴:增强肺功能"
    mySorter = sorter()
    sorted_path = mySorter.sort_acupoints(test_text)
    print(sorted_path)
    myPlanner = planner(mySorter.acupoins_metadata,sorted_path,50,'line')
    res = myPlanner.path_generator()
    print(res)
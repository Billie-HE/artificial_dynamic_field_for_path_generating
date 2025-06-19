import numpy as np
import matplotlib.pyplot as plt

# ======================= 参数配置 ======================= #
GRID_SIZE = (8, 26)  # 行x列
TOTAL_STEPS = 50

# 完整穴位坐标（示例，注意纵坐标对应列）
acupoints = {
    "崇骨": (4, 25), "大椎": (4, 24), "肩中左俞": (3, 23), "肩中右俞": (5, 23),
    "肩外左俞": (2, 22), "肩外右俞": (6, 22), "秉风左": (1, 21), "曲垣左": (2, 21),
    "大杼左": (3, 21), "陶道": (4, 21), "大杼右": (5, 21), "曲垣右": (6, 21),
    "秉风右": (7, 21), "附分左": (2, 20), "风门左": (3, 20), "风门右": (5, 20),
    "附分右": (6, 20), "魄户左": (2, 19), "肺俞左": (3, 19), "身柱": (4, 19),
    "肺俞右": (5, 19), "魄户右": (6, 19), "天宗左": (1, 18), "膏肓左": (2, 18),
    "厥阴左俞": (3, 18), "厥阴右俞": (5, 18), "膏肓右": (6, 18), "天宗右": (7, 18),
    "神堂左": (2, 17), "心俞左": (3, 17), "神道": (4, 17), "心俞右": (5, 17),
    "神堂右": (6, 17), "譩譆左": (2, 16), "督俞左": (3, 16), "灵台": (4, 16),
    "督俞右": (5, 16), "譩譆右": (6, 16), "膈关左": (2, 15), "膈俞左": (3, 15),
    "至阳": (4, 15), "膈俞右": (5, 15), "膈关右": (6, 15), "魂门左": (2, 14),
    "肝俞左": (3, 14), "筋缩": (4, 14), "肝俞右": (5, 14), "魂门右": (6, 14),
    "阳纲左": (2, 13), "胆俞左": (3, 13), "中枢": (4, 13), "胆俞右": (5, 13),
    "阳纲右": (6, 13), "意舍左": (2, 12), "脾俞左": (3, 12), "脊中": (4, 12),
    "脾俞右": (5, 12), "意舍右": (6, 12), "胃仓左": (2, 11), "胃俞左": (3, 11),
    "胃俞右": (5, 11), "胃仓右": (6, 11), "肓门左": (2, 10), "三焦左俞": (3, 10),
    "悬枢": (4, 10), "三焦右俞": (5, 10), "肓门右": (6, 10), "京门左": (1, 9),
    "志室左": (2, 9), "肾俞左": (3, 9), "命门": (4, 9), "肾俞右": (5, 9),
    "志室右": (6, 9), "京门右": (7, 9), "气海左俞": (3, 8), "气海右俞": (5, 8),
    "大肠左俞": (3, 7), "腰阳关": (4, 7), "大肠右俞": (5, 7), "关元左俞": (3, 6),
    "关元右俞": (5, 6), "小肠左俞": (3, 5), "小肠右俞": (5, 5), "胞肓左": (2, 4),
    "膀胱左俞": (3, 4), "膀胱右俞": (5, 4), "胞肓右": (6, 4), "中膂左俞": (3, 3),
    "中膂右俞": (5, 3), "秩边左": (2, 2), "白环左俞": (3, 2), "白环右俞": (5, 2),
    "秩边右": (6, 2), "会阳左": (3, 1), "会阳右": (5, 1)
}
path_sequence = [
    "肩外左俞", "志室左", "肩中左俞", "三焦左俞", "大肠左俞", "会阳左"
]
# ======================= 高斯场调度器 ======================= #
class GaussianSchedule:
    """单个高斯场随时间演化的调度"""
    def __init__(self, center, height_fn, sigma):
        self.center = np.array(center)
        self.height_fn = height_fn
        # 如果传入的是函数，则用它；否则封装为恒定函数
        self.sigma_fn = sigma if callable(sigma) else (lambda t: sigma)

    def field_at(self, t):
        gx, gy = np.meshgrid(np.arange(GRID_SIZE[0]), np.arange(GRID_SIZE[1]), indexing='ij')
        d2 = (gx - self.center[0])**2 + (gy - self.center[1])**2
        height = self.height_fn(t)
        sigma = self.sigma_fn(t)
        return height * np.exp(-d2 / (2 * sigma**2))  # type: ignore


class GaussianPathSchedule:
    def __init__(self, start, end, height_fn, sigma):
        self.start = np.array(start)
        self.end = np.array(end)
        self.height_fn = height_fn
        self.sigma_fn = sigma if callable(sigma) else (lambda t: sigma)

    def field_at(self, t):
        gx, gy = np.meshgrid(np.arange(GRID_SIZE[0]), np.arange(GRID_SIZE[1]), indexing='ij')
        height = self.height_fn(t)
        sigma = self.sigma_fn(t)

        # 向量 AB 和点P的到线段距离
        AB = self.end - self.start
        AB_norm_sq = np.dot(AB, AB)
        P = np.stack([gx, gy], axis=-1)  # shape=(GRID_SIZE, GRID_SIZE, 2)

        # 向量 AP
        AP = P - self.start

        # 投影因子（0~1）
        proj = np.clip(np.sum(AP * AB, axis=-1) / AB_norm_sq, 0, 1)

        # 最近点 D 在线段 AB 上
        D = self.start + proj[..., None] * AB  # shape=(GRID_SIZE, GRID_SIZE, 2)

        # 点到线距离
        dist2 = np.sum((P - D)**2, axis=-1)

        return height * np.exp(-dist2 / (2 * sigma**2)) #type: ignore

class TimedGaussianSchedule:
    def __init__(self, center, height_fn, sigma, t_start=0, t_end=float('inf'), hold_final=False):
        self.center = np.array(center)
        self.height_fn = height_fn
        self.sigma_fn = sigma if callable(sigma) else (lambda t: sigma)
        self.t_start = t_start
        self.t_end = t_end
        self.hold_final = hold_final

    def field_at(self, t):
        if t < self.t_start:
            return np.zeros((GRID_SIZE[0], GRID_SIZE[1]))
        elif t > self.t_end:
            if self.hold_final:
                t = self.t_end  # 保持终点值
            else:
                return np.zeros((GRID_SIZE[0], GRID_SIZE[1]))

        gx, gy = np.meshgrid(np.arange(GRID_SIZE[0]), np.arange(GRID_SIZE[1]), indexing='ij')
        d2 = (gx - self.center[0])**2 + (gy - self.center[1])**2
        height = self.height_fn(t)
        sigma = self.sigma_fn(t)
        return height * np.exp(-d2 / (2 * sigma**2)) #type: ignore

class FieldScheduler:
    """管理多个高斯 schedule 并叠加形成总势场"""
    def __init__(self, schedules, base_height=10):
        self.schedules = schedules
        self.base_height = base_height

    def get_field(self, t):
        total_field = np.zeros((GRID_SIZE[0], GRID_SIZE[1]))
        for sched in self.schedules:
            total_field += sched.field_at(t)
        return total_field + self.base_height

# ======================= Agent 移动策略 ======================= #
class Agent:
    def __init__(self, start_pos):
        self.pos = np.array(start_pos)
        self.velocity = np.array([0, 0])
        self.path = [tuple(self.pos)]

    def step(self, Z):
        x, y = self.pos
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE[0] and 0 <= ny < GRID_SIZE[1]:
                    neighbors.append(((nx, ny), Z[nx, ny], np.array([dx, dy])))
        min_score = float('inf')
        for (nx, ny), z, dvec in neighbors:
            momentum_bonus = -np.dot(dvec, self.velocity)
            score = z + 0.5 * momentum_bonus  # 动量影响因子
            if score < min_score:
                min_score = score
                best = ((nx, ny), dvec)
        # 动量保持 + 最低势能选择
        min_score = float('inf')
        for (nx, ny), z, dvec in neighbors:
            momentum_bonus = -np.dot(dvec, self.velocity)  # 越同方向越优
            score = z + 0.8 * momentum_bonus
            if score < min_score:
                min_score = score
                best = ((nx, ny), dvec)

        next_pos, dvec = best
        self.pos = np.array(next_pos)
        self.velocity = dvec
        self.path.append(tuple(self.pos))

# ======================= 可视化模块 ======================= #
class Visualizer:
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
        plt.ion()

    def show(self, Z, agent_pos, t):
        plt.clf()
        plt.title(f"Time {t}")
        plt.imshow(Z.T, origin='lower', cmap='viridis')
        plt.colorbar(label='Height')
        plt.plot(agent_pos[0], agent_pos[1], 'ro', label='Agent')
        plt.plot(self.start[0], self.start[1], 'bs', label='Start')
        plt.plot(self.goal[0], self.goal[1], 'g*', label='Goal')
        plt.legend()
        plt.pause(0.1)

    def close(self):
        plt.ioff()
        plt.show()

def plan_segment(start, goal, agent_path_accum, vis):
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

    field_scheduler = FieldScheduler(schedule_list, base_height=10)
    agent = Agent(start)

    for t in range(TOTAL_STEPS + 1):
        Z = field_scheduler.get_field(t)
        agent.step(Z)
        vis.show(Z, agent.pos, t)

        if np.array_equal(agent.pos, goal):
            break

    vis.close()
    agent_path_accum.extend(agent.path[1:])  # 除去起点防重复


def main():
    full_path = []
    for i in range(len(path_sequence) - 1):
        name_start = path_sequence[i]
        name_goal = path_sequence[i + 1]
        start = acupoints[name_start]
        goal = acupoints[name_goal]

        print(f"规划路径: {name_start} {start} ➝ {name_goal} {goal}")

        vis = Visualizer(start, goal)
        plan_segment(start, goal, full_path, vis)

    print("完整按摩路径点:", full_path)


if __name__ == "__main__":
    main()
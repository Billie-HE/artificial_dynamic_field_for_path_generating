import numpy as np
import matplotlib.pyplot as plt

# ======================= 参数配置 ======================= #
GRID_SIZE = 8
START = (1, 1)
GOAL = (6, 6)
TOTAL_STEPS = 40

# ======================= 高斯场调度器 ======================= #
# 点状高斯源
class GaussianSchedule:          
    """单个高斯场随时间演化的调度"""
    def __init__(self, center, height_fn, sigma):
        self.center = np.array(center)
        self.height_fn = height_fn
        # 如果传入的是函数，则用它；否则封装为恒定函数
        self.sigma_fn = sigma if callable(sigma) else (lambda t: sigma)

    def field_at(self, t):
        gx, gy = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing='ij')
        d2 = (gx - self.center[0])**2 + (gy - self.center[1])**2
        height = self.height_fn(t)
        sigma = self.sigma_fn(t)
        return height * np.exp(-d2 / (2 * sigma**2))  # type: ignore

# 路径状高斯源（沿线段）
class GaussianPathSchedule:
    def __init__(self, start, end, height_fn, sigma):
        self.start = np.array(start)
        self.end = np.array(end)
        self.height_fn = height_fn
        self.sigma_fn = sigma if callable(sigma) else (lambda t: sigma)

    def field_at(self, t):
        gx, gy = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing='ij')
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

# 带时间窗口的高斯源
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
            return np.zeros((GRID_SIZE, GRID_SIZE))
        elif t > self.t_end:
            if self.hold_final:
                t = self.t_end  # 保持终点值
            else:
                return np.zeros((GRID_SIZE, GRID_SIZE))

        gx, gy = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE), indexing='ij')
        d2 = (gx - self.center[0])**2 + (gy - self.center[1])**2
        height = self.height_fn(t)
        sigma = self.sigma_fn(t)
        return height * np.exp(-d2 / (2 * sigma**2)) #type: ignore

# 势场叠加器
class FieldScheduler:
    """管理多个高斯 schedule 并叠加形成总势场"""
    def __init__(self, schedules, base_height=10):
        self.schedules = schedules
        self.base_height = base_height

    def get_field(self, t):
        total_field = np.zeros((GRID_SIZE, GRID_SIZE))
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
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
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

# ======================= 主程序入口 ======================= #
def main():
    # 定义高斯变化 schedule
    schedule_list = [
    TimedGaussianSchedule(
        center=START,
        height_fn=lambda t: -8 * (1 - (t - 0) / 20),
        sigma=1.5,
        t_start=0,
        t_end=20,
        hold_final=True
    ),
    TimedGaussianSchedule(
        center=GOAL,
        height_fn=lambda t: -8 * ((t - 20) / 20),
        sigma=1.5,
        t_start=20,
        t_end=40,
        hold_final=True
    ),
    GaussianPathSchedule(
        start=START,
        end=GOAL,
        height_fn=lambda t: -3,
        sigma=1.5
    )
    ]

    

    field_scheduler = FieldScheduler(schedule_list, base_height=0)
    agent = Agent(START)
    vis = Visualizer(START, GOAL)

    for t in range(TOTAL_STEPS + 1):
        Z = field_scheduler.get_field(t)
        agent.step(Z)
        vis.show(Z, agent.pos, t)

        # if np.array_equal(agent.pos, GOAL):
        #     print(f"Agent reached goal at time {t}")
        #     break

    vis.close()
    print("Agent path:", agent.path)

if __name__ == "__main__":
    main()

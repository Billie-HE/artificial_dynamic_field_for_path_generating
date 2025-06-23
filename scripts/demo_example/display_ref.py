import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import json
from matplotlib.patches import Circle, Ellipse
import matplotlib.transforms as transforms
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import time
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 路径类型映射
PATH_MAPPING = {
    'cycloid': '摆线',
    'point': '定点',
    'point_rub': '定点',
    'line': '直线',
    'in_spiral': '螺旋线',
    'out_spiral': '螺旋线'
}

# 路径颜色映射
PATH_COLORS = {
    'cycloid': 'blue',
    'point': 'red',
    'point_rub': 'red',
    'line': 'green',
    'in_spiral': 'purple',
    'out_spiral': 'purple'
}

# 定义网格和穴位坐标
grid_size = (7, 25)
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

def read_json_file(file_path):
    """读取并解析JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return None
    except json.JSONDecodeError:
        print("错误：文件不是有效的JSON格式")
        return None
def display_all_therapies(therapy_data):
    """
    显示所有疗法信息
    :param therapy_data: 包含所有疗法数据的字典
    """
    if not therapy_data:
        return
    
    for therapy_name, therapy_info in therapy_data.items():
        print("=" * 60)
        print(f"疗法名称: {therapy_name}")
        print(f"介绍: {therapy_info['introduction']}")
        print(f"身体部位: {therapy_info['body_part']}")
        print(f"选择的任务: {therapy_info['choose_task']}")
        print("\n任务计划:")
        
        for i, task in enumerate(therapy_info['task_plan'], 1):
            print(f"{i}. 从 {task['start_point']} 到 {task['end_point']}，路径类型: {task['path']}")
        
        print(f"\n总步骤数: {len(therapy_info['task_plan'])}")
        print("=" * 60 + "\n")
def parse_point(point_str, default_x=4, default_y=1):
    """
    解析穴位点字符串，处理组合点和比例点
    规则:
    - "A" -> A穴位的坐标
    - "A&B" -> A和B的中点
    - "A&B@ratio" -> AB连线上距离A比例为ratio的点 (ratio可以是小数或分数如0.3或1/3)
    """
    def parse_ratio(ratio_str):
        """解析比例字符串，支持小数和分数形式"""
        try:
            if '/' in ratio_str:
                numerator, denominator = ratio_str.split('/')
                return float(numerator) / float(denominator)
            return float(ratio_str)
        except:
            return 0.5  # 默认返回中点

    try:
        if '&' in point_str:
            # 处理组合点
            if '@' in point_str:
                # 带比例的组合点 (A&B@ratio)
                points_part, ratio_str = point_str.split('@')
                ratio = parse_ratio(ratio_str)
                points = points_part.split('&')
                if len(points) != 2:
                    raise ValueError(f"组合点格式错误: {point_str}")
                
                # 获取两个穴位的坐标
                point_a = parse_point(points[0])
                point_b = parse_point(points[1])
                
                # 计算比例点
                x = point_a[0] + ratio * (point_b[0] - point_a[0])
                y = point_a[1] + ratio * (point_b[1] - point_a[1])
                return (x, y)
            else:
                # 简单中点 (A&B)
                points = point_str.split('&')
                coords = [parse_point(p) for p in points]
                x = sum(c[0] for c in coords) / len(coords)
                y = sum(c[1] for c in coords) / len(coords)
                return (x, y)
        elif point_str in acupoints:
            # 单个穴位点
            return acupoints[point_str]
        else:
            # 未知穴位，返回默认位置
            return (default_x, default_y)
    except Exception as e:
        print(f"解析穴位点错误: {point_str}, 错误: {e}")
        return (default_x, default_y)

def draw_cycloid(ax, start, end, color='blue'):
    """绘制摆线轨迹"""
    x1, y1 = start
    x2, y2 = end
    
    # 计算控制点
    ctrl_x = (x1 + x2) / 2
    ctrl_y = (y1 + y2) / 2 + abs(x2 - x1) * 0.5
    
    # 创建贝塞尔曲线
    t = np.linspace(0, 1, 100)
    x = (1-t)**2 * x1 + 2 * (1-t) * t * ctrl_x + t**2 * x2
    y = (1-t)**2 * y1 + 2 * (1-t) * t * ctrl_y + t**2 * y2
    
    ax.plot(x, y, color=color, linestyle='-', linewidth=2)

def draw_spiral(ax, center, size=0.5, color='purple', clockwise=True):
    """绘制螺旋线轨迹"""
    t = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0, size, 100)
    
    if not clockwise:
        t = -t
    
    x = center[0] + r * np.cos(t)
    y = center[1] + r * np.sin(t)
    
    ax.plot(x, y, color=color, linestyle='-', linewidth=2)

def draw_point(ax, point, color='red'):
    """绘制定点轨迹"""
    circle = Circle(point, radius=0.2, color=color, alpha=0.7)
    ax.add_patch(circle)

def draw_line(ax, start, end, color='green'):
    """绘制直线轨迹"""
    ax.plot([start[0], end[0]], [start[1], end[1]], 
            color=color, linestyle='-', linewidth=2)

def create_legend(ax):
    """创建图例"""
    legend_elements = []
    for path_type, color in PATH_COLORS.items():
        if path_type in PATH_MAPPING:
            legend_elements.append(
                plt.Line2D([0], [0], color=color, lw=2, 
                          label=f'{PATH_MAPPING[path_type]} ({path_type})'))
    
    legend_elements.extend([
        plt.Line2D([0], [0], marker='o', color='k', lw=0, label='起点'),
        plt.Line2D([0], [0], marker='x', color='k', lw=0, label='终点')
    ])
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))

def visualize_task_steps_sequentially(json_path, therapy_name, step_interval=0.5):
    """
    按顺序动态绘制疗法任务的每个步骤
    :param json_path: JSON文件路径
    :param therapy_name: 疗法名称
    :param step_interval: 每个步骤的显示时间（秒）
    """
    data = read_json_file(json_path)
    if not data or therapy_name not in data:
        print(f"疗法 '{therapy_name}' 不存在")
        return
    
    therapy_info = data[therapy_name]
    task_plan = therapy_info['task_plan']
    
    # 初始化图形
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xticks(np.arange(0, grid_size[0] + 1, 1))
    ax.set_yticks(np.arange(0, grid_size[1] + 1, 1))
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 绘制所有穴位（灰色半透明）
    for name, (x, y) in acupoints.items():
        ax.plot(x, y, 'o', color='gray', markersize=8, alpha=0.2)
        ax.text(x, y, name, ha='center', va='center', fontsize=6, alpha=0.3)
    
    # 动态绘制每个步骤
    for step_idx, task in enumerate(task_plan, 1):
        start = parse_point(task['start_point'])
        end = parse_point(task['end_point'])
        path_type = task['path']
        
        # 清除上一步的临时图形（保留穴位和已完成的步骤）
        for artist in ax.lines + ax.patches + ax.texts:
            if hasattr(artist, '_is_temp') and artist._is_temp:
                artist.remove()
        
        # 绘制当前步骤
        if path_type in ['cycloid']:
            draw_cycloid(ax, start, end, PATH_COLORS[path_type])
        elif path_type in ['point', 'point_rub']:
            draw_point(ax, start, PATH_COLORS[path_type])
        elif path_type == 'line':
            draw_line(ax, start, end, PATH_COLORS[path_type])
        elif path_type in ['in_spiral', 'out_spiral']:
            draw_spiral(ax, start, color=PATH_COLORS[path_type], clockwise=(path_type == 'in_spiral'))
        
        # 标记当前步骤的起点和终点（临时标记）
        start_marker, = ax.plot(start[0], start[1], 'ko', markersize=10, alpha=0.7)
        end_marker, = ax.plot(end[0], end[1], 'kx', markersize=10, alpha=0.7)
        step_text = ax.text((start[0]+end[0])/2, (start[1]+end[1])/2, 
                          str(step_idx), fontsize=12, ha='center', va='center',
                          bbox=dict(facecolor='white', alpha=0.8))
        
        # 标记为临时对象（下次循环会被清除）
        for artist in [start_marker, end_marker, step_text]:
            artist._is_temp = True
        
        # 更新标题
        ax.set_title(
            f"{therapy_name}\n"
            f"步骤 {step_idx}/{len(task_plan)}: {task['start_point']} → {task['end_point']}\n"
            f"类型: {PATH_MAPPING.get(path_type, path_type)}",
            fontsize=12
        )
        
        plt.draw()
        plt.pause(step_interval)  # 暂停一段时间
    plt.show()

# 使用示例
if __name__ == "__main__":
    json_path = "C:/Users/ZIWEI/Documents/work/向量化/cur_plans_hzw.json"
    data = read_json_file(json_path)
    display_all_therapies(data)
    visualize_task_steps_sequentially(json_path,"点振波理疗头-肩颈-默认")
from sorter import sorter
import numpy as np
from planner import planner

if __name__ == "__main__":
    test_text = "1. 肩外俞穴:缓解肩颈僵硬2. 天宗穴:放松背部肌肉3. 肺俞穴:增强肺功能"
    mySorter = sorter()

    TOTAL_STEPS = 80

    acupoints = mySorter.acupoins_metadata

    path_sequence = mySorter.sort_acupoints(test_text)

    myPlanner = planner(mySorter.acupoins_metadata,path_sequence,TOTAL_STEPS,'line')
    
    full_path = myPlanner.path_generator()

    for i in range(len(path_sequence) - 1):
        name_start = path_sequence[i]
        name_goal = path_sequence[i + 1]
        start = acupoints[name_start]["pos"]
        goal = acupoints[name_goal]["pos"]

        print(f"规划路径: {name_start} {start} ➝ {name_goal} {goal}")
    
    print("完整按摩路径点:", full_path)
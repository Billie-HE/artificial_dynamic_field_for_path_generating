import re
import json
import numpy as np

class sorter:
    ''' 重点穴位按摩排列器 '''
    def __init__(self, body_part = 'back'):
        with open('config/acupoint_metadata.json','r',encoding='utf-8') as file:
            metadata = json.load(file)
        self.acupoints_metadata = metadata
        self.body_part_list = ['back','shoulder','waist']
        if body_part in self.body_part_list:       
            self.body_part = body_part # 默认部位为'back'
        else:
            raise ValueError("按摩位置不在可按摩区域")
    
    def _extract_acupoints(self,respnse_from_llm:str)->list[str]:
        pattern = r"[0-9]+\.\s*([\u4e00-\u9fa5]{2,5}穴)"
        matches = re.findall(pattern, respnse_from_llm)
        # 去重 & 排除空值
        unique_names = sorted(set(name for name in matches if name.strip()))
        print(unique_names)
        return unique_names
    
    def sort_acupoints(self,respnse_from_llm:list[str])->list[str]:
        if self.body_part == 'back':
            allowed_names = []
        if self.body_part == 'shoulder':
            allowed_names = ["肩中左俞","肩外左俞","秉风左","天宗左","曲垣左","附分左",
                             "肩中右俞","肩外右俞","秉风右","天宗右","曲垣右","附分右"]
        if self.body_part == 'waist':
            allowed_names = ["志室左","肓门左","胃仓左","意舍左","阳纲左","胞肓左","气海左俞",
                             "大肠左俞","小肠左俞","中膂左俞","肾俞左","关元左俞","膀胱左俞","白环左俞","秩边左","京门左",
                             "志室右","肓门右","胃仓右","意舍右","阳纲右","胞肓右","气海右俞","大肠右俞","小肠右俞",
                             "中膂右俞","肾俞右","关元右俞","膀胱右俞","白环右俞","秩边右","京门右"]

        def __filter_acupoints(acupoints:list[str],allowed_names: list[str])->list[str]:
             acupoints_cleaned = []
             matched_keys = []
             for acupoint in acupoints:
                  res = re.sub(r"[穴]$","",acupoint)
                  res = re.sub(r"[俞]$","",res)
                  acupoints_cleaned.append(res)
             for name in acupoints_cleaned:
                # 第一步：模糊匹配 metadata 中包含 name 的 key
                keys_candidates = [k for k in self.acupoints_metadata.keys() if name in k]

                # 第二步：如果不是背部，限制匹配在 allowed_names 范围内
                if self.body_part != "back":
                    keys_candidates = [k for k in keys_candidates if k in allowed_names]

                matched_keys.extend(keys_candidates)
             return matched_keys
        acupoints_filtered = __filter_acupoints(respnse_from_llm,allowed_names)
        def __get_coords(name:str):
             return self.acupoints_metadata[name]["pos"]

        left_group = [pt for pt in acupoints_filtered if __get_coords(pt)[0] < 4]
        right_group = [pt for pt in acupoints_filtered if __get_coords(pt)[0] > 4]
        
        left_group_sorted = sorted(left_group,key=lambda pt:(__get_coords(pt)[0],-__get_coords(pt)[1]))
        right_group_sorted = sorted(right_group,key=lambda pt:(__get_coords(pt)[0],__get_coords(pt)[1]),reverse=True)

        pt_sorted = left_group_sorted + right_group_sorted

        return pt_sorted


if __name__ == "__main__":
    mySorter = sorter()
    print(type(mySorter.acupoints_metadata))
    test_response = ["肩井穴","天宗穴","肺俞穴"]
    pts_sorted = mySorter.sort_acupoints(test_response)
    print(pts_sorted)

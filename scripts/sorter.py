import re
import json

class sorter:
    ''' 重点穴位按摩排列器 '''
    def __init__(self):
        self.counterparts = {"thermotherapy":["belly","back","waist"],
                             "stone":["belly","back","waist"],
                             "shockwave":["back","back_shoulder","shoulder","leg"],
                             "finger":["back","back_shoulder","shoulder","leg"]}
        self.Gaussian_mode = ['line','point']
        self.acupoints = None
        self.body_part = None
        self.GRID_SIZE = (8,26)
        with open('config/acupoint_metadata.json','r',encoding='utf-8') as file:
            metadata = json.load(file)
        self.acupoins_metadata = metadata
    
    def _extract_acupoints(self,respnse_from_llm:str)->list[str]:
        pattern = r"[0-9]+\.\s*([\u4e00-\u9fa5]{2,5}穴)"
        matches = re.findall(pattern, respnse_from_llm)
        # 去重 & 排除空值
        unique_names = sorted(set(name for name in matches if name.strip()))
        return unique_names
    
    def sort_acupoints(self,respnse_from_llm:str)->list[str]:
        def __filter_acupoints(acupoints:list[str]):
             acupoints_cleaned = []
             matched_keys = []
             for acupoint in acupoints:
                  acupoints_cleaned.append(re.sub(r"[穴]$","",acupoint))
             for name in acupoints_cleaned:
                keys_filtered = [k for k in self.acupoins_metadata.keys() if name in k]
                matched_keys.extend(keys_filtered)
             return matched_keys
        
        acupoints_filtered = __filter_acupoints(self._extract_acupoints(respnse_from_llm))

        def __get_coords(name:str):
             return self.acupoins_metadata[name]["pos"]
        print(acupoints_filtered)
        left_group = [pt for pt in acupoints_filtered if __get_coords(pt)[0] < 4]
        right_group = [pt for pt in acupoints_filtered if __get_coords(pt)[0] > 4]
        
        left_group_sorted = sorted(left_group,key=lambda pt:(__get_coords(pt)[0],-__get_coords(pt)[1]))
        right_group_sorted = sorted(right_group,key=lambda pt:(__get_coords(pt)[0],__get_coords(pt)[1]),reverse=True)

        pt_sorted = left_group_sorted + right_group_sorted

        return pt_sorted


if __name__ == "__main__":
    mySorter = sorter()
    test_response = "1. 肩井穴:缓解肩颈僵硬2. 天宗穴:放松背部肌肉3. 肺俞穴:增强肺功能"
    pts_sorted = mySorter.sort_acupoints(test_response)
    print(pts_sorted)

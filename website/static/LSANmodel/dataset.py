import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, visit_info,  label, value):
        self.x_lines = visit_info
        self.y_lines = label
        self.z_lines = value

    def __len__(self):
        return len(self.x_lines)

    def __getitem__(self, index):
        visit_diag_code = self.x_lines[index]
        visit_label = self.y_lines[index]
        visit_value = self.z_lines[index]
        return visit_diag_code, visit_label, visit_value

    @staticmethod
    def collate_fn(batch):
        maxVisitTime = 13
        maxCodeLengh = 50
        padding_idx = 51
        padding_value = 0


        x_result = []
        padding_result = []  
        label_result = [] 
        value_result = []
        y_result = []    

        for b, l, v in batch:
            x_result.append(b)
            y_result.append(l)

            j_list = []
            for j in b:
                code_padding_amount = maxCodeLengh - len(j)
                j_list.append(j + [padding_idx] * code_padding_amount)
            padding_result.append(j_list)
            
            j_list = []
            for j in v:
                code_padding_amount = maxCodeLengh - len(j)
                j_list.append(j + [padding_idx] * code_padding_amount)
            value_result.append(j_list)
            
            visit_padding_amount = maxVisitTime - len(b)
            for time in range(visit_padding_amount):
                padding_result[-1].append([padding_idx] * maxCodeLengh)
                value_result[-1].append([padding_value] * maxCodeLengh)
        
        return (torch.tensor(padding_result), torch.tensor(y_result), torch.tensor(value_result), x_result, y_result)

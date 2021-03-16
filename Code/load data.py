# author: SaKuRa Pop
# data: 2021/3/13 10:48
import xlrd
import numpy as np


def xlsx_to_pkl(open_file_path, save_file_path_1, save_file_path_2):
    data = xlrd.open_workbook(open_file_path)
    table = data.sheet_by_name('Sheet1')
    table_array = np.zeros((10000, 4097))
    for i in range(2000):
        for j in range(4097):
            table_array[i, j] = table.cell_value(i, j)

    data_x = table_array
    data_y = np.array([x for x in range(10000)])
    data_y = data_y[:, np.newaxis]
    np.save(save_file_path_1, data_x)
    np.save(save_file_path_2, data_y)
    return data_x, data_y

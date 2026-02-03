import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
from icecream import ic

@dataclass
class VisualColor:
    color : list

def _init_data(config):
    '''
    _init_data 的 Docstring
    
    :param center_num: 说明
    :param dispersion: 说明
    :param data_limit: 说明
    :param data_num: 说明

    :return: [[x1,y1],[x2,y2],...]
    '''
    # center_num, 2, data_num
    # 预计分为几块数据
    center_num_list = []
    for j in range(config["center_num"]):
        point_x_list = []
        point_y_list = []
        # 随机生成中心点的坐标
        center_x, center_y = np.random.uniform(0, config["data_limit"], 2)
        range_x = [center_x - config["dispersion"], center_x + config["dispersion"]]
        range_y = [center_y - config["dispersion"], center_y + config["dispersion"]]
        for i in range(config["data_num"]):
            np_xy = []
            # 随机生成一个中心点[x,y]
            point_x = np.random.uniform(range_x[0], range_x[1])
            point_y = np.random.uniform(range_y[0], range_y[1])
            # 把这簇的数据保存
            point_x_list.append(point_x)
            point_y_list.append(point_y)
        
        center_num_list.append([point_x_list, point_y_list])

    return np.array(center_num_list)


def _visual_data(data):
    # random_color = np.random.rand(3)
    for i in range(data.shape[0]):
        random_color = np.random.rand(3)
        list_x = data[i][0]
        list_y = data[i][1]
        plt.scatter(list_x, list_y, c=random_color, alpha=0.5)
    plt.show()

def _visual_k_means(list_x, list_y, k_means_num, k_centers_x, k_centers_y):
    if len(VisualColor.color) == 0:
        for i in range(k_means_num):
            random_color = np.random.rand(4)
            # 最后增加一个透明值
            random_color[-1] = 0.5
            VisualColor.color.append(random_color)
    ic(VisualColor.color)
    # visual center as a star
    plt.scatter(k_centers_x, k_centers_y, marker='*', s=100)
    for i in range(k_means_num):
        plt.scatter(k_centers_x[i], k_centers_y[i], c=VisualColor.color[i], alpha=0.5)
    plt.show()
    # visual data



def run_k_means(data_xy, config):

    k_centers_x = []
    k_centers_y = []
    for i in range(config["k_means_num"]):
        # 初始化聚类中心
        center_x, center_y = np.random.uniform(0, config["data_limit"], 2)
        k_centers_x.append(center_x)
        k_centers_y.append(center_y)
        
    # k_means_xy.append([k_centers_x, k_centers_y])
    np_k_means = np.array([k_centers_x, k_centers_y]) # 2, k_means_num

    for iter in range(config["iter_num"]):
        # 计算距离
        # data_xy --> (3, 2, 100) (3组, xy, 100个点)
        # np_k_means --> (2, k_means_num)
        # all_data_xy --> (2, 300) (2个维度, 300个点)
        # 第一步：转置，将 (3, 2, 100) → (2, 3, 100)，把「xy维度」提到最前面
        # 第二步：reshape，将 (2, 3, 100) → (2, 300)
        all_data_xy = data_xy.transpose(1, 0, 2).reshape(2, config["data_num"] * config["center_num"])
        # ic(all_data_xy.shape)
        # 每个点和聚类中心的距离
        all_data_xy_N2 = all_data_xy.T  # 转置即可，(2, 300) → (300, 2)
        centers_K2 = np_k_means.T  # (2, 3) → (3, 2)

        # 利用numpy广播计算欧几里得距离
        # 广播规则：(300, 2) - (3, 2) → 先扩展为 (300, 3, 2)，再逐元素相减
        diff = all_data_xy_N2[:, np.newaxis, :] - centers_K2[np.newaxis, :, :]
        # 计算平方和 → (300, 3)
        square_sum = np.sum(diff ** 2, axis=2)
        # 开平方得到最终距离 → (300, 3) 距离矩阵
        distance_matrix = np.sqrt(square_sum)

        




if __name__ == "__main__":
    config = {
        "center_num": 3, # 生成数据的聚类中心数
        "dispersion": 2, # 聚类中心的分散程度
        "data_limit": 10, # 数据的边界
        "data_num": 100, # 每簇生成的数据量
        "k_means_num": 3, # 聚类中心的个数
        "iter_num": 3 # 迭代次数
    }
    vc = VisualColor(color=[])
    # (3, 2, 100)
    data_xy = _init_data(config)
    _visual_data(data_xy)

    run_k_means(data_xy, config)
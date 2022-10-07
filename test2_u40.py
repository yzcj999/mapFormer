from sympy import *
import math
import tensorflow as tf
import numpy as np
import sklearn

print(sklearn.__version__)
from scipy.stats import norm
import os
from sklearn.model_selection import train_test_split
import setting
from sklearn.impute import SimpleImputer

from tqdm import *
from multiprocessing import Pool  # 并行运算加快速度
import time
from multiprocessing.dummy import Pool as ThreadPool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# pool = ThreadPool(32)
fig = 0


class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def read_node(name):
    """

    :param name: 文件名字
    :return: 返回节点列表及其标记点的列表
    """
    dic = []
    with open(name, "r") as fp:
        list_node = fp.read().split("\n")

        for i, j in zip(list_node, range(len(list_node))):
            k = i.split("\t")
            if k == [""]:
                continue
            r1 = float(k[0])
            r2 = float(k[1])
            dic.append([r1, r2])
        del fp
    return dic


def matching_arc(path, node_list):
    """

    :param path: 路径
    :param node_list: node节点
    :return: 返回各个路径上的节点
    """
    list_arc = []
    with open(path, "r") as fp:
        list_data = fp.read().split("\n")
        del fp
        list1 = list_data[0].split("\t")
        label0 = list1[0]
        point0 = list1[1]
        list_temp = []
        for i in list_data:

            if i == "":
                continue
            list2 = i.split("\t")
            label = list2[0]
            point = list2[1]
            if label0 != label:
                list_arc.append(list_temp)
                list_temp.clear()
                label0 = label
            list_temp.append(node_list[int(point)])

    return list_arc


def read_route(path):
    """

    :param path: route文件路径
    :return: route列表
    """
    list1 = []
    with open(path, "r") as fp:
        for i in fp.read().split("\n"):
            if i != "":
                list1.append(i)
        del fp
        return list1


def read_GPS_point_list(path):
    with open(path, "r") as fp:
        list_track = []

        # u = 0  # 均值μ
        # u01 = -2
        # sig = math.sqrt(0.2)
        def normal(x, u, sig):
            print(x, u, sig)
            return math.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)

        for i in fp.read().split("\n"):
            if i == "":
                continue
            list1 = i.split("\t")

            x = float(list1[0]) + normal(float(list1[0]), 0, 40)
            y = float(list1[1]) + normal(float(list1[1]), 0, 40)
            print(x,y)
            list_track.append([x, y])
        del fp
        return list_track


class CANDIDATE:
    def __init__(self, path):
        self.nodes = "data/" + path + "/" + path + ".nodes"
        self.arcs = "data/" + path + "/" + path + ".arcs"
        self.track = "data/" + path + "/" + path + ".track"

        self.grim = 20
        self.mean = 4.07
        self.H_END = []
        self.p_bar = None
        self.nodes1 = None
        self.list1 = None

    def get_point(self, point1, point2, l, r1):
        """
        :param point1: 点1
        :param point2: 点3
        :param l: 1-2的长度
        :param r: 判断是否是锐角或钝角
        :return: 点四的坐标
        """
        x, y = symbols('x,y')
        r = math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

        a, b, c, d = r1
        x2, y2 = solve([(point1.x - point1.y) * x + (point2.x - point2.y) * y + 5 * r / 4 - l, x ** 2 + y ** 2 - 1],
                       [x, y])

        for x1, y1 in zip(x2, y2):
            if c == 1:
                if x1 < 0:
                    x = x1
            elif c == -1:
                if x1 > 0:
                    x = x1
            elif c == 0:
                x = x1
            if d == 0:
                if y1 < 0:
                    y = y1
            elif d == 1:
                if y1 > 0:
                    y = y1

        return x * r + (point1.x + point2.x) / 2.0, y * r + (point1.y + point2.y) / 2.0

    def get_angle(self, point1, point2, point3):
        """
        :param point1: 点1
        :param point2: 点2
        :param point3: 点3
        :return: 返回钝角，直角，锐角 分别用1，0，-1表示 顺序代表角1，2，3
        后面第四个参数返回的是是为正三角还是倒三角 分别用1，0表示
        """
        len1_2 = (point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2
        len1_3 = (point1.x - point3.x) ** 2 + (point1.y - point3.y) ** 2
        len2_3 = (point2.x - point3.x) ** 2 + (point2.y - point3.y) ** 2
        sort_list = [len2_3, len1_2, len1_3]
        sort_list.sort()
        max = sort_list[2]
        min = sort_list[0]
        mid = sort_list[1]
        if point1.y < point2.y:
            fig = 0
        else:
            fig = 1
        """
        钝角
        """
        if mid + min > max:
            if len2_3 > max(len1_2, len1_3):
                return 1, -1, -1, fig
            elif len1_2 > max(len2_3, len1_3):
                return -1, -1, 1, fig
            elif len1_3 > max(len2_3, len1_2):
                return -1, 1, -1, fig

        """
        锐角
        """
        if mid + min < max:
            return -1, -1, -1, fig
        if mid + min == max:
            if len2_3 > max(len1_2, len1_3):
                return 0, -1, -1, fig
            elif len1_2 > max(len2_3, len1_3):
                return -1, -1, 0, fig
            elif len1_3 > max(len2_3, len1_2):
                return -1, 0, -1, fig

    def POSITION_CONTEXT_ANALYSIS(self, cand_list):
        """
        计算标准差
        :param cand_list: 候选人列表
        :return: 概率
        """
        list_position = []
        for i in cand_list:
            cand = norm.ppf(i, loc=self.mean, scale=self.grim)
            list_position.append(cand)
        return list_position

    def get_h(self, point1, point2, point3):
        return (
                           point1.x * point2.y - point1.x * point3.y + point2.x * point3.y - point2.x * point1.y + point3.x * point1.y - point2.x * point2.y) / (
                           2.0 * math.sqrt(
                       (point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2))

    def fuction(self, dic):
        i = dic
        h_list_possibly = []
        for line, j in zip(self.list1, range(len(self.list1))):
            if line == "":
                continue
            route_list = line.split("\t")
            point1 = point(float(i[0]), float(i[1]))
            point2 = point(self.nodes1[int(route_list[0])][0], self.nodes1[int(route_list[0])][1])
            point3 = point(self.nodes1[int(route_list[1])][0], self.nodes1[int(route_list[1])][1])
            # self.get_point(point1, point2, self.get_h(point1, point2, point3), self.get_angle(point1, point2, point3)))
            # r1=self.get_angle(point1,point2, point3)

            h = self.get_h(point1, point2, point3)
            if h < 0:
                continue
            if h < 0.1719:
                cand = norm.ppf(h, loc=self.mean, scale=self.grim)
                h_list_possibly.append([j, cand])
            if len(h_list_possibly) > 100:
                break
            # x_4,y_4=self.get_point(point1,point2,h,r1)
            # point4=point(x_4,y_4)

        """
                   排序选出前100个
        """

        h_list_possibly = sorted(h_list_possibly, key=lambda x: x[1] if x[1] != nan else None,
                                 reverse=True)[:100]
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        try:
            imp_mean.fit(h_list_possibly)
            list_outof_nan = imp_mean.transform(h_list_possibly)
            # print(h_list_possibly)
            self.H_END.append(list_outof_nan)
            h_list_possibly = []
            self.p_bar.update(1)

        except:
            fig = 1

    def CANDIDATA_GET(self):
        gps_list = read_GPS_point_list(self.track)
        nodes = read_node(self.nodes)
        self.nodes1 = nodes

        with open(self.arcs, "r") as fp:
            list1 = fp.read().split("\n")
            self.list1 = list1
        with tqdm(total=len(gps_list)) as p_bar:
            self.p_bar = p_bar
            # with ThreadPoolExecutor(10000) as t:
            #     # 需要调用50次函数，也可以换参数传递
            #     for i in gps_list:
            #         t.submit(self.fuction, dic=i)
            # with ProcessPoolExecutor(2) as t:
            #     # 需要调用50次函数，也可以换参数传递
            #     for i in gps_list:
            #         t.submit(self.fuction, dic=i)
            # pool.apply_async(self.fuction, args=(gps_list,))
            #
            # pool.close()
            # pool.join()
            for i in gps_list:
                if fig == 1:
                    break
                self.fuction(i)

            if len(self.H_END) <= 3000:

                pad_list = np.zeros([100, 2])
                for i in range(3000 - len(self.H_END)):
                    self.H_END.append(pad_list)
                return self.H_END
            else:
                return self.H_END[:3000]


def dataset_make(list_val, list_label):
    list_val = tf.convert_to_tensor(list_val)
    list_label = tf.convert_to_tensor(list_label)
    dataset = tf.data.Dataset.from_tensor_slices((list_val, list_label))
    dataset = dataset.shuffle(setting.BUFFER_SIZE).prefetch(
        tf.data.experimental.AUTOTUNE).batch(setting.batch)
    print(dataset)
    return dataset


def normailze(list_track):
    """

    :param list_track: 需要归一化的route
    :return: 列表
    """
    gps_x = np.array([i[0] for i in list_track], dtype=float)
    gps_y = np.array([i[1] for i in list_track], dtype=float)

    gps_x_max = float(max(gps_x))
    gps_y_max = float(max(gps_y))
    gps_x_min = float(min(gps_x))
    gps_y_min = float(min(gps_y))

    gps_x = (gps_x - gps_x_min) / (gps_x_max - gps_x_min)
    gps_y = (gps_y - gps_y_min) / (gps_y_max - gps_y_min)

    return [[i, j] for i, j in zip(gps_x.tolist(), gps_y.tolist())]


def main_dataset():
    """
    #这个是旧的数据处理方式已经舍去
    :return: 处理好的数据集
    """
    list_vals = []
    list_labels = []
    for root, dirs, files in os.walk("data", topdown=False):

        # data/00000000
        for path in dirs:
            list_label = read_route("data\\" + path + "\\" + path + ".route")
            list_val = normailze(read_GPS_point_list("data\\" + path + "\\" + path + ".track"))
            """
            padding操作
            """
            if len(list_val) > 3000 or len(list_label) > 500:
                continue
            if len(list_val) < 3000:
                for _ in range(3000 - len(list_val)):
                    list_val.append([0.0, 0.0])
            if len(list_label) < 500:
                for _ in range(500 - len(list_label)):
                    list_label.append(0)
            list_vals.append(list_val)
            """
            这一步用于构建词表                                                           
            """
            try:
                list_label = [int(i) for i in list_label]
                list_labels.append(tf.one_hot(list_label, depth=10000, dtype=tf.float32, axis=0))
            except:
                pass

    """
    构造数据集
    """
    print(tf.convert_to_tensor(list_labels).shape)
    x_train, x_test, y_train, y_test = train_test_split(
        list_vals, list_labels,  # x,y是原始数据
        test_size=0.2  # test_size默认是0.25
    )  # 返回的是 剩余训练集+测试集

    dataset_train = dataset_make(x_train, y_train)
    dataset_test = dataset_make(x_test, y_test)
    return dataset_train, dataset_test


# def main2_dataset():
#     list_vals = []
#     list_labels = []
#     for root, dirs, files in os.walk("data", topdown=False):
#
#         # data/00000000
#         for path in dirs:
#             list_label = read_route("data\\" + path + "\\" + path + ".route")
#             list_val = CANDIDATE()
#             """
#             padding操作
#             """
#             if len(list_val) > 3000 or len(list_label) > 500:
#                 continue
#             if len(list_val) < 3000:
#                 for _ in range(3000 - len(list_val)):
#                     list_val.append([0.0, 0.0])
#             if len(list_label) < 500:
#                 for _ in range(500 - len(list_label)):
#                     list_label.append(0)
#             list_vals.append(list_val)
#             """
#             这一步用于构建词表
#             """
#             try:
#                 list_labels.append(tf.one_hot(list_label, depth=10000, dtype=tf.float32, axis=0))
#             except:
#                 pass
#
#     """
#     构造数据集
#     """
#     print(tf.convert_to_tensor(list_labels).shape)
#     x_train, x_test, y_train, y_test = train_test_split(
#         list_vals, list_labels,  # x,y是原始数据
#         test_size=0.2  # test_size默认是0.25
#     )  # 返回的是 剩余训练集+测试集
#
#     dataset_train = dataset_make(x_train, y_train)
#     dataset_test = dataset_make(x_test, y_test)
#     return dataset_train, dataset_test
#     pass
def test():
    list_vals = []
    list_labels = []
    list_label = read_route("data\\" + "00000000" + "\\" + "00000000" + ".route")
    if len(list_label) > 500:
        pass
    else:
        if len(list_label) < 500:
            for _ in range(500 - len(list_label)):
                list_label.append(0)
            try:
                list_labels.append(tf.one_hot(list_label, depth=10000, dtype=tf.float32, axis=0))
            except:
                pass
    """
    这一步开启多线程进行数据处理
    """
    # ca = CANDIDATE("00000000")
    # save_path="deal_with"
    # path="00000000"
    # pool = Pool()
    # pool.map(resize,save_path)
    # pool.close()
    # pool.join()
    # ca.CANDIDATA_GET()


# label的总体list
list_val = []
list_labels = []


def fuct1(path):
    ca = CANDIDATE(path)
    list_val.append(ca.CANDIDATA_GET())


def main():
    begin = time.time()
    try:
        for root, dir, files in os.walk("data"):
            dirs = dir
            for path in dirs:
                fuct1(path)
                list_label = read_route("data\\" + path + "\\" + path + ".route")

                list_label = [int(i) for i in list_label]

                """
                padding操作
                """
                if len(list_label) > 500:
                    list_label = list_label[:500]
                elif len(list_label) < 500:
                    for i in range(500 - len(list_label)):
                        list_label.append(0)

                list_labels.append(tf.one_hot(list_label, depth=10000, dtype=tf.int32, axis=0))

        end = time.time()
        print("消耗时间：", begin - end)
        list_val_np = np.array(list_val)
        print("")
        np.savez('dataset_u_40.npz', val=list_val_np)
    except:
        np.savez('error.npz', val=list_val_np)


main()
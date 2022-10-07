import tensorflow as tf
from mytools import np_function
def tensor_equal(a, b):
    # 判断类型是否均为tensor
    if type(a) != type(b):
        return False
    if isinstance(a, type(tf.constant([]))) is not True:
        if isinstance(a, type(tf.Variable([]))) is not True:
            return False
    # 判断形状相等
    if a.shape != b.shape:
        return False
    # 逐值对比后若有False则不相等
    if not tf.reduce_min(tf.cast(a == b, dtype=tf.int32)):
        return False
    return True

class HAM(tf.keras.metrics.Metric):
    # 计算正确预测的个数
    def __init__(self, name='categorical_tp', **kwargs):
        super(HAM, self).__init__(name=name, **kwargs)

        self.tp = 0
        self.zero=[0 for i in range(10000)]
        self.cout=1
        self.t3=tf.convert_to_tensor([0 for i in range(10000)], dtype=tf.int32)
    def fuction(self,list):
        i=list[0]
        j=list[1]
        if tensor_equal(i, self.t3) or tensor_equal(j, self.t3):
            return
        if tensor_equal(i, j):
            self.tp += 1
        self.cout += 1
    def update_state(self, y_true, y_pred, sample_weight=None):
        # list_tensor=tf.concat([tf.cast(y_pred,tf.int32),y_true],axis=-1)
        # tf.map_fn(fn=self.fuction,elems=list_tensor)
        y_pred=y_pred.numpy().tolist()
        y_true=y_true.numpy().tolist()

        # y_pred=[i.index(1) for i in y_pred]
        # y_true=[i.index(1) for i in y_true]
        y_p=[]
        y_t=[]
        for i1,j1 in zip(y_true,y_pred):
            try:
                dd=i1.index(1)
                y_t.append(dd)
            except:
                y_t.append(None)
            try:
                dd = j1.index(1)
                y_p.append(dd)
            except:
                y_p.append(None)
        # for i,j,k in zip(y_pred.tolist(),y_true.tolist(),range(len(y_pred))):
        #     if i==self.zero or j==self.zero:
        #         break
        for i,j in zip(y_true,y_pred):
            if i==None or j==None:
                break
            if i==j:
                self.tp+=1

            self.cout+=1



    def result(self):
        return self.tp/self.cout

    def reset_states(self):
        self.tp.assign(0.)
class BLEU(tf.keras.metrics.Metric):
    # 计算正确预测的个数
    def __init__(self, name='categorical_tp', **kwargs):
        super(BLEU, self).__init__(name=name, **kwargs)

        self.score = 0
        self.zero=[0 for i in range(10000)]
        self.cout=1
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_p = []
        y_t = []
        for i1, j1 in zip(y_true, y_pred):
            try:
                dd = i1.index(1)
                y_t.append(dd)
            except:
                y_t.append(None)
            try:
                dd = j1.index(1)
                y_p.append(dd)
            except:
                y_p.append(None)
        list1 = set(y_t) & set(y_p)
        self.cout+=1
        self.score+=len(list1) / len(y_pred)



    def result(self):
        return self.score/self.cout

    def reset_states(self):
        self.tp.assign(0.)







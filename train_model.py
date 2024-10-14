"""训练模型，预测黄金涨跌"""
import os
import pickle
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm
from download_data import update_data


def turn_prob_to_label(prob_data):
    """将预测概率转化为标签"""
    result = []
    for each_data in prob_data:
        if isinstance(each_data, (list, np.ndarray)):
            # 多分类情况下
            max_loc = list(each_data).index(max(list(each_data)))
            result.append(str(max_loc))
        else:
            # 二分类情况下
            label = '1' if each_data > 0.5 else '0'
            result.append(label)
    return result


class GoldPricePredictor:
    """预测金价7天后的涨跌，因为需要持有7天才能卖出"""

    def __init__(self, gold_data_path='data/gold_data_update.csv',
                 nasdaq_data_path='data/nasdaq_data_update.csv',
                 model_path='data/lgboostmodel.pickle.dat'):
        self.gold_data_path = gold_data_path
        self.nasdaq_data_path = nasdaq_data_path
        self.model_path = model_path
        self.data = None
        self.model = None

    def load_data(self):
        """加载黄金和纳斯达克指数数据，并进行日期对齐"""
        # 加载黄金数据
        gold_data = pd.read_csv(self.gold_data_path, index_col='Date', parse_dates=True)
        gold_data.sort_index(inplace=True)

        # 加载纳斯达克指数数据
        nasdaq_data = pd.read_csv(self.nasdaq_data_path, index_col='Date', parse_dates=True)
        nasdaq_data.sort_index(inplace=True)

        # 合并数据，取交集日期，确保两个数据都有对应的值
        self.data = pd.merge(gold_data[['Close']], nasdaq_data[['Close']], left_index=True,
                             right_index=True, suffixes=('_gold', '_nasdaq'))
        print("数据加载并合并完成。")

    def create_features_and_labels(self, _window_size=90, hold_days=7):
        """生成特征和标签"""
        gold_close = self.data['Close_gold'].values
        nasdaq_close = self.data['Close_nasdaq'].values

        features = []
        labels = []

        total_length = len(self.data)
        # 最大索引避免越界
        max_index = total_length - hold_days

        # 遍历数据，生成特征和标签
        for i in range(_window_size, max_index):
            # 提取过去window_size天的黄金价格作为特征
            feature_gold = gold_close[i - _window_size:i]
            # 提取过去window_size天的纳斯达克指数收盘价作为特征
            feature_nasdaq = nasdaq_close[i - _window_size:i]
            # 将两个特征组合
            feature = np.concatenate([feature_gold, feature_nasdaq])
            # 当前黄金价格
            current_price = gold_close[i]
            # hold_days天后的黄金价格
            future_price = gold_close[i + hold_days]
            # 标签：未来价格高于当前价格为1，否则为0，要算上0.1%的手续费
            label = 1 if future_price / current_price >= 1.001 else 0
            features.append(feature)
            labels.append(label)

        return np.array(features), np.array(labels)

    @staticmethod
    def shuffle_data(data_x, data_y):
        """
        对输入的numpy数组进行手动打乱
        """
        np.random.seed(20)
        np.random.shuffle(data_x)
        np.random.seed(20)
        np.random.shuffle(data_y)

    def train_lightgbm(self, data_x, data_y):
        """训练LightGBM模型"""
        # 按照时间顺序划分训练集和测试集，前80%为训练集，后20%为测试集
        self.shuffle_data(data_x, data_y)
        x_train, x_test_valid, y_train, y_test_valid = train_test_split(
            data_x, data_y, test_size=0.2, shuffle=True, random_state=1)

        # 初始化模型
        model = LGBMClassifier(max_depth=30, objective='binary', n_estimators=100000,
                               learning_rate=0.01)

        # 训练模型
        model.fit(x_train, y_train,
                  eval_set=[(x_train, y_train), (x_test_valid, y_test_valid)],
                  eval_metric="binary_logloss",
                  callbacks=[lightgbm.log_evaluation(period=1),
                             lightgbm.early_stopping(stopping_rounds=300)])

        # 在验证集上进行预测
        preds = model.predict_proba(x_test_valid)[:, 1]  # 获取正类的概率

        # 将概率转化为标签
        test_result = turn_prob_to_label(preds)
        y_test_valid_labels = [str(label) for label in y_test_valid]

        # 输出前10个预测结果和实际标签
        print("预测结果前10个：", test_result[:10])
        print("实际标签前10个：", y_test_valid_labels[:10])

        # 计算准确率
        test_accuracy = accuracy_score(y_test_valid_labels, test_result)
        print(f'测试集准确率: {test_accuracy * 100.0:.2f}%')
        self.model = model
        if test_accuracy > 0.70:  # 可以根据需要调整阈值
            # 保存模型
            with open(self.model_path, "wb") as _:
                pickle.dump(model, _)
            print("模型已保存。")

    def predict(self, _recent_gold_prices, _recent_nasdaq_prices):
        """使用训练好的模型进行预测
        recent_gold_prices: numpy array，最近90天的黄金价格
        recent_nasdaq_prices: numpy array，最近90天的纳斯达克指数收盘价
        """
        # 确保模型已加载
        if self.model is None:
            if os.path.exists(self.model_path):
                with open(self.model_path, "wb") as _:
                    pickle.dump(self.model, _)
                print("已加载保存的模型。")
            else:
                print("模型文件不存在，请先训练模型。")
                return None

        # 确保输入的特征是二维数组
        if _recent_gold_prices.ndim == 1:
            _recent_gold_prices = _recent_gold_prices.reshape(1, -1)
        if _recent_nasdaq_prices.ndim == 1:
            _recent_nasdaq_prices = _recent_nasdaq_prices.reshape(1, -1)

        # 合并特征
        recent_features = np.concatenate([_recent_gold_prices, _recent_nasdaq_prices], axis=1)

        # 进行预测
        pred_prob = self.model.predict_proba(recent_features)[:, 1]
        _prediction = turn_prob_to_label(pred_prob)
        return _prediction

    def run(self, _window_size=90):
        """运行整个流程"""
        self.load_data()
        data_x, data_y = self.create_features_and_labels(_window_size=_window_size)
        self.train_lightgbm(data_x, data_y)


if __name__ == "__main__":
    update_data()
    predictor = GoldPricePredictor()
    WINDOW_SIZE = 90
    predictor.run(WINDOW_SIZE)

    # 示例：使用最近的90天数据进行预测
    # 假设您想预测最新日期之后的走势
    recent_data = predictor.data[-WINDOW_SIZE:]
    recent_gold_prices = recent_data['Close_gold'].values
    recent_nasdaq_prices = recent_data['Close_nasdaq'].values
    prediction = predictor.predict(recent_gold_prices, recent_nasdaq_prices)
    if prediction is not None:
        if prediction[0] == '1':
            print("模型预测未来7天后黄金价格将上涨。")
        else:
            print("模型预测未来7天后黄金价格将下跌或持平。")

"""用于下载和更新数据"""
import os
from datetime import datetime
import yfinance as yf
import pandas as pd


def download_data(symbol, start_date, end_date, filename):
    """
    下载指定股票代码的历史数据，并保存为CSV文件。

    参数：
    symbol (str): 股票或指数的代码。
    start_date (str): 数据开始日期，格式为'YYYY-MM-DD'。
    end_date (str): 数据结束日期，格式为'YYYY-MM-DD'。
    filename (str): 保存数据的CSV文件名。
    """
    # 下载历史数据
    data = yf.download(symbol, start=start_date, end=end_date)
    # 将数据保存为CSV文件
    data.to_csv(filename)
    print(f"已将{symbol}的数据保存至{filename}")


def _update_data(symbol, start_date, filename):
    """
    更新指定股票代码的历史数据，起始日期由用户输入，终止日期默认为今天。
    如果数据已经是最新的，则不会再次下载。

    参数：
    symbol (str): 股票或指数的代码。
    start_date (str): 数据开始日期，格式为'YYYY-MM-DD'。
    filename (str): 保存数据的CSV文件名。
    """
    # 获取今天的日期，格式为'YYYY-MM-DD'
    end_date = datetime.today().strftime('%Y-%m-%d')

    # 创建一个用于记录更新日期的空文件
    update_date_file = f'{filename}_update_{end_date}.txt'
    with open(update_date_file, 'w') as _:
        pass

    # 检查数据文件是否存在
    if os.path.exists(filename):
        # 读取已有的数据
        existing_data = pd.read_csv(filename, index_col=0, parse_dates=True)
        # 获取已有数据的最后日期
        last_date = existing_data.index[-1]
        # 如果已有数据已经是最新的，则不再下载
        if last_date.strftime('%Y-%m-%d') >= end_date:
            print(f"{symbol}的数据已经是最新的，无需更新")
        else:
            # 从最后日期的下一天开始下载新数据
            new_start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            new_data = yf.download(symbol, start=new_start_date, end=end_date)
            # 如果有新数据，进行更新
            if not new_data.empty:
                updated_data = pd.concat([existing_data, new_data])
                # 删除重复的数据
                updated_data = updated_data[~updated_data.index.duplicated(keep='last')]
                # 保存更新后的数据
                updated_data.to_csv(filename)
                print(f"已更新{symbol}的数据保存至{filename}")
            else:
                print(f"{symbol}没有新的数据需要更新")
    else:
        # 如果数据文件不存在，下载全部数据
        data = yf.download(symbol, start=start_date, end=end_date)
        data.to_csv(filename)
        print(f"已将{symbol}的数据保存至{filename}")


def update_data():
    """
    更新数据
    """
    print("正在更新数据....")
    start_date = '2004-01-01'
    # 黄金ETF的股票代码
    gold_etf = 'GLD'
    # 纳斯达克综合指数的代码
    nasdaq_index = '^IXIC'
    _update_data(gold_etf, start_date, 'data/gold_data_update.csv')
    _update_data(nasdaq_index, start_date, 'data/nasdaq_data_update.csv')
    print("数据更新完成！")


def main():
    """
    主函数，下载黄金ETF和纳斯达克指数的数据，并演示更新函数的使用。
    """
    # 定义初始数据的时间范围
    start_date = '2004-01-01'
    end_date = '2005-09-01'

    # 黄金ETF的股票代码
    gold_etf = 'GLD'
    # 纳斯达克综合指数的代码
    nasdaq_index = '^IXIC'

    # 创建数据保存的目录
    if not os.path.exists('data'):
        os.makedirs('data')

    # 下载黄金ETF的数据
    download_data(gold_etf, start_date, end_date, 'data/gold_data.csv')

    # 下载纳斯达克指数的数据
    download_data(nasdaq_index, start_date, end_date, 'data/nasdaq_data.csv')

    # 更新数据示例，起始日期由用户输入，终止日期默认为今天
    update_data()


if __name__ == "__main__":
    main()

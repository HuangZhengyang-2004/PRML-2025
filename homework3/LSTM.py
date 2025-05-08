import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error as mse
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error

def data_load(full_path):
    df = pd.read_csv(full_path)

    print("原始数据概览:")
    print(df.head())
    print(df.info())

    # 映射风向为数字
    mapping = {'NE': 0, 'SE': 1, 'NW': 2, 'cv': 3}
    df['wnd_dir'] = df['wnd_dir'].map(mapping)

    # 转换为时间索引并排序
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)

    # 特征缩放列
    columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df_scaled[columns])

    # 划分训练集和测试集：按时间最后一年
    last_year = df_scaled.index[-1].year
    df_train = df_scaled[df_scaled.index.year < last_year]
    df_test = df_scaled[df_scaled.index.year == last_year]

    # 构造LSTM训练/测试样本
    n_past = 11
    n_future = 1

    def create_xy(data):
        X, y = [], []
        data = np.array(data)
        for i in range(n_past, len(data) - n_future + 1):
            X.append(data[i - n_past:i, 1:])  # 除了 pollution
            y.append(data[i + n_future - 1:i + n_future, 0])  # pollution
        return np.array(X), np.array(y)

    X_train, y_train = create_xy(df_train)
    X_test, y_test = create_xy(df_test)

    print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
    print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')
    return X_train, y_train, X_test, y_test


def LSTM_model(full_path, flag):
    X_train, y_train, X_test, y_test = data_load(full_path)
    model_path = 'E:/TESTpy/PRML/homework3'

    if flag == 1:
        model = Sequential()
        model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(16, return_sequences=False))
        model.add(Dense(y_train.shape[1]))

        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

        model.summary()
        history = model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping, checkpoint],
            shuffle=False
        )

        # 绘制 loss 曲线
        plt.figure(figsize=(15, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("E:/TESTpy/PRML/homework3/training_loss_curve.png")
        plt.show()

    # 载入最佳模型
    best_model = load_model(model_path)

    # 测试集预测
    test_predictions = best_model.predict(X_test).flatten()
    test_results = pd.DataFrame(data={
        'Train Predictions': test_predictions,
        'Actual': y_test.flatten()
    })

    # 保存预测结果
    with open("test_results.txt", "w") as f:
        f.write(f"{'Train Predictions':<20}{'Actual':<20}\n")
        for _, row in test_results.iterrows():
            f.write(f"{row['Train Predictions']:<20.6f}{row['Actual']:<20.6f}\n")

    # 可视化预测结果
    plt.figure(figsize=(15, 6))
    plt.plot(test_results['Train Predictions'][:350], label='Predicted')
    plt.plot(test_results['Actual'][:350], label='Actual')
    plt.legend()
    plt.title('PM2.5 Prediction vs Actual (前350条)')
    plt.xlabel('Index')
    plt.ylabel('Normalized PM2.5')
    plt.savefig("E:/TESTpy/PRML/homework3/prediction_vs_actual.png")
    plt.show()

    mae = mean_absolute_error(y_test, test_predictions)
    rmse = sqrt(mse(y_test, test_predictions))
    print('Test MAE: %.5f' % mae)
    print('Test RMSE: %.5f' % rmse)


if __name__ == '__main__':
    # 修改为你的实际数据路径
    full_path = "E:/TESTpy/PRML/homework3/LSTM-Multivariate_pollution.csv"
    LSTM_model(full_path, flag=0)

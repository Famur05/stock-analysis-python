import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Загрузка данных о ценах акций с Yahoo Finance
symbol = 'AAPL'  # Акции компании Apple
data = yf.download(symbol, start='2010-01-01', end='2024-01-01')

# 2. Преобразование данных: используем цену закрытия и предыдущие 5 дней
data['Prev Close'] = data['Close'].shift(1)  # Закрытие предыдущего дня
data = data.dropna()  # Убираем строки с отсутствующими значениями

# Используем только данные за последние 1000 дней
data = data.tail(1000)

X = data[['Prev Close']]  # Признак: цена закрытия предыдущего дня
y = data['Close']  # Целевая переменная: цена закрытия текущего дня

# 3. Разделение данных на обучающую и тестовую выборки
train_size = int(len(data) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. Построение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Прогнозирование цены на тестовой выборке
predicted_price = model.predict(X_test)

# 6. Оценка модели: MSE (среднеквадратичная ошибка)
mse = np.mean((y_test - predicted_price) ** 2)
print(f"Среднеквадратичная ошибка: {mse}")

# 7. Визуализация результатов
plt.figure(figsize=(10,6))
plt.plot(y_test.index, y_test, label="Истинные данные", color='blue')
plt.plot(y_test.index, predicted_price, label="Предсказанные данные", color='red')
plt.xlabel('Дата')
plt.ylabel('Цена акции')
plt.title(f"Прогнозирование цены акции {symbol}")
plt.legend()
plt.savefig('stock_prediction.png', dpi=300)
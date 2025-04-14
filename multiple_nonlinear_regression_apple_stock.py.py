# Regresión No Lineal Múltiple para Predicción del Precio de Acciones

# Explicaciones clave:
# - Importación del dataset: Contiene datos históricos de Apple desde 1980 a 2021 con precios y volumen. Pero solo usaremos el año 2020 para la predicción.
# - Preprocesamiento: Se eliminan valores nulos y se ordena por fecha. Se filtra el dataset para el año 2020.
# - Preprocesamiento: Se convierte la fecha a formato numérico y se eliminan valores faltantes.
# - Variables predictoras: 'Open', 'High', 'Low', 'Volume'. (Precios de apertura, máximo, mínimo y volumen de acciones).
# - Variable objetivo: 'Close' (precio de cierre).
# - Entrenamiento del modelo: Se usa un modelo de regresión polinómica de grado 2 para capturar relaciones no lineales.
# - División del dataset: Conjunto de entrenamiento (2/3) y prueba (1/3).
# - Visualización: Dos gráficos con predicción y datos reales para entrenamiento y prueba.
# - Evaluación: Métricas R² y visualización de resultados.

# Importar bibliotecas necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Cargar el dataset
dataset = pd.read_csv('AAPL.csv')

# Eliminar valores nulos por seguridad
dataset = dataset.dropna()

# Convertir la columna Date a formato datetime y ordenar por fecha
dataset['Date'] = pd.to_datetime(dataset['Date'])
# Filtrar el dataset para el año 2020
dataset = dataset[dataset['Date'].dt.year == 2020]   
dataset.sort_values('Date', inplace=True)
# Convertir la fecha en índice si queremos usarla luego
dates = dataset['Date']

# Selección de variables independientes y dependientes
X = dataset[['Open','High', 'Low', 'Volume']].values  # variables predictoras
y = dataset['Close'].values  # variable a predecir

# Crear variables polinómicas (no linealidad)
poly = PolynomialFeatures(degree=2, include_bias=False)  # Puedes probar con degree=3
X_poly = poly.fit_transform(X)

# División temporal (sin shuffle)
split_index = int(len(X_poly) * 2/3)  # 2/3 para entrenamiento, 1/3 para prueba
X_train, X_test = X_poly[:split_index], X_poly[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
dates_train, dates_test = dates[:split_index], dates[split_index:]


## Entrenamiento del modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluación
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Entrenamiento - MSE:", train_mse)
print("Entrenamiento - R²:", train_r2)
print("Prueba - MSE:", test_mse)
print("Prueba - R²:", test_r2)

# 🔹 Gráfico de entrenamiento (orden temporal)
plt.figure(figsize=(12, 6))
plt.plot(dates_train, y_train, label='Real (Train)', color='blue')
plt.plot(dates_train, y_train_pred, label='Predicción (Train)', color='cyan')
plt.title('Predicción de Precio de Cierre (Entrenamiento)', fontsize=14)
plt.xlabel('Año')
plt.xticks(rotation=45)
plt.ylabel('Precio de Cierre (USD)')
plt.text(0.02, 0.95, f'R² = {train_r2:.4f}\nMSE = {train_mse:.4f}', 
         transform=plt.gca().transAxes, fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.8))
plt.legend()
plt.grid(True)
plt.show()

# 🔸 Gráfico de prueba (orden temporal)
plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_test, label='Real (Test)', color='green')
plt.plot(dates_test, y_test_pred, label='Predicción (Test)', color='orange')
plt.title('Predicción de Precio de Cierre (Prueba)', fontsize=14)
plt.xlabel('Año')
plt.xticks(rotation=45)
plt.ylabel('Precio de Cierre (USD)')
plt.text(0.02, 0.95, f'R² = {test_r2:.4f}\nMSE = {test_mse:.4f}', 
         transform=plt.gca().transAxes, fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.8))
plt.legend()
plt.grid(True)
plt.show()
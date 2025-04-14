# 📈 Predicción del Precio de Acciones de Apple mediante Regresión No Lineal Múltiple

Este proyecto implementa un modelo de **Regresión Polinómica Múltiple** para predecir el precio de cierre de las acciones de Apple (AAPL) utilizando datos históricos del año 2020.

## 📁 Dataset

Se utilizó un dataset con datos históricos de Apple desde 1980 a 2021, con las siguientes características:

- `Date`: Fecha de la sesión bursátil
- `Open`: Precio de apertura de la acción
- `High`: Precio máximo durante la sesión
- `Low`: Precio mínimo durante la sesión
- `Volume`: Volumen de acciones negociadas
- `Close`: Precio de cierre (variable objetivo)

Para este análisis específico, se filtró el dataset para incluir únicamente datos del año 2020, un periodo que incluye tanto la caída bursátil por la pandemia de COVID-19 como la posterior recuperación.

## ⚙️ Metodología

1. **Preprocesamiento**:
   - Limpieza de valores nulos
   - Ordenamiento cronológico de los datos
   - Filtrado para el año 2020

2. **Variables del modelo**:
   - **Predictoras (X)**: 'Open', 'High', 'Low', 'Volume'
   - **Objetivo (y)**: 'Close' (precio de cierre)

3. **Transformación polinómica**:
   - Se aplicó una transformación polinómica de grado 2 para capturar relaciones no lineales entre las variables
   - Esta transformación generó términos cuadráticos y productos cruzados de las variables originales

4. **División del dataset**:
   - Conjunto de entrenamiento: 2/3 del dataset (enero a agosto 2020)
   - Conjunto de prueba: 1/3 del dataset (septiembre a diciembre 2020)

5. **Entrenamiento del modelo**:
   - Se utilizó `LinearRegression` de scikit-learn sobre las características polinómicas

## 📊 Resultados

### Entrenamiento

![Entrenamiento](resultados%20entrenamiento.png)

- **R² = 0.9907**: El modelo explica el 99.07% de la varianza en los datos de entrenamiento
- **MSE = 2.4606**: Error cuadrático medio bajo, indicando una alta precisión en la predicción
- Se observa un excelente ajuste del modelo a los datos de entrenamiento, siguiendo con precisión tanto tendencias alcistas como bajistas
- El modelo captura con precisión la fuerte caída de marzo 2020 y la recuperación posterior

### Prueba

![Prueba](resultados%20prueba.png)

- **R² = 0.7571**: El modelo explica el 75.71% de la varianza en los datos no vistos, un rendimiento sólido
- **MSE = 11.3062**: Error cuadrático medio más elevado que en entrenamiento, pero aceptable para datos financieros
- El modelo sigue razonablemente bien la tendencia general, aunque con desviaciones notables en ciertos picos
- Se observa una tendencia a sobreestimar los valores en algunos períodos (especialmente a finales de diciembre)

## 📈 Análisis Técnico

- **Alto R² en entrenamiento (0.9907)**: Indica que el modelo se ajusta extremadamente bien a los datos utilizados para su construcción
- **R² moderado-alto en prueba (0.7571)**: Muestra buena capacidad de generalización, aunque con cierta pérdida de precisión
- **Diferencia entre MSE de entrenamiento y prueba**: Sugiere cierto nivel de sobreajuste, pero no crítico considerando la volatilidad inherente a los mercados financieros
- **Buen seguimiento de tendencias generales**: El modelo captura adecuadamente tanto periodos alcistas como bajistas

## 🧠 Conclusiones

1. **Efectividad del modelo polinómico**: La transformación polinómica de grado 2 captura relaciones no lineales importantes entre las variables, mejorando la capacidad predictiva comparada con un modelo lineal simple
   
2. **Alta precisión en entrenamiento**: El modelo muestra una excelente capacidad para ajustarse a los datos históricos, especialmente durante periodos de alta volatilidad como la caída de marzo 2020

3. **Capacidad predictiva sólida**: Un R² de 0.7571 en el conjunto de prueba indica buena capacidad para predecir precios futuros basados en los patrones aprendidos

4. **Limitaciones observadas**: 
   - Mayor error en datos de prueba, especialmente en periodos de alta volatilidad o cambios abruptos
   - Tendencia a sobreestimar algunos valores, particularmente a finales del periodo de prueba

## 🛠 Tecnologías Utilizadas

- **Python**: Lenguaje de programación principal
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Cálculos numéricos
- **Matplotlib**: Visualización de resultados
- **Scikit-learn**: Implementación del modelo de regresión y métricas de evaluación

## 💡 Insights Clave

1. La transformación polinómica proporciona una mejora significativa sobre un modelo lineal para datos financieros
2. El modelo captura adecuadamente tanto la caída por COVID-19 como la recuperación posterior
3. La precisión en datos no vistos (R² = 0.7571) es notable considerando la naturaleza volátil de los mercados bursátiles
4. El enfoque demuestra valor para análisis de tendencias y soporte en decisiones de inversión, aunque no debe ser la única herramienta utilizada

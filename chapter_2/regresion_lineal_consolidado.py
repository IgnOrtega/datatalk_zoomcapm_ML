"""
Proyecto: Predicción del precio de autos con Regresión Lineal
ML Zoomcamp - Módulo 2 (Regression)

Código consolidado de las lecciones 01 a 15:
    1. Carga y preparación de los datos
    2. Análisis exploratorio (EDA)
    3. Framework de validación (train / val / test)
    4. Regresión lineal (ecuación normal) + regularización
    5. Métrica RMSE
    6. Ingeniería de características y variables categóricas
    7. Ajuste del parámetro de regularización r
    8. Modelo final, evaluación en test y predicción de un auto

Uso:
    python regresion_lineal_consolidado.py
"""

import numpy as np
import pandas as pd

DATA_URL = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'

# Poner en False para correr sin gráficos (no requiere matplotlib/seaborn)
MOSTRAR_GRAFICOS = True

SEED = 2
BASE = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
CATEGORICAL_VARIABLES = [
    'make', 'engine_fuel_type', 'transmission_type', 'driven_wheels',
    'market_category', 'vehicle_size', 'vehicle_style'
]


# ---------------------------------------------------------------------------
# 1. Carga y preparación de los datos (lección 02)
# ---------------------------------------------------------------------------

def load_data(path=DATA_URL):
    df = pd.read_csv(path)

    # Nombres de columnas consistentes: minúsculas y '_' en vez de espacios
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Normalizar también los valores de las columnas de texto
    strings = list(df.dtypes[df.dtypes == 'object'].index)
    for col in strings:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    return df


# ---------------------------------------------------------------------------
# 2. Análisis exploratorio (lección 03)
# ---------------------------------------------------------------------------

def eda(df):
    for col in df.columns:
        print(col)
        print(df[col].unique()[:5])
        print(df[col].nunique())
        print()

    print('Valores nulos por columna:')
    print(df.isnull().sum())
    print()

    if MOSTRAR_GRAFICOS:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # El precio tiene una cola larga -> se aplica log1p para normalizarlo
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        sns.histplot(df.msrp, bins=50, ax=axes[0]).set_title('msrp')
        sns.histplot(df.msrp[df.msrp < 100000], bins=50, ax=axes[1]).set_title('msrp < 100k')
        sns.histplot(np.log1p(df.msrp), bins=50, ax=axes[2]).set_title('log1p(msrp)')
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# 3. Framework de validación (lección 04)
# ---------------------------------------------------------------------------

def split_data(df, seed=SEED):
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    # Mezclar los registros de forma reproducible
    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    df_test = df.iloc[idx[n_train + n_val:]].reset_index(drop=True)

    # Variable objetivo en escala logarítmica
    y_train = np.log1p(df_train.msrp.values)
    y_val = np.log1p(df_val.msrp.values)
    y_test = np.log1p(df_test.msrp.values)

    # Quitar msrp de las features para no usar el objetivo como entrada
    del df_train['msrp']
    del df_val['msrp']
    del df_test['msrp']

    return df_train, df_val, df_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# 4. Regresión lineal: ecuación normal w = (XᵀX)⁻¹Xᵀy (lecciones 05-07, 13)
# ---------------------------------------------------------------------------

def train_linear_regression(X, y):
    # Columna de unos para el sesgo w0 (la "feature ficticia")
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


def train_linear_regression_reg(X, y, r=0.001):
    # Igual que la anterior, pero suma r a la diagonal de XᵀX (regularización)
    # para evitar una matriz casi singular y pesos enormes
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


def predict(X, w0, w):
    return w0 + X.dot(w)


# ---------------------------------------------------------------------------
# 5. RMSE (lección 09)
# ---------------------------------------------------------------------------

def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)


# ---------------------------------------------------------------------------
# 6. Ingeniería de características y variables categóricas (lecciones 10-12)
# ---------------------------------------------------------------------------

def get_categories(df_train, top=5):
    # Para cada variable categórica, los `top` valores más frecuentes en train
    categories = {}
    for c in CATEGORICAL_VARIABLES:
        categories[c] = list(df_train[c].value_counts().head(top).index)
    return categories


def prepare_X(df, categories):
    # Se copia el dataframe para no modificar los datos originales
    df = df.copy()
    features = BASE.copy()

    # Antigüedad del auto (el dataset es de 2017)
    df['age'] = 2017 - df.year
    features.append('age')

    # One-hot encoding del número de puertas
    for v in [2, 3, 4]:
        df['num_doors_%s' % v] = (df.number_of_doors == v).astype('int')
        features.append('num_doors_%s' % v)

    # One-hot encoding de las demás variables categóricas
    for c, values in categories.items():
        for v in values:
            df['%s_%s' % (c, v)] = (df[c] == v).astype('int')
            features.append('%s_%s' % (c, v))

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X


def plot_predictions(y_pred, y_real, title=''):
    if not MOSTRAR_GRAFICOS:
        return
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.histplot(y_pred, label='prediction', color='red', alpha=0.5, bins=50)
    sns.histplot(y_real, label='target', color='blue', alpha=0.5, bins=50)
    plt.title(title)
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------------

def main():
    # 1-2. Datos y EDA
    df = load_data()
    eda(df)

    # 3. Split train / val / test
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)
    print('Tamaños train/val/test:', len(df_train), len(df_val), len(df_test))

    # 4-5. Modelo baseline: solo features numéricas base (lecciones 08-10)
    X_train = df_train[BASE].fillna(0).values
    X_val = df_val[BASE].fillna(0).values
    w0, w = train_linear_regression(X_train, y_train)
    y_pred = predict(X_val, w0, w)
    print('RMSE baseline (val):', rmse(y_val, y_pred))
    plot_predictions(y_pred, y_val, 'Baseline')

    # 6. Modelo con age + puertas + categóricas, sin regularización.
    #    Puede dar un RMSE enorme por XᵀX casi singular (lección 12)
    categories = get_categories(df_train)
    X_train = prepare_X(df_train, categories)
    X_val = prepare_X(df_val, categories)
    w0, w = train_linear_regression(X_train, y_train)
    print('RMSE con categóricas, sin regularizar (val):', rmse(y_val, predict(X_val, w0, w)))

    # 7. Ajuste de r con regularización (lecciones 13-14)
    print('\nr, w0, RMSE val')
    for r in [0.0, 0.00001, 0.0001, 0.001, 0.1, 1, 10]:
        w0, w = train_linear_regression_reg(X_train, y_train, r=r)
        score = rmse(y_val, predict(X_val, w0, w))
        print(r, w0, score)

    r = 0.001
    w0, w = train_linear_regression_reg(X_train, y_train, r=r)
    y_pred = predict(X_val, w0, w)
    print('\nRMSE con r=%s (val):' % r, rmse(y_val, y_pred))
    plot_predictions(y_pred, y_val, 'Regularizado r=%s' % r)

    # 8. Modelo final: entrenar con train + val y evaluar en test (lección 15)
    df_full_train = pd.concat([df_train, df_val]).reset_index(drop=True)
    y_full_train = np.concatenate([y_train, y_val])

    X_full_train = prepare_X(df_full_train, categories)
    w0, w = train_linear_regression_reg(X_full_train, y_full_train, r=r)

    X_test = prepare_X(df_test, categories)
    print('RMSE final (test):', rmse(y_test, predict(X_test, w0, w)))

    # Predecir el precio de un auto concreto
    car = df_test.iloc[20].to_dict()
    print('\nAuto de ejemplo:', car)

    X_small = prepare_X(pd.DataFrame([car]), categories)
    y_pred = predict(X_small, w0, w)[0]
    print('Precio predicho: $%.2f' % np.expm1(y_pred))
    print('Precio real:     $%.2f' % np.expm1(y_test[20]))


if __name__ == '__main__':
    main()

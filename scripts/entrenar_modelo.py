import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import joblib

# Cargar los datos desde pacientes.csv
def cargar_datos():
    if os.path.exists('pacientes.csv'):
        df = pd.read_csv('pacientes.csv')
        return df
    else:
        print("El archivo pacientes.csv no existe.")
        return None

# Preprocesamiento de datos
def preprocesar_datos(df):
    # Asegúrate de que el archivo CSV tenga las columnas correctas
    if 'edad' not in df.columns or 'fuma' not in df.columns or 'actividad_fisica' not in df.columns:
        print("Las columnas necesarias no están presentes en el archivo CSV.")
        return None
    
    # Aquí podrías agregar más procesamiento si es necesario, como codificación de datos.
    return df

# Entrenar el modelo
def entrenar_modelo(df):
    # Características de entrada y etiquetas
    X = df[['edad', 'fuma', 'actividad_fisica', 'historial_familiar', 'dieta', 'estres', 'visitas_medico', 'sueño']]
    y = df['riesgo']  # Asumiendo que 'riesgo' es la etiqueta que queremos predecir

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Crear y entrenar el modelo RandomForest
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Evaluar precisión
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisión del modelo: {accuracy:.2f}')

    # Guardar el modelo entrenado
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/modelo_riesgo.pkl')
    print('Modelo entrenado y guardado en "models/modelo_riesgo.pkl"')

    return model

# Entrenar el modelo con los datos
def actualizar_modelo():
    # Cargar los datos desde pacientes.csv
    df = cargar_datos()

    if df is not None:
        # Preprocesar los datos si es necesario
        df = preprocesar_datos(df)

        if df is not None:
            # Entrenar el modelo con los nuevos datos
            modelo = entrenar_modelo(df)
        else:
            print("No se pudieron procesar los datos.")
    else:
        print("No se pueden actualizar los datos.")

# Llamar a la función para actualizar el modelo
actualizar_modelo()

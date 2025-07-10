import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df_pacientes = pd.read_csv("data/pacientes.csv")

df_pacientes['sexo'] = df_pacientes['sexo'].map({'M': 1, 'F': 0}) 
df_pacientes['fuma'] = df_pacientes['fuma'].map({1: 1, 0: 0}) 
df_pacientes['actividad_fisica'] = df_pacientes['actividad_fisica'].map({1: 1, 0: 0}) 
df_pacientes['historial_familiar'] = df_pacientes['historial_familiar'].map({1: 1, 0: 0})
df_pacientes['dieta'] = df_pacientes['dieta'].map({1: 1, 0: 0})
df_pacientes['estres'] = df_pacientes['estres'].map({1: 1, 0: 0})
df_pacientes['visitas_medico'] = df_pacientes['visitas_medico'].map({1: 1, 0: 0})

def asignar_riesgo(diagnostico):

    if pd.isna(diagnostico) or diagnostico == "":
        return 0  

    enfermedades_posibles = diagnostico.split(",")
    enfermedades_con_porcentaje = {}

    for enfermedad in enfermedades_posibles:
        nombre, porcentaje = enfermedad.split(":")
        enfermedades_con_porcentaje[nombre.strip()] = float(porcentaje.strip().replace('%', ''))

    enfermedad_mayor_probabilidad = max(enfermedades_con_porcentaje, key=enfermedades_con_porcentaje.get)

    enfermedades_bajo_riesgo = [
        'resfriado', 'alergia', 'migraña', 'gastroenteritis', 'sinusitis', 'celulitis_bacteriana', 
        'fibromialgia', 'sindrome_de_fatiga_cronica', 'sindrome_del_intestino_irritable', 
    ]

    enfermedades_medio_riesgo = [
        'anemia', 'esclerosis_multiple', 'hipotiroidismo', 'hipertiroidismo', 'celiaquia', 
        'lupus', 'endometriosis', 'epoc', 'hemochromatosis', 'cálculos_renales'
    ]

    enfermedades_alto_riesgo = [
        'gripe', 'dengue', 'covid19', 'apendicitis', 'asma', 'neumonía', 'insuficiencia_renal', 
        'diabetes', 'hipertensión', 'tuberculosis', 'hepatitis', 'meningitis', 'gripe_aviaria', 'viruela'
    ]
    
    if any(enfermedad in enfermedad_mayor_probabilidad for enfermedad in enfermedades_bajo_riesgo):
        return 0  
    elif any(enfermedad in enfermedad_mayor_probabilidad for enfermedad in enfermedades_medio_riesgo):
        return 1  
    elif any(enfermedad in enfermedad_mayor_probabilidad for enfermedad in enfermedades_alto_riesgo):
        return 2 
    else:
        return 0 

df_pacientes['riesgo'] = df_pacientes['diagnostico'].apply(asignar_riesgo)

features = ['edad', 'sexo', 'fuma', 'actividad_fisica', 'historial_familiar', 'dieta', 'estres', 'visitas_medico', 'sueño']

X = df_pacientes[features]

y = df_pacientes['riesgo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Precisión del modelo: {accuracy_score(y_test, y_pred):.2f}")


df_pacientes['riesgo_predicho'] = model.predict(X)

df_pacientes['riesgo_predicho'] = df_pacientes['riesgo_predicho'].map({0: 'Bajo', 1: 'Medio', 2: 'Alto'})


df_pacientes['riesgo_predicho'].value_counts().plot(kind='bar', color=['red', 'green', 'yellow'])
plt.title('Distribución de pacientes por grupos de riesgo')
plt.xlabel('Grupo de riesgo')
plt.ylabel('Número de pacientes')
plt.xticks(rotation=0)


print(df_pacientes[['nombre', 'edad', 'sexo', 'fuma', 'actividad_fisica', 'riesgo_predicho']])
plt.show()

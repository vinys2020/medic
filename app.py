from flask import Flask, render_template, request, jsonify, session
from pyswip import Prolog
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import pandas as pd
import joblib
import csv
import matplotlib
matplotlib.use('Agg')  # Usa el backend 'Agg' para desactivar la GUI
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import logging




app = Flask(__name__)
app.secret_key = 'tu_clave_secreta'  

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


model_path = 'models/modelo_riesgo.pkl'
if os.path.exists(model_path):
    risk_model = joblib.load(model_path)
    print("Modelo cargado con éxito.")
else:
    print(f"No se encontró el archivo en {model_path}.")

# Cargar detalles de enfermedades desde el CSV
def cargar_detalles_enfermedades():
    df = pd.read_csv('data/detalle_enfermedades.csv')  # Lee el archivo CSV
    # Crear un diccionario donde la clave es el nombre de la enfermedad y el valor es el detalle
    return {row['Enfermedad'].lower(): row['Descripcion'] for index, row in df.iterrows()}

# Cargar detalles de enfermedades al iniciar el servidor
detalles_enfermedades = cargar_detalles_enfermedades()

@app.route('/detalle_enfermedades', methods=['POST'])
def detalle_enfermedad():
    # Obtener el nombre de la enfermedad desde la solicitud JSON
    enfermedad = request.json.get('enfermedad')

    # Verificar que se haya enviado un nombre de enfermedad
    if not enfermedad:
        return jsonify({'error': 'El nombre de la enfermedad es obligatorio.'}), 400

    # Buscar el detalle de la enfermedad (ignorando mayúsculas y minúsculas)
    detalle = detalles_enfermedades.get(enfermedad.lower())

    if detalle:
        # Si se encuentra el detalle, devolver la respuesta con la enfermedad y el detalle
        return jsonify({'enfermedad': enfermedad, 'detalle': detalle})
    else:
        # Si no se encuentra el detalle, devolver un mensaje de error
        return jsonify({'error': 'Detalle no encontrado para la enfermedad proporcionada.'}), 404



def cargar_enfermedades():
    prolog = Prolog()
    df = pd.read_csv('data/enfermedades.csv')
    
    for index, row in df.iterrows():
        enfermedad = row['enfermedad']
        sintoma = row['sintoma']
        prolog.assertz(f"sintoma({sintoma}, {enfermedad})")

    return prolog

def cargar_preguntas():
    df = pd.read_csv('data/preguntas.csv')
    return {row['enfermedad']: row['pregunta'] for index, row in df.iterrows()}

def cargar_tratamientos():
    df = pd.read_csv('data/tratamientos.csv')
    return {row['enfermedad']: row['tratamiento'] for index, row in df.iterrows()}

def cargar_acciones_descartar():
    df = pd.read_csv('data/acciones_descartar.csv')
    return {row['enfermedad']: row['accion'] for index, row in df.iterrows()}

prolog = cargar_enfermedades()
preguntas = cargar_preguntas()
tratamientos = cargar_tratamientos()
acciones_descartar = cargar_acciones_descartar()

def normalizar_porcentajes(diagnostico):
    total = sum(diagnostico.values())
    if total == 0:
        return {k: 0 for k in diagnostico}  

    return {enfermedad: round((conteo / total) * 100, 2) for enfermedad, conteo in diagnostico.items()}

def diagnosticar(sintomas_usuario):
    enfermedades_posibles = {}
    
    for sintoma in sintomas_usuario:
        query = f"sintoma({sintoma}, Enfermedad)"
        resultados = list(prolog.query(query))
        for resultado in resultados:
            enfermedad = resultado['Enfermedad']
            if enfermedad not in enfermedades_posibles:
                enfermedades_posibles[enfermedad] = 0
            enfermedades_posibles[enfermedad] += 1  

    return normalizar_porcentajes(enfermedades_posibles)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enfermedades')
def enfermedades():
    return render_template('enfermedades.html')


@app.route('/estadisticas')
def estadisticas():
    graph_url_diagnosticos = generar_grafico_diagnosticos()
    graph_url_edad = generar_grafico_edad_por_diagnostico()
    
    return render_template('estadisticas.html', 
                           graph_url_diagnosticos=graph_url_diagnosticos,
                           graph_url_edad=graph_url_edad)




@app.route('/guardar_riesgo', methods=['POST'])
def guardar_riesgo():
    try:
        data = request.json
        
        edad = data.get('edad', '0')
        sexo = data.get('sexo', 'Desconocido')
        fuma = data.get('fuma', '0')
        actividad_fisica = data.get('actividadFisica', '0')
        historial_familiar = data.get('historialFamiliar', '0')
        dieta = data.get('dieta', '0')
        estres = data.get('estres', '0')
        visitas_medico = data.get('visitasMedico', '0')
        sueño = data.get('sueño', '0')
        sintomas = data.get('sintomas', '')

        if not edad or not sintomas:
            return jsonify({'error': 'Edad y síntomas son campos obligatorios'}), 400

        sintomas_usuario = [sintoma.strip().replace(' ', '_') for sintoma in sintomas.split(',')]

        diagnostico_resultados = data.get('resultados', [])

        if diagnostico_resultados:
            diagnostico_str = ', '.join([f"{resultado['enfermedad']}: {resultado['porcentaje']}%" for resultado in diagnostico_resultados])
        else:
            diagnostico_str = "No se encontraron enfermedades"
        

        columns = ['edad', 'sexo', 'fuma', 'actividad_fisica', 'historial_familiar', 'dieta', 'estres', 'visitas_medico', 'sueño', 'sintomas', 'diagnostico']

        file_exists = os.path.isfile('data/pacientes.csv')
        
        with open('data/pacientes.csv', mode='a', newline='') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
            
            if not file_exists:
                writer.writerow(columns)

            sintomas_str = ', '.join(sintomas_usuario)

            writer.writerow([edad, sexo, fuma, actividad_fisica, historial_familiar, dieta, estres, visitas_medico, sueño, sintomas_str, diagnostico_str])

        return jsonify({'message': 'Datos guardados con éxito', 'diagnostico': diagnostico_str})
    
    except KeyError as e:
        logger.error(f"KeyError: Faltan datos en la solicitud - {str(e)}")
        return jsonify({'error': f"Datos faltantes: {str(e)}"}), 400
    except FileNotFoundError as e:
        logger.error(f"FileNotFoundError: No se pudo acceder al archivo - {str(e)}")
        return jsonify({'error': "No se pudo acceder al archivo de datos."}), 500
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        return jsonify({'error': 'Ocurrió un error inesperado. Por favor, inténtelo más tarde.'}), 500





def generar_grafico_diagnosticos():
    df = pd.read_csv('data/pacientes.csv')

    df['diagnostico_principal'] = df['diagnostico'].apply(lambda x: x.split(':')[0].strip())
    
    diagnostico_counts = df['diagnostico_principal'].value_counts()

    plt.figure(figsize=(8, 6))
    diagnostico_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribución de Diagnósticos')
    plt.xlabel('Diagnósticos Realizados')
    plt.ylabel('Número de Pacientes')
    plt.xticks(rotation=45, ha='right', fontsize=8.5)  

    plt.tight_layout()

    ruta_imagen = os.path.join('static', 'img', 'grafico_diagnosticos.png')
    plt.savefig(ruta_imagen, format='png')
    plt.close()
    
    return ruta_imagen

def generar_grafico_edad_por_diagnostico():
    df = pd.read_csv('data/pacientes.csv')
    
    df['diagnostico_principal'] = df['diagnostico'].apply(lambda x: x.split(':')[0].strip())
    
    edad_promedio_por_diagnostico = df.groupby('diagnostico_principal')['edad'].mean()
    
    plt.figure(figsize=(8, 6))
    edad_promedio_por_diagnostico.plot(kind='bar', color='lightgreen')
    plt.title('Edad Promedio por Diagnóstico')
    plt.xlabel('Diagnósticos Realizados')
    plt.ylabel('Edad Promedio')
    plt.xticks(rotation=45, ha='right', fontsize=8.5)  

    plt.tight_layout()

    ruta_imagen = os.path.join('static', 'img', 'grafico_edad_por_diagnostico.png')
    plt.savefig(ruta_imagen, format='png')
    plt.close()
    
    return ruta_imagen




@app.route('/diagnosticar', methods=['POST'])
def iniciar_diagnostico():
    sintomas = request.json['sintomas']
    sintomas_usuario = [sintoma.strip().replace(' ', '_') for sintoma in sintomas.split(',')]
    
    diagnostico = diagnosticar(sintomas_usuario)
    
    if diagnostico:
        session['diagnostico'] = diagnostico
        session['enfermedades'] = list(diagnostico.keys())
        session['respuestas'] = {}
        session['pregunta_index'] = 0
        
        return jsonify({
            'pregunta': preguntas.get(session['enfermedades'][0]),
            'enfermedad': session['enfermedades'][0]
        })
    
    return jsonify({"message": "No se encontraron enfermedades para los síntomas ingresados."})

@app.route('/respuesta', methods=['POST'])
def procesar_respuesta():

    if 'respuestas' not in session:
        session['respuestas'] = {}
    
    if 'pregunta_index' not in session:
        session['pregunta_index'] = 0
    
    if 'enfermedades' not in session:

        session['enfermedades'] = ['enfermedad1', 'enfermedad2', 'enfermedad3']  

    if 'diagnostico' not in session:
        session['diagnostico'] = {'enfermedad1': 50, 'enfermedad2': 30, 'enfermedad3': 20}  

    enfermedad = request.json['enfermedad']
    respuesta = request.json['respuesta']

    session['respuestas'][enfermedad] = respuesta

    session['pregunta_index'] += 1

    if session['pregunta_index'] < len(session['enfermedades']):
        siguiente_enfermedad = session['enfermedades'][session['pregunta_index']]
        return jsonify({
            'pregunta': preguntas.get(siguiente_enfermedad, 'Pregunta no encontrada'),  
            'enfermedad': siguiente_enfermedad
        })
    
    nuevo_diagnostico = {}
    
    for enfermedad, respuesta in session['respuestas'].items():
        porcentaje_base = session['diagnostico'].get(enfermedad, 0)
        if respuesta:
            nuevo_diagnostico[enfermedad] = porcentaje_base
    
    nuevo_diagnostico = normalizar_porcentajes(nuevo_diagnostico)

    resultados = []
    for enfermedad, porcentaje in nuevo_diagnostico.items():
        tratamiento = tratamientos.get(enfermedad, "Consulta a tu médico para más información.")
        accion_descartar = acciones_descartar.get(enfermedad, "Consulta a tu médico para más información.")
        resultados.append({
            'enfermedad': enfermedad,
            'porcentaje': porcentaje,
            'tratamiento': tratamiento,
            'accion_descartar': accion_descartar
        })
    
    session.pop('diagnostico', None)
    session.pop('enfermedades', None)
    session.pop('respuestas', None)
    session.pop('pregunta_index', None)

    return jsonify(resultados)


if __name__ == '__main__':
    app.run(debug=True)


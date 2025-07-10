from flask import Flask, render_template, request, jsonify, session
from pyswip import Prolog
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import pandas as pd
import joblib
import csv
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import logging
import firebase_admin
from firebase_admin import credentials, firestore
from flask import send_from_directory


cred = credentials.Certificate('diagnostico-medicodb-firebase-adminsdk-fbsvc-ccf5810950.json')
firebase_admin.initialize_app(cred)

db = firestore.client()


app = Flask(__name__)
app.secret_key = 'tu_clave_secreta'  

@app.route('/public/<path:filename>')
def public_files(filename):
    return send_from_directory('public', filename)

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


model_path = 'models/modelo_riesgo.pkl'
if os.path.exists(model_path):
    risk_model = joblib.load(model_path)
    print("Modelo cargado con éxito.")
else:
    print(f"No se encontró el archivo en {model_path}.")

def cargar_detalles_enfermedades_firestore():
    detalles = {}
    docs = db.collection('Enfermedades').stream()
    for doc in docs:
        data = doc.to_dict()
        nombre = data.get('nombre', '').lower()
        descripcion = data.get('descripcion', '')
        if nombre:
            detalles[nombre] = descripcion
    return detalles

detalles_enfermedades = cargar_detalles_enfermedades_firestore()

@app.route('/detalle_enfermedades', methods=['POST'])
def detalle_enfermedad():

    enfermedad = request.json.get('enfermedad')

    if not enfermedad:
        return jsonify({'error': 'El nombre de la enfermedad es obligatorio.'}), 400

    detalle = detalles_enfermedades.get(enfermedad.lower())

    if detalle:

        return jsonify({'enfermedad': enfermedad, 'detalle': detalle})
    else:

        return jsonify({'error': 'Detalle no encontrado para la enfermedad proporcionada.'}), 404



def cargar_enfermedades_firestore():
    prolog = Prolog()

    try:
        docs = db.collection('Enfermedades').stream()

        for doc in docs:
            data = doc.to_dict()

            # Normalizamos el nombre de la enfermedad (espacios a guiones bajos)
            nombre_original = data.get('nombre', '')
            if not nombre_original:
                continue

            enfermedad = nombre_original.strip().replace(' ', '_')

            # Cargamos los síntomas
            sintomas = data.get('sintomas', [])
            for sintoma in sintomas:
                sintoma_normalizado = sintoma.strip().replace(' ', '_')
                regla = f"sintoma({sintoma_normalizado}, {enfermedad})"
                try:
                    prolog.assertz(regla)
                except Exception as e:
                    print(f"Error al cargar regla Prolog '{regla}': {e}")

        return prolog

    except Exception as e:
        print(f"Error al cargar enfermedades desde Firestore: {e}")
        return prolog  # Retorna el prolog, aunque esté vacío para evitar errores

prolog = cargar_enfermedades_firestore()



def cargar_preguntas_firestore():
    preguntas = {}
    docs = db.collection('Enfermedades').stream()
    for doc in docs:
        data = doc.to_dict()
        enfermedad = data.get('nombre', '')
        pregunta = data.get('pregunta', '')
        if enfermedad:
            preguntas[enfermedad] = pregunta
    return preguntas

preguntas = cargar_preguntas_firestore()

def cargar_tratamientos_firestore():
    tratamientos = {}
    docs = db.collection('Enfermedades').stream()
    for doc in docs:
        data = doc.to_dict()
        enfermedad = data.get('nombre', '')
        tratamiento = data.get('tratamiento', '')
        if enfermedad:
            tratamientos[enfermedad] = tratamiento
    return tratamientos

tratamientos = cargar_tratamientos_firestore()

def cargar_acciones_descartar_firestore():
    acciones = {}
    docs = db.collection('Enfermedades').stream()
    for doc in docs:
        data = doc.to_dict()
        enfermedad = data.get('nombre', '')
        accion = data.get('accion', '')
        if enfermedad:
            acciones[enfermedad] = accion
    return acciones

prolog = cargar_enfermedades_firestore()
preguntas = cargar_preguntas_firestore()
tratamientos = cargar_tratamientos_firestore()
acciones_descartar = cargar_acciones_descartar_firestore()

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

        nombre = data.get('nombre')
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

        # Normalizar los síntomas
        sintomas_usuario = [sintoma.strip().replace(' ', '_') for sintoma in sintomas.split(',')]

        diagnostico_resultados = data.get('resultados', [])

        if diagnostico_resultados:
            diagnostico_str = ', '.join([f"{resultado['enfermedad']}: {resultado['porcentaje']}%" for resultado in diagnostico_resultados])
        else:
            diagnostico_str = "No se encontraron enfermedades"

        # ✅ Guardar en Firestore
        paciente_ref = db.collection('pacientes').document()
        paciente_ref.set({
            'nombre': nombre,
            'edad': int(edad),
            'sexo': sexo,
            'fuma': fuma,
            'actividad_fisica': actividad_fisica,
            'historial_familiar': historial_familiar,
            'dieta': dieta,
            'estres': estres,
            'visitas_medico': visitas_medico,
            'sueno': sueño,
            'sintomas': sintomas_usuario,
            'diagnostico': diagnostico_str,
        })

        return jsonify({'message': 'Datos guardados con éxito', 'diagnostico': diagnostico_str})

    except KeyError as e:
        logger.error(f"KeyError: Faltan datos en la solicitud - {str(e)}")
        return jsonify({'error': f"Datos faltantes: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        return jsonify({'error': 'Ocurrió un error inesperado. Por favor, inténtelo más tarde.'}), 500






def generar_grafico_diagnosticos():
    # Obtener los documentos de la colección 'pacientes'
    pacientes_ref = db.collection('pacientes')
    docs = pacientes_ref.stream()

    # Crear una lista de diccionarios con los datos
    datos = []
    for doc in docs:
        paciente = doc.to_dict()
        if 'diagnostico' in paciente and paciente['diagnostico']:  # Validación
            datos.append({
                'diagnostico': paciente['diagnostico']
            })

    # Crear un DataFrame
    df = pd.DataFrame(datos)

    if df.empty:
        print("No hay datos disponibles para generar el gráfico.")
        return None

    # Procesar para obtener el diagnóstico principal
    df['diagnostico_principal'] = df['diagnostico'].apply(lambda x: x.split(':')[0].strip())
    diagnostico_counts = df['diagnostico_principal'].value_counts()

    # Generar el gráfico
    plt.figure(figsize=(8, 6))
    diagnostico_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribución de Diagnósticos')
    plt.xlabel('Diagnósticos Realizados')
    plt.ylabel('Número de Pacientes')
    plt.xticks(rotation=45, ha='right', fontsize=8.5)
    plt.tight_layout()

    # Guardar la imagen
    ruta_imagen = os.path.join('static', 'img', 'grafico_diagnosticos.png')
    plt.savefig(ruta_imagen, format='png')
    plt.close()

    return ruta_imagen

def generar_grafico_edad_por_diagnostico():
    db = firestore.client()
    pacientes_ref = db.collection('pacientes')
    docs = pacientes_ref.stream()

    # Recolectar los datos en listas
    edades = []
    diagnosticos_principales = []

    for doc in docs:
        data = doc.to_dict()
        edad = data.get('edad')
        diagnostico = data.get('diagnostico')

        if edad is not None and diagnostico:
            diagnostico_principal = diagnostico.split(':')[0].strip()
            edades.append(edad)
            diagnosticos_principales.append(diagnostico_principal)

    # Verificar que haya datos
    if not edades or not diagnosticos_principales:
        return None

    # Crear un DataFrame con los datos recolectados
    df = pd.DataFrame({
        'edad': edades,
        'diagnostico_principal': diagnosticos_principales
    })

    # Agrupar y calcular edad promedio
    edad_promedio_por_diagnostico = df.groupby('diagnostico_principal')['edad'].mean().sort_values()

    # Generar gráfico
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

    diagnostico = diagnosticar(sintomas_usuario)  # Tu función que devuelve dict con enfermedades

    if diagnostico:
        session['diagnostico'] = diagnostico
        session['enfermedades'] = list(diagnostico.keys())
        session['respuestas'] = {}
        session['pregunta_index'] = 0
        
        # Obtener datos de la primera enfermedad desde Firestore
        enfermedad_id = session['enfermedades'][0]
        doc_ref = db.collection('Enfermedades').document(enfermedad_id)
        doc = doc_ref.get()
        
        if doc.exists:
            enfermedad_data = doc.to_dict()
            pregunta = enfermedad_data.get('pregunta', 'No hay pregunta definida')
            
            return jsonify({
                'pregunta': pregunta,
                'enfermedad': enfermedad_id
            })
        else:
            return jsonify({"message": "Error: enfermedad no encontrada en la base de datos."}), 404

    return jsonify({"message": "No se encontraron enfermedades para los síntomas ingresados."})

@app.route('/respuesta', methods=['POST'])
def procesar_respuesta():
    if 'respuestas' not in session:
        session['respuestas'] = {}
    if 'pregunta_index' not in session:
        session['pregunta_index'] = 0
    if 'enfermedades' not in session or 'diagnostico' not in session:
        return jsonify({"error": "Sesión inválida. Por favor inicia un nuevo diagnóstico."}), 400

    enfermedad = request.json.get('enfermedad')
    respuesta = request.json.get('respuesta')

    if not enfermedad:
        return jsonify({"error": "No se indicó la enfermedad para la respuesta."}), 400

    session['respuestas'][enfermedad] = respuesta

    session['pregunta_index'] += 1

    if session['pregunta_index'] < len(session['enfermedades']):
        siguiente_enfermedad = session['enfermedades'][session['pregunta_index']]
        pregunta_siguiente = preguntas.get(siguiente_enfermedad, "Pregunta no encontrada")
        return jsonify({
            'pregunta': pregunta_siguiente,
            'enfermedad': siguiente_enfermedad
        })

    # Si ya no quedan preguntas, calcular nuevo diagnóstico
    nuevo_diagnostico = {}
    for enf, resp in session['respuestas'].items():
        porcentaje_base = session['diagnostico'].get(enf, 0)
        if resp:  # Solo si la respuesta es afirmativa
            nuevo_diagnostico[enf] = porcentaje_base

    nuevo_diagnostico = normalizar_porcentajes(nuevo_diagnostico)

    resultados = []
    for enf, porcentaje in nuevo_diagnostico.items():
        tratamiento = tratamientos.get(enf, "Consulta a tu médico para más información.")
        accion_descartar = acciones_descartar.get(enf, "Consulta a tu médico para más información.")
        resultados.append({
            'enfermedad': enf,
            'porcentaje': porcentaje,
            'tratamiento': tratamiento,
            'accion': accion_descartar
        })

    # Limpiar sesión para nuevo diagnóstico
    session.pop('respuestas', None)
    session.pop('pregunta_index', None)
    session.pop('enfermedades', None)
    session.pop('diagnostico', None)

    return jsonify({'resultados': resultados})



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))


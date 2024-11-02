from flask import Flask, render_template, request, jsonify, session
from pyswip import Prolog
import pandas as pd

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta'  # Necesario para usar sesiones

# Cargar enfermedades y síntomas
def cargar_enfermedades():
    prolog = Prolog()
    df = pd.read_csv('enfermedades.csv')
    
    for index, row in df.iterrows():
        enfermedad = row['enfermedad']
        sintoma = row['sintoma']
        prolog.assertz(f"sintoma({sintoma}, {enfermedad})")

    return prolog

# Cargar preguntas
def cargar_preguntas():
    df = pd.read_csv('preguntas.csv')
    return {row['enfermedad']: row['pregunta'] for index, row in df.iterrows()}

# Cargar tratamientos
def cargar_tratamientos():
    df = pd.read_csv('tratamientos.csv')
    return {row['enfermedad']: row['tratamiento'] for index, row in df.iterrows()}

# Cargar acciones de descartar
def cargar_acciones_descartar():
    df = pd.read_csv('acciones_descartar.csv')
    return {row['enfermedad']: row['accion'] for index, row in df.iterrows()}

prolog = cargar_enfermedades()
preguntas = cargar_preguntas()
tratamientos = cargar_tratamientos()
acciones_descartar = cargar_acciones_descartar()

# Normalizar porcentajes para que sumen 100%
def normalizar_porcentajes(diagnostico):
    total = sum(diagnostico.values())
    if total == 0:
        return {k: 0 for k in diagnostico}  # Si no hay síntomas, retorna todos en 0%

    return {enfermedad: round((conteo / total) * 100, 2) for enfermedad, conteo in diagnostico.items()}

# Diagnóstico basado en síntomas
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

    # Normalizar los porcentajes
    return normalizar_porcentajes(enfermedades_posibles)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnosticar', methods=['POST'])
def iniciar_diagnostico():
    sintomas = request.json['sintomas']
    sintomas_usuario = [sintoma.strip().replace(' ', '_') for sintoma in sintomas.split(',')]
    
    diagnostico = diagnosticar(sintomas_usuario)
    
    if diagnostico:
        # Guardar el diagnóstico en la sesión para continuar
        session['diagnostico'] = diagnostico
        session['enfermedades'] = list(diagnostico.keys())
        session['respuestas'] = {}
        session['pregunta_index'] = 0
        
        # Obtener la primera pregunta
        return jsonify({
            'pregunta': preguntas.get(session['enfermedades'][0]),
            'enfermedad': session['enfermedades'][0]
        })
    
    return jsonify({"message": "No se encontraron enfermedades para los síntomas ingresados."})

@app.route('/respuesta', methods=['POST'])
def procesar_respuesta():
    enfermedad = request.json['enfermedad']
    respuesta = request.json['respuesta']  # true o false

    # Guardar la respuesta
    session['respuestas'][enfermedad] = respuesta
    
    # Avanzar al siguiente diagnóstico
    session['pregunta_index'] += 1
    
    if session['pregunta_index'] < len(session['enfermedades']):
        siguiente_enfermedad = session['enfermedades'][session['pregunta_index']]
        return jsonify({
            'pregunta': preguntas.get(siguiente_enfermedad),
            'enfermedad': siguiente_enfermedad
        })
    
    # Calcular el diagnóstico final
    nuevo_diagnostico = {}
    
    for enfermedad, respuesta in session['respuestas'].items():
        porcentaje_base = session['diagnostico'][enfermedad]
        if respuesta:
            nuevo_diagnostico[enfermedad] = porcentaje_base
    
    # Normalizar porcentajes finales para mostrar resultados que sumen 100%
    nuevo_diagnostico = normalizar_porcentajes(nuevo_diagnostico)

    # Mostrar resultado con tratamientos y acciones de descartar
    resultados = []
    for enfermedad in nuevo_diagnostico:
        tratamiento = tratamientos.get(enfermedad, "Consulta a tu médico para más información.")
        accion_descartar = acciones_descartar.get(enfermedad, "Consulta a tu médico para más información.")
        resultados.append({
            'enfermedad': enfermedad,
            'porcentaje': nuevo_diagnostico[enfermedad],
            'tratamiento': tratamiento,
            'accion_descartar': accion_descartar
        })
    
    # Limpiar la sesión
    session.pop('diagnostico', None)
    session.pop('enfermedades', None)
    session.pop('respuestas', None)
    session.pop('pregunta_index', None)
    
    return jsonify(resultados)

if __name__ == '__main__':
    app.run(debug=True)

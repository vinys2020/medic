import tkinter as tk
from tkinter import messagebox
from pyswip import Prolog
import pandas as pd

# Cargar enfermedades y síntomas
def cargar_enfermedades():
    prolog = Prolog()
    df = pd.read_csv('enfermedades.csv')
    
    for index, row in df.iterrows():
        enfermedad = row['enfermedad']
        sintoma = row['sintoma']
        prolog.assertz(f"sintoma({sintoma}, {enfermedad})")

    return prolog

prolog = cargar_enfermedades()

# Preguntas únicas por enfermedad
preguntas = {
    "gripe": "¿Ha tenido fiebre alta en los últimos días?",
    "dengue": "¿Ha sentido dolor intenso en las articulaciones recientemente?",
    "resfriado": "¿Ha notado congestión nasal persistente?",
    "covid19": "¿Ha tenido dificultad para respirar?",
    "alergia": "¿Ha presentado estornudos frecuentes sin fiebre?",
    "migraña": "¿Siente náuseas o vómitos durante el dolor de cabeza?",
    "sinusitis": "¿Siente presión en la frente o detrás de los ojos?",
    "apendicitis": "¿Ha notado un dolor agudo en la parte baja del abdomen?",
    "asma": "¿Experimenta dificultad al realizar actividades físicas?",
    "neumonía": "¿Siente una respiración más difícil de lo normal?",
    "gastroenteritis": "¿Ha tenido episodios frecuentes de diarrea?",
    "insuficiencia_renal": "¿Siente hinchazón en las piernas o tobillos?",
    "diabetes": "¿Siente sed excesiva a pesar de beber agua?",
    "hipertensión": "¿Ha experimentado confusión mental o mareos?",
    "artritis": "¿Ha notado hinchazón en las articulaciones afectadas?",
    "cálculos_renales": "¿Siente dolor al orinar o sangre en la orina?",
    "dermatitis": "¿Ha notado enrojecimiento o inflamación en la piel?",
    "gripe_aviaria": "¿Ha tenido fiebre alta y dificultad para respirar?",
    "meningitis": "¿Ha experimentado fiebre y rigidez en el cuello?",
    "tuberculosis": "¿Ha notado sudores nocturnos excesivos?",
    "hepatitis": "¿Ha tenido cambios en el color de su piel o ojos?",
    "viruela": "¿Ha tenido contacto cercano con alguien con erupciones?"
}

# Diagnóstico basado en síntomas
def diagnosticar(sintomas_usuario):
    enfermedades_posibles = {}
    total_sintomas = 0
    
    for sintoma in sintomas_usuario:
        query = f"sintoma({sintoma}, Enfermedad)"
        resultados = list(prolog.query(query))
        for resultado in resultados:
            enfermedad = resultado['Enfermedad']
            if enfermedad not in enfermedades_posibles:
                enfermedades_posibles[enfermedad] = 0
            enfermedades_posibles[enfermedad] += 1
            total_sintomas += 1  

    diagnostico_con_porcentaje = {}
    if total_sintomas > 0:
        for enfermedad, conteo in enfermedades_posibles.items():
            porcentaje = (conteo / total_sintomas) * 100
            diagnostico_con_porcentaje[enfermedad] = round(porcentaje, 2)

    return diagnostico_con_porcentaje

# Diccionario de tratamientos
tratamientos = {
    "gripe": "Descanso, mucha hidratación, analgésicos para aliviar el dolor y la fiebre.",
    "dengue": "Reposo, hidratación constante, analgésicos (paracetamol), evitar aspirina.",
    "resfriado": "Descanso, tomar líquidos, descongestionantes y analgésicos.",
    "covid19": "Aislamiento, descanso, hidratación, y monitoreo de síntomas. Consulta médica si es necesario.",
    "alergia": "Antihistamínicos, evitar alérgenos, uso de gotas oculares y nasales.",
    "migraña": "Analgésicos, descanso en un lugar oscuro, medicamentos específicos para migraña.",
    "sinusitis": "Analgésicos, descongestionantes, irrigación nasal con solución salina.",
    "apendicitis": "Cirugía de emergencia (apendicectomía).",
    "asma": "Uso de inhaladores (broncodilatadores), evitar desencadenantes, seguimiento médico.",
    "neumonía": "Antibióticos (si es bacteriana), descanso, líquidos, y seguimiento médico.",
    "gastroenteritis": "Reposo, hidratación, y evitar alimentos irritantes.",
    "insuficiencia_renal": "Dieta baja en sodio, diálisis en casos graves, seguimiento médico.",
    "diabetes": "Control de glucosa, dieta equilibrada, ejercicio y medicación (insulina o antidiabéticos).",
    "hipertensión": "Dieta baja en sal, ejercicio regular, medicación antihipertensiva.",
    "artritis": "Analgésicos, fisioterapia, medicamentos antiinflamatorios.",
    "cálculos_renales": "Hidratación, analgésicos, posible intervención quirúrgica si son grandes.",
    "dermatitis": "Cremas hidratantes, corticoides tópicos, evitar alérgenos.",
    "gripe_aviaria": "Hospitalización, antivirales, soporte respiratorio en casos graves.",
    "meningitis": "Antibióticos o antivirales según el caso, hospitalización y seguimiento.",
    "tuberculosis": "Antibióticos (terapia prolongada), seguimiento médico.",
    "hepatitis": "Reposo, dieta equilibrada, y evitar alcohol.",
    "viruela": "Tratamiento de soporte, aislamiento, antivirales en casos graves."
}

# Diccionario de acciones médicas
acciones_descartar = {
    "gripe": "Realizar un test rápido de influenza.",
    "dengue": "Solicitar un análisis de sangre para verificar el recuento de plaquetas.",
    "resfriado": "Realizar un examen físico para descartar infecciones bacterianas.",
    "covid19": "Hacer una prueba PCR o de antígenos para COVID-19.",
    "alergia": "Hacer pruebas de alergia específicas.",
    "migraña": "Consultar a un neurólogo para una evaluación exhaustiva.",
    "sinusitis": "Realizar una tomografía computarizada de senos paranasales.",
    "apendicitis": "Solicitar una ecografía o tomografía abdominal.",
    "asma": "Hacer pruebas de función pulmonar.",
    "neumonía": "Realizar una radiografía de tórax.",
    "gastroenteritis": "Consultar sobre el historial alimenticio y realizar análisis de heces.",
    "insuficiencia_renal": "Realizar análisis de sangre y orina para evaluar la función renal.",
    "diabetes": "Solicitar un examen de glucosa en sangre.",
    "hipertensión": "Hacer un monitoreo de presión arterial durante varias visitas.",
    "artritis": "Realizar radiografías o análisis de sangre para marcadores de inflamación.",
    "cálculos_renales": "Solicitar una tomografía abdominal o ecografía.",
    "dermatitis": "Consultar a un dermatólogo para pruebas específicas.",
    "gripe_aviaria": "Hacer pruebas específicas para virus aviares.",
    "meningitis": "Realizar una punción lumbar para análisis del líquido cefalorraquídeo.",
    "tuberculosis": "Solicitar una prueba de tuberculina o una radiografía de tórax.",
    "hepatitis": "Realizar análisis de sangre para marcadores virales.",
    "viruela": "Consultar a un médico especialista en enfermedades infecciosas."
}


# Modificar la función para incluir tratamientos y acciones de descartar
def obtener_diagnostico():
    sintomas = entrada_sintomas.get().split(',')
    sintomas_usuario = [sintoma.strip().replace(' ', '_') for sintoma in sintomas]
    
    diagnostico = diagnosticar(sintomas_usuario)
    
    if diagnostico:
        posibles_enfermedades = sorted(diagnostico.items(), key=lambda x: x[1], reverse=True)[:3]
        total_preguntas = len(posibles_enfermedades)
        
        respuestas = {}
        for enfermedad, _ in posibles_enfermedades:
            respuesta = messagebox.askyesno("Pregunta de Diagnóstico", preguntas[enfermedad])
            respuestas[enfermedad] = respuesta
        
        certeza_total = sum(respuestas.values())
        nuevo_diagnostico = {}
        
        for enfermedad, respuesta in respuestas.items():
            porcentaje_base = diagnostico[enfermedad]
            if certeza_total > 0:
                porcentaje = (respuesta / certeza_total) * 100
            else:
                porcentaje = porcentaje_base / total_preguntas
            
            nuevo_diagnostico[enfermedad] = round(porcentaje, 2)
        
        nuevo_diagnostico = {enfermedad: porcentaje for enfermedad, porcentaje in nuevo_diagnostico.items() if porcentaje > 0}
        
        suma_porcentajes = sum(nuevo_diagnostico.values())
        if suma_porcentajes != 100:
            factor_ajuste = 100 / suma_porcentajes
            for enfermedad in nuevo_diagnostico:
                nuevo_diagnostico[enfermedad] = round(nuevo_diagnostico[enfermedad] * factor_ajuste, 2)

        # Mostrar resultado con tratamientos y acciones de descartar
        message = "Posibles enfermedades:\n"
        for enfermedad, porcentaje in nuevo_diagnostico.items():
            tratamiento = tratamientos.get(enfermedad, "Consulta a tu médico para más información.")
            accion_descartar = acciones_descartar.get(enfermedad, "Consulta a tu médico para más información.")
            message += f"{enfermedad}: {porcentaje}%\nTratamiento: {tratamiento}\nAcción para descartar: {accion_descartar}\n\n"
        
    else:
        message = "No se encontraron enfermedades para los síntomas ingresados."
    
    messagebox.showinfo("Diagnóstico", message)


# Interfaz gráfica
ventana = tk.Tk()
ventana.title("Sistema de Diagnóstico Médico")

icono = tk.PhotoImage(file="img/diagnostico medico.png")  # Cambia esta ruta a la ubicación de tu archivo .png
ventana.iconphoto(True, icono)

width = 400
height = 180
screen_width = ventana.winfo_screenwidth()
screen_height = ventana.winfo_screenheight()
x = (screen_width // 2) - (width // 2)
y = (screen_height // 2) - (height // 2)

ventana.geometry(f"{width}x{height}+{x}+{y}")

label_instrucciones = tk.Label(ventana, text="Ingrese los síntomas separados por comas:")
label_instrucciones.pack(pady=10)

entrada_sintomas = tk.Entry(ventana, width=30)
entrada_sintomas.pack(pady=10)

boton_diagnosticar = tk.Button(ventana, text="Diagnosticar", command=obtener_diagnostico)
boton_diagnosticar.pack(pady=20)

ventana.mainloop()

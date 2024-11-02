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

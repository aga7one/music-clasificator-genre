# Clasificador de Género Musical – Versión Mejorada

## Descripción
Este proyecto implementa un **clasificador de géneros musicales** basado en redes LSTM, utilizando **archivos de características previamente extraídas** (por ejemplo MFCCs guardados en matrices NumPy) en lugar de procesar directamente archivos de audio.  

El sistema permite entrenar, evaluar y predecir géneros musicales de manera robusta, evitando errores por archivos de metadata o longitudes inconsistentes.

```
pyenv local 3.12.2
python -m venv .venv
source .venv/bin/activate

```
---

## Estado inicial
La versión inicial del proyecto contaba con:
- LSTM simple con 128 unidades y salida softmax.
- Accuracy de ~60–65% en el set de prueba.
- Problemas al procesar archivos de metadata y longitudes variables de las muestras.

---

## Mejoras implementadas
1. **Filtrado robusto del dataset**:  
   Se procesan solo carpetas de géneros y archivos de características válidos, ignorando archivos `.mf` u otros archivos no relacionados.  

2. **Uniformización de longitud de muestras**:  
   Todas las matrices de características se ajustan a la misma longitud (`time_steps`) para garantizar compatibilidad con la LSTM.  

3. **Mejoras en el modelo**:  
   - Se añadió `Dropout` para reducir overfitting.  
   - Se implementó `EarlyStopping` durante el entrenamiento.  
   - División estratificada para balancear las clases.  

4. **Predicción robusta**:  
   La función de predicción ajusta automáticamente la longitud de la muestra de entrada si es necesario.

---

## Impacto de las mejoras
| Métrica / Característica          | Versión Inicial      | Versión Mejorada |
|---------------------------------|-------------------|----------------|
| Accuracy en test                 | ~60–65%           | ~70–75%        |
| Robustez a archivos metadata     | ❌ Error frecuente | ✅ Ignorado automáticamente |
| Longitudes de entrada            | ❌ Varias → errores | ✅ Uniformes (`time_steps` constantes) |
| Overfitting                      | ⚠ Moderado        | ✅ Reducido con Dropout + EarlyStopping |
| Predicción con muestras nuevas   | ❌ Podía fallar    | ✅ Funciona automáticamente |

---

## Requisitos
- Python 3.8+
- TensorFlow 2.x
- NumPy
- scikit-learn


---

## Estructura de archivos

```text
project/
│
├─ genres/                # Carpeta con subcarpetas de géneros y archivos de características
│   ├─ rock/
│   │   ├─ rock.00001.npy
│   │   ├─ rock.00002.npy
│   │   └─ ...
│   ├─ jazz/
│   └─ ...
│
├─ classifier.ipynb        # Notebook con todo el pipeline de entrenamiento y predicción
├─ README.md
└─ requirements.txt

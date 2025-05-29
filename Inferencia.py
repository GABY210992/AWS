import joblib

# Datos fijos para prueba
features = [[200, 100, 40]]
# Cargar el modelo
modelo = joblib.load("modelo_regresion.pkl")
# Hacer la predicción
prediccion = modelo.predict(features)
print("Predicción:", prediccion[0])

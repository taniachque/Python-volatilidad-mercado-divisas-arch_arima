import sys
import numpy as np
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QLineEdit, QPushButton, QMessageBox, QGraphicsView, QGraphicsScene
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Configuración de la ventana principal
        self.setStyleSheet("background-color: #b4bbc5;")
        self.setWindowTitle("MODELOS ARCH y ARIMA")
        self.setGeometry(100, 100, 800, 600)  # Establecer una geometría inicial

        self.showMaximized()  # Mostrar la ventana en pantalla completa
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)  # Habilitar el botón de maximizar
        
        # Fuente global
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(12)

        # Crear el layout principal como QHBoxLayout
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(50, 50, 50, 50)

        # Crear el contenedor principal como QWidget
        self.central_widget = QWidget()
        self.central_widget.setContentsMargins(20, 20, 20, 20)
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        # Crear el layout de la primera columna como QVBoxLayout
        self.column1_layout = QVBoxLayout()
       
        # Título de la aplicación
        self.titulo = QLabel("Modelo ARCH y ARIMA")
        self.titulo.setFont(QFont("Arial", 24))
        self.titulo.setStyleSheet("color: red;")
        self.column1_layout.addWidget(self.titulo)

        # Etiquetas e input para los datos de la serie temporal
        self.label1 = QLabel("Ingrese los datos de la serie temporal en Bs:")
        self.label2 = QLabel("Debe ingresar los valores separados por espacios, no comas")
        self.label3 = QLabel("Los valores ingresados con decimales deben marcarse con puntos (.) no comas.")
        self.input_edit = QLineEdit()
        self.input_edit.setStyleSheet("background-color: #ffffff;")
        self.input_edit.setFixedHeight(50)
        self.input_edit.setFont(QFont("Arial", 14))

        self.button = QPushButton("CALCULAR GANANCIAS")
        self.button.setStyleSheet("background-color: #b1575e;")

        # Crear el botón y agregarlo a la ventana
        self.vaciar_button = QPushButton('HACER NUEVO CÁLCULO')
        self.column1_layout.addWidget(self.label1)
        self.column1_layout.addWidget(self.label2)
        self.column1_layout.addWidget(self.label3)
        self.column1_layout.addWidget(self.input_edit)
        self.column1_layout.addWidget(self.button)
        self.column1_layout.addWidget(self.vaciar_button)

        # Agregar el layout de la primera columna al layout principal
        self.layout.addLayout(self.column1_layout)

        # Crear el layout de la segunda columna como QVBoxLayout
        self.column2_layout = QVBoxLayout()

        # Etiquetas para los resultados y la interpretación
        self.resultados_label = QLabel()
        self.interpretacion_arch_label = QLabel()
        self.interpretacion_arima_label = QLabel()
        self.iterations_label = QLabel("", self)
        self.escalado_label = QLabel("", self)

        # Aplicar la fuente a los widgets deseados
        self.label1.setFont(font)
        self.label2.setFont(font)
        self.label3.setFont(font)
        self.resultados_label.setFont(font)
        self.interpretacion_arch_label.setFont(font)
        self.interpretacion_arima_label.setFont(font)
        self.iterations_label.setFont(font)
        self.escalado_label.setFont(font)

        self.column2_layout.addWidget(self.resultados_label)
        self.column2_layout.addWidget(self.interpretacion_arch_label)
        self.column2_layout.addWidget(self.interpretacion_arima_label)
        self.column2_layout.addWidget(self.iterations_label)
        self.column2_layout.addWidget(self.escalado_label)

        # Crear un widget de QGraphicsView para mostrar el gráfico
        self.graphics_view = QGraphicsView(self)
        self.column2_layout.addWidget(self.graphics_view)

        # Agregar el layout de la segunda columna al layout principal
        self.layout.addLayout(self.column2_layout)

        # Crear una figura de matplotlib
        self.fig = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.fig)

        # Crear una escena de QGraphicsScene
        self.scene = QGraphicsScene(self)
        self.scene.addWidget(self.canvas)

        # Establecer la escena en self.graphics_view
        self.graphics_view.setScene(self.scene)

        # Conectar el botón a la función de cálculo
        self.button.clicked.connect(self.calcular_ganancias)

        # Conectar el botón a la función para vaciar los elementos
        self.vaciar_button.clicked.connect(self.vaciar_elementos)

        # Aplicar estilos adicionales
        self.setStyleSheet("""
            QLabel#resultados_label {
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 10px;
            }
            
            QLabel#interpretacion_arch_label, QLabel#interpretacion_arima_label {
                margin-bottom:10px;
                font-size: 14px;
                font-weight: bold;
            }

            QLabel#iterations_label, QLabel#escalado_label {
                margin-top: 20px;
                font-size: 14px;
                font-weight: bold;
            }
        """)

        # Estilos para el botón "Calcular"
        calcular_button_style = """
            QPushButton#calcular_button {
                background-color: #b1575e;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: none;
                padding: 10px 20px;
            }

            QPushButton#calcular_button:hover {
                background-color: #d96b71;
            }

            QPushButton#calcular_button:pressed {
                background-color: #a1484e;
            }
        """

        # Estilos para el botón "Vaciar"
        vaciar_button_style = """
            QPushButton#vaciar_button {
                background-color: #5e89b1;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: none;
                padding: 10px 20px;
            }

            QPushButton#vaciar_button:hover {
                background-color: #7199d9;
            }

            QPushButton#vaciar_button:pressed {
                background-color: #4e6ca1;
            }
        """

        # Establecer los estilos para los botones
        self.button.setObjectName("calcular_button")
        self.button.setStyleSheet(calcular_button_style)

        self.vaciar_button.setObjectName("vaciar_button")
        self.vaciar_button.setStyleSheet(vaciar_button_style) 

    def calcular_ganancias(self):
        # Obtener los datos de la serie temporal ingresados por el usuario
        datos = self.input_edit.text()

        # Convertir los datos a una lista de números
        valores = datos.split()
        try:
            # Validar que todos los valores sean números
            datos_validos = [float(valor) for valor in valores]
        except ValueError:
            self.mostrar_error("Error: Los datos ingresados no son válidos.")
            return

        if len(datos_validos) < 2:
            self.mostrar_error2("Error: Debe ingresar al menos dos valores.")
            return
        
        if len(datos_validos) > 0:
            # Verificar si es necesario escalar los datos
            escalar = False
            max_valor = max(datos_validos)
            if max_valor > 1000:
                escalar = True

            # Escalar los datos si es necesario
            if escalar:
                datos_validos = [valor * 0.01 for valor in datos_validos]
                self.mostrar_escalado()

            # Ajustar el modelo ARCH a los datos
            modelo_arch = arch_model(datos_validos)
            resultado_arch = modelo_arch.fit()

            # Obtener los residuos del modelo ARCH
            residuos = resultado_arch.resid

            # Ajustar el modelo ARIMA a los residuos del modelo ARCH
            modelo_arima = ARIMA(residuos, order=(1, 0, 0))
            resultado_arima = modelo_arima.fit()

            # Calcular las ganancias esperadas
            media_residuos = np.mean(residuos)
            varianza_residuos = np.var(residuos)
            ganancias_esperadas_arch = media_residuos + np.sqrt(varianza_residuos)
            ganancias_esperadas_arima = resultado_arima.params[0]

            # Interpretar los resultados del modelo ARCH
            interpretacion_arch = ""
            if ganancias_esperadas_arch > 0:
                interpretacion_arch = "El modelo ARCH indica que se esperan ganancias positivas en la serie temporal. Esto sugiere que se espera un aumento en la volatilidad en el futuro."
            elif ganancias_esperadas_arch < 0:
                interpretacion_arch = "El modelo ARCH indica que se esperan ganancias negativas en la serie temporal. Esto sugiere que se espera una disminución en la volatilidad en el futuro."
            else:
                interpretacion_arch = "El modelo ARCH no muestra una expectativa clara de ganancias en la serie temporal. Esto sugiere que la volatilidad podría mantenerse estable en el futuro."

            # Interpretar los resultados del modelo ARIMA
            interpretacion_arima = ""
            if ganancias_esperadas_arima > 0:
                interpretacion_arima = "El modelo ARIMA indica que se esperan ganancias positivas en la serie temporal. Esto sugiere un aumento en la tendencia alcista en el futuro."
            elif ganancias_esperadas_arima < 0:
                interpretacion_arima = "El modelo ARIMA indica que se esperan ganancias negativas en la serie temporal. Esto sugiere una disminución en la tendencia bajista en el futuro."
            else:
                interpretacion_arima = "El modelo ARIMA no muestra una expectativa clara de ganancias en la serie temporal. Esto sugiere que la tendencia podría mantenerse estable en el futuro."


            # Mostrar los resultados debajo del botón de calcular ganancias
            resultados = f" * Volatilidad estimada (ARCH): {ganancias_esperadas_arch}\n"
            resultados += f" * Ganancias esperadas (ARIMA): {ganancias_esperadas_arima}"
            #resultados += f"Iteraciones: {iteraciones}\n"
            #resultados += f"Evaluaciones de funciones: {evaluaciones_funciones}\n"
            #resultados += f"Evaluaciones de gradientes: {evaluaciones_gradientes}"

            self.mostrar_resultados(resultados)
            self.mostrar_interpretacion_arch(interpretacion_arch)
            self.mostrar_interpretacion_arima(interpretacion_arima)

            # Obtener las predicciones del modelo ARIMA
            predicciones_arima = resultado_arima.predict()

            # Limpiar la figura antes de dibujar
            self.fig.clear()

            # Crear un eje para el gráfico dentro de la figura
            ax = self.fig.add_subplot(111)

            # Graficar los datos originales
            ax.plot(residuos, label='Datos originales')

            # Graficar las predicciones del modelo ARIMA
            ax.plot(predicciones_arima, label='Predicciones ARIMA')

            # Establecer etiquetas y título del gráfico
            ax.set_xlabel('Tiempo')
            ax.set_ylabel('Valores')
            ax.set_title('Gráfico del modelo ARIMA')

            # Mostrar la leyenda
            ax.legend()

            # Actualizar el canvas de matplotlib
            self.canvas.draw()



    def mostrar_escalado(self):
        mensaje = "Los datos han sido escalados para mejorar la convergencia del modelo."
        self.escalado_label.setText(mensaje)
        self.escalado_label.setWordWrap(True)
    
    def mostrar_resultados(self, resultados):
        self.resultados_label.setText(resultados)

    def mostrar_interpretacion_arch(self, interpretacion):
        self.interpretacion_arch_label.setText(interpretacion)
        self.interpretacion_arch_label.setWordWrap(True)


    def mostrar_interpretacion_arima(self, interpretacion):
        self.interpretacion_arima_label.setText(interpretacion)
        self.interpretacion_arima_label.setWordWrap(True)


    def mostrar_error(self, mensaje):
        QMessageBox.critical(self, "Error", mensaje)

    def mostrar_error2(self, mensaje):
        QMessageBox.critical(self, "Error", mensaje + " Ingrese más datos por favor.")


    def vaciar_elementos(self):
        # Vaciar la caja de entrada de datos
        self.input_edit.clear()

        # Vaciar los resultados
        self.resultados_label.clear()
        self.interpretacion_arch_label.clear()
        self.interpretacion_arima_label.clear()

        # Limpiar la gráfica
        self.fig.clear()
        self.canvas.draw()

        # Restablecer mensaje de escalado
        self.escalado_label.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
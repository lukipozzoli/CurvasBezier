from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt

#Funcion para ecuaciones de bezier
def bezier_f0(t, p0, p1):
    return (1-t)*p0 + t*p1

def bezier_f1(t, p1, p2):
    return (1-t)*p1 + t*p2

def bezier_g(t,f1, f2):
    return (1-t)*f1 + t*f2

# Graficamos la curva de Bézier cuadrática
def graficar_curva_bezier_cuadratica(t_values, puntos_de_control):
    bezier_curve = np.array([bezier_g(t, bezier_f0(t,p0,p1), bezier_f1(t,p1,p2)) for t in t_values])

    plt.figure(figsize=(8, 6))
    plt.plot(bezier_curve[:, 0], bezier_curve[:, 1], label='g(t)')
    plt.plot(puntos_de_control[:, 0], puntos_de_control[:, 1], 'ro--', label='Puntos de control', markersize=8)
    plt.title('Curva de Bézier Cuadrática')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.show()

########## PUNTO 2 #########

def bezier_cubic(t, p0, p1, p2, p3):
    return (1-t)**3 * p0 + 3 * (1-t)**2 * t * p1 + 3 * (1-t) * t**2 * p2 + t**3 * p3

# Graficar la curva de Bézier
def graficar_curva_bezier_cubic(t_values, puntos_de_control):
    bezier_curve2 = np.array([bezier_cubic(t, *puntos_de_control) for t in t_values])

    plt.figure(figsize=(8, 6))
    plt.plot(bezier_curve2[:, 0], bezier_curve2[:, 1], label='Curva de Bézier cúbica')
    plt.scatter(puntos_de_control[:, 0], puntos_de_control[:, 1], c='red', s=80)
    plt.plot(puntos_de_control[:, 0], puntos_de_control[:, 1], 'ro--', label='Puntos de Control')

    for j, (x, y) in enumerate(puntos_de_control):
        plt.text(x, y, f'p{j}')

    plt.title('Curva de Bézier cúbica')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.show()

def graficar_coeficientes_cubicos(t_values, coef_c0, coef_c1, coef_c2, coef_c3):
    plt.figure(figsize=(8, 6))
    plt.plot(t_values, coef_c0, label='Coeficiente de p0: (1-t)^3')
    plt.plot(t_values, coef_c1, label='Coeficiente de p1: 3t(1-t)^2')
    plt.plot(t_values, coef_c2, label='Coeficiente de p2: 3t^2(1-t)')
    plt.plot(t_values, coef_c3, label='Coeficiente de p3: t^3')
    plt.title('Coeficientes de los puntos de control en función de t')
    plt.xlabel('t')
    plt.ylabel('Coeficientes')
    plt.legend()
    plt.grid()
    plt.show()

###### PUNTO 3########
def grafico_envolvente_convexa(puntos_de_control):
    envolvente = ConvexHull(puntos_de_control)
    plt.figure(figsize=(8, 6))
    plt.scatter(puntos_de_control[:, 0], puntos_de_control[:, 1], color='red', label='Puntos de Control', s=100, zorder=2)
    for simplex in envolvente.simplices:
        plt.plot(puntos_de_control[simplex, 0], puntos_de_control[simplex, 1], '--ro')
    for i, punto in enumerate(puntos_de_control):
        plt.text(punto[0], punto[1], f'P{i+1}', fontsize=12, ha='right', color='black')
    plt.title('Envolvente Convexa de Puntos de Control', fontsize=16, color='black')
    plt.xlabel('Coordenada X', fontsize=14)
    plt.ylabel('Coordenada Y', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.gca().set_facecolor('#f0f0f0')
    plt.gca().set_aspect('equal')
    plt.show()
    return envolvente

###### Punto 4 ######

def u_derivada(t):
    return np.array([0, 1, 2 * t, 3 * t ** 2])

def bezier_derivada1(t, G, MB):
    return G.T @ MB @ u_derivada(t)

def u_derivada2(t):
    return np.array([0, 0, 2, 6*t])

def bezier_derivada2(t, G, MB):
    return G.T @ MB @ u_derivada2(t)



if __name__ == "__main__":
    ##PUNTO 1
    print("Generando curva de Bézier cuadrática")
    # Definimos los puntos de control
    p0 = np.array([0, 0]) #usamos np.array para poder hacer operaciones vectoriales con facilidad
    p1 = np.array([1, 2])
    p2 = np.array([2, 0])
    puntos_de_control = np.array([p0, p1, p2])
    t_values = np.linspace(0, 1, 100) #nos permite generar 100 valores de t asi la curva se grafica de manera mas suave
    # Graficamos la curva de Bézier cuadrática
    graficar_curva_bezier_cuadratica(t_values, puntos_de_control)
    # Despues lo saco para el informe, Explicación de la relación entre f0(t), f1(t) y g(t)
    print("\n### Explicación ###")
    print("La curva g(t) es una combinación de las curvas f0(t) y f1(t).")
    print("f0(t) es una interpolación lineal entre p0 y p1, mientras que f1(t) es una interpolación lineal entre p1 y p2.")
    print("g(t) se construye usando estas dos funciones intermedias, lo que resulta en una curva suave que se ajusta a p0, p1 y p2.")
    print("Para una curva cúbica de Bézier, el proceso sería similar, utilizando tres funciones intermedias f0(t), f1(t) y f2(t), combinadas en dos pasos adicionales.")

    ##PUNTO 2
    print("Generando curva de Bézier cúbica")
    
    # Para hacer la funcion cubica agregamos un nuevo punto de control
    p0 = np.array([0, 0])
    p1 = np.array([1, 3])
    p2 = np.array([3, 3])
    p3 = np.array([4, 0])
    puntos_de_control_2 = np.array([p0, p1, p2, p3])

    # Graficar la curva de Bézier cúbica (inciso A)
    graficar_curva_bezier_cubic(t_values, puntos_de_control_2)

    ## Inciso B: Calcular y graficar los coeficientes de los puntos de control
    coef_c0, coef_c1, coef_c2, coef_c3 = [], [], [], []
    for t in t_values:
        c0, c1, c2, c3 = (1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t**2, t**3
        coef_c0.append(c0)
        coef_c1.append(c1)
        coef_c2.append(c2)
        coef_c3.append(c3)

    graficar_coeficientes_cubicos(t_values, coef_c0, coef_c1, coef_c2, coef_c3)

    # Calcular la suma de los coeficientes para valores específicos de t
    t_values_specific = [0.3, 0.5, 0.8]
    print("Suma de los coeficientes para valores específicos de t:")
    for t in t_values_specific:
        c0, c1, c2, c3 = (1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t**2, t**3
        suma_coeficientes = c0 + c1 + c2 + c3
        print(f"t = {t}: Suma de coeficientes = {suma_coeficientes:.2f}")
    
    ## Inciso C: 3 puntos de control aleatorios
    np.random.seed(42)  # Para reproducibilidad
    puntos_de_control_aleatorios = [np.random.rand(4, 2) * 5 for _ in range(3)]

    # Graficamos cada curva de Bézier cúbica
    for i, puntos_de_control in enumerate(puntos_de_control_aleatorios):
        graficar_curva_bezier_cubic(t_values, puntos_de_control)
    
    ##PUNTO 3
    print("Generando envolvente convexa")
    puntos_convexa = np.array([[0, 0], [1, 2], [2, 0]])
    grafico_envolvente_convexa(puntos_convexa)
    plt.show()

    ##PUNTO 4
    # Definimos la matriz base de Bézier cúbica MB
    # Definir los puntos de control y armamos un vector
    p0 = np.array([0, 0])
    p1 = np.array([1, 3])
    p2 = np.array([3, 3])
    p3 = np.array([4, 0])
   
    puntos_de_control = np.array([p0, p1, p2, p3])

    # Matriz de Beizer
    MB = np.array([
        [1, 0, 0, 0],
        [-3, 3, 0, 0],
        [3, -6, 3, 0],
        [-1, 3, -3, 1]])

    # primer derivada
    t = 0.5
    result = bezier_derivada1(t, puntos_de_control, MB)
    print(f"x'(t) en t={t}: {result}")


    # segunda derivada
    t = 0.5
    result = bezier_derivada2(t, puntos_de_control, MB)
    print(f"x''(t) en t={t}: {result}")
"""
discretizacion.py

Módulo de discretización espacial para el sistema de transporte advectivo.
Contiene la construcción del dominio y los operadores de diferencias finitas.
"""

import numpy as np

def crear_dominio(L, dx):
    """
    Crea el vector de posiciones nodales del dominio espacial [0, L].

    Parámetros:
    L: float -> Longitud del dominio [m].
    dx: float -> Paso espacial [m].

    Retorna:
    x: ndarray -> Vector de posiciones nodales.
    N: int -> Número total de nodos.
    """

    x = np.arange(0, L, dx) # dominio periódico: [0, L), el nodo L = nodo 0
    N = len(x)
    return x, N

def condiciones_iniciales(x, L):
    """
    Evalúa las condiciones iniciales del sistema en todos los nodos.

    Condición inicial dada:
        u(x, 0) = exp(cos(2*pi*x / L))
        V(X, 0) = 0

    Parámetros:
    x: ndarray -> Vector de posiciones nodales.
    L: float -> Longitud del dominio [m].

    Retorna:
    u0: ndarray -> Condición inicial para u.
    v0: ndarray -> Condición inicial para v.
    """

    u0 = np.exp(np.cos(2 * np.pi * x / L))
    v0 = np.zeros(len(x))

    return u0, v0

def derivada_espacial_cfd(u, dx, N):
    """
    Calcula la derivada espacial du/dx usando Diferencias Finitas Centradas
    (CFD) con condiciones de frontera periódicas.

    Esquema CFD:
        du/dx |_i ≈  (u_{i+1} - u_{i-1}) / (2*dx)

    Los nodos de frontera se tratan con índices modulares:
        - vecino izquierdo de i = 0 -> i = N - 1 (oeriodicidad).
        - vecino derecho de i = N - 1 -> i = 0 (periodicidad).

    Parámetros:
    u: ndarray -> Vector de valores de u en todos los nodos.
    dx: float -> paso espacial [m].
    N: int -> Número total de nodos.

    Retorna:
    dudx: ndarray -> Derivada espacial aproximada de u.
    """

    # Paso 00: inicializar vector de derivadas
    dudx = np.zeros(N)

    # Paso 01: índices de vecinos con periodicidad 
    # (np.roll para desplazar el vector)
    u_derecha = np.roll(u, -1) # u_{i+1}: desplazamiento hacia la izquierda
    u_izquierda = np.roll(u, 1) # u_{i-1}: desplazamiento hacia la derecha

    # Paso 02: aplicar esquema CFD en todos los nodos (incluye frontera 
    # por periodicidad)
    dudx = (u_derecha - u_izquierda) / (2.0 * dx)

    return dudx
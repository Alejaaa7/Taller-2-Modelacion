"""
solver_laplace.py

Módulo de solución numérica de la ecuación de Laplace 2D mediante diferencias
finitas centradas (CFD).

Discretización en nodo interior (i, j):
    (T_{i+1,j} - 2T_{i,j} + T_{i-1,j}) / dx^2
  + (T_{i,j+1} - 2T_{i,j} + T_{i,j-1}) / dy^2 = 0

El sistema resultante Ax = b se resuelve con scipy.linalg.sovle.

Convención de indexación:
    - i: dirección x (columnas), i = 0, ..., Nx - 1
    - j: dirección y (filas), j = 0, ..., Ny - 1
    - Nodos interiores: i = 1..Nx-2, j = 1..Ny-2
    - Índice global del nodo (i, j) interior: k = (j-1)*(Nx-2) + (i-1)
"""

import numpy as np
from scipy import linalg

def construir_sistema(Nx, Ny, dx, dy, T_izq, T_der, T_inf, T_sup):
    """
    Construye la matriz A y el vector b del sistema lineal Ax = b para la 
    ecuación de Laplace 2D con condiciones Dirichlet.

    Parámetros:
    Nx, Ny: int -> Número de nodos totales en x y en y.
    dx, dy: float -> Pasos especiales [m].
    T_izq, T_der: float -> Temperatura en borde izquierdo y derecho [K].
    T_inf, T_sup: float -> Temperatura en borde inferior y superior [K].

    Retorna:
    A: ndarray (n_inc, n_inc) -> Matriz de coeficientes.
    b: ndarray (n_inc, ) -> Vector de términos independientes.
    """

    # Paso 00: número de nodos interiores en cada dirección
    nx_int = Nx - 2 # nodos interiores en x
    ny_int = Ny - 2 # nodos interiores en y
    n_inc = nx_int * ny_int # total de incógnitas

    # Paso 01: coeficientes de la discretización CFD
    rx = 1.0 / dx**2
    ry = 1.0 / dy**2
    rc = -2.0 * (rx + ry) # coeficiente central

    # Paso 02: inicializar A y b
    A = np.zeros((n_inc, n_inc))
    b = np.zeros(n_inc)

    # Paso 03: llenar A y b recorriendo nodos interiores
    for j in range(1, Ny - 1):      # j: índice global en y
        for i in range (1, Nx - 1):     # i: índice global en x
            # Índice global del nodo (i, j) en el vector de incógnitas
            k = (j - 1) * nx_int + (i - 1)

            # Coeficiente central
            A[k, k] = rc
            # Vecino derecho (i+1, j), si es frontera derecha, va a b
            if i < Nx - 2:
                k_der = k + 1
                A[k, k_der] = rx
            else:
                b[k] -= rx * T_der
            # Vecino izquierdo (i-1, j), si es frontera izquierda va a b
            if i > 1:
                k_izq = k - 1
                A[k, k_izq] = rx
            else:
                b[k] -= rx * T_izq
            # Vecino superior (i, j+1), si es frontera superior va a b
            if j < Ny - 2:
                k_sup = k + nx_int
                A[k, k_sup] = ry
            else:
                b[k] -= ry * T_sup
            # Vecino inferior (i, j-1), si es frontera inferior va a b
            if j > 1:
                k_inf = k - nx_int
                A[k, k_inf] = ry
            else:
                b[k] -= ry * T_inf
    return A, b

def resolver_laplace(Nx, Ny, dx, dy, T_izq, T_der, T_inf, T_sup):
    """
    Resuelve la ecuación de Laplace 2D y reconstruye el campo completo
    de temperatura incluyendo fronteras.

    Parámetros:
    Nx, Ny: int -> Número de nodos totales en x y en y.
    dx, dy: float -> Pasos espaciales [m].
    T_izq, T_der: float -> Temperatura borde izquierdo y derecho [K].
    T_inf, T_sup: float -> Temperatura borde inferior y superior [K].

    Retorna:
    T: ndarray (Ny, Nx) -> Campo completo de temperatura [K].
    """

    # Paso 00: construir sistema Ax = b
    A, b = construir_sistema(Nx, Ny, dx, dy, T_izq, T_der, T_inf, T_sup)

    # Paso 01: resolver sistema lineal
    T_int = linalg.solve(A, b)

    # Paso 02: reconstruir campo completo con fronteras
    T = np.zeros((Ny, Nx))

    # Llenar nodos interiores desde el vector solución
    nx_int = Nx - 2
    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            k = (j - 1) * nx_int + (i - 1)
            T[j, i] = T_int[k]

    # Paso 03: aplicar condiciones de frontera Dirichlet
    T[:, 0] = T_izq     # borde izquierdo
    T[:, -1] = T_der    # borde derecho
    T[0, :] = T_inf     # borde inferior
    T[-1, :] = T_sup    # borde superior

    return T

def calcular_error(T_num, T_analitica):
    """
    Calcular métricas de error entre solución numérica y analítica.

    Parámetros:
    T_num: ndarray -> Campo numérico [K].
    T_analitica: ndarray -> Campo analítico [K].

    Retorna:
    error_max: float -> Error máximo absoluto [K].
    error_L2: float -> Error norma L2 [K].
    error_rel: float -> Error relativo máximo [K].
    """

    # Se excluyen fronteras para el cálculo de error
    # (la serie de Fourier tiene convergencia lenta en esquinas - Gibbs)
    diff = np.abs(T_num[1:-1, 1:-1] - T_analitica[1:-1, 1:-1])

    error_max = np.max(diff)
    error_L2 = np.sqrt(np.mean(diff**2))
    error_rel = np.max(diff / np.abs(T_analitica[1:-1, 1:-1])) * 100.0

    return error_max, error_L2, error_rel

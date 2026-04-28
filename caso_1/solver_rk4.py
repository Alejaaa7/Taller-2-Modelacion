"""
solver_rk4.py

Módulo de integración temporal mediante Runge-Kutta de cuarto orden (RK4)
para el sistema acoplado de transporte advectivo y relajación.

Sistema de ecuaciones (las dadas):
    du/dt = -c * du/dx
    dc/dt = gamma * (u - v)
"""

import numpy as np
from discretizacion import derivada_espacial_cfd

def pendientes(u, v, c, gamma, dx, N):
    """
    Evalúa las derivadas temporales (pendientes) del sistema acoplado
    en un instante dado. Esta función es el corazón del método de líneas.

    Parámetros:
    u: ndarray -> Vector de u en todos los nodos.
    v: ndarray -> Vector de v en todos los nodos.
    c: float -> Velocidad de transporte [m/s].
    gamma: float -> Tasa de relajación [1/s].
    dx: float -> Paso espacial [m].
    N: int  -> Número total de nodos.

    Retorna:
    dudt: ndarray -> Derivada temporal de u en todos los nodos.
    dvdt: ndarray -> Derivada temporal de v en todos los nodos.
    """

    # Paso 00: calcular derivada espacial de u con CFD periódica
    dudx = derivada_espacial_cfd(u, dx, N)

    # Paso 01: evaluar ec. de transporte advectivo -> du/dt = -c * du/dx
    dudt = -c * dudx

    # Paso 02: evaluar ec. de relajación -> dv/dt = gamma * (u - v)
    dvdt = gamma * (u - v)

    return dudt, dvdt

def rk4_paso(u, v, c, gamma, dx, dt, N):
    """
    Avanza la solución un paso de tiempo dt usando RK4.

    El esquema RK4 calcula 4 estimaciones de la pendiente en el intervalo
    [t^n, t^n + dt] y las promedia de forma ponderada:

        k1 = F(U^n)
        k2 = F(U^n + dt/2 * k1)
        k3 = F(U^n + dt/2 * k2)
        k4 = F(U^n + dt * k3)
        U^{n+1} = U^n  dt/6 * (k1 + 2k2  2k3 + k4)

    Parámetros:
    u: ndarray -> Vector de u al tiempo t^n.
    v: ndarray -> Vector de v al tiempo t^n.
    c: float -> Velocidad de transporte [m/s].
    gamma: float -> Tasa de relajación [1/s].
    dx: float -> Paso espacial [m].
    dt: float -> Paso temporal [s].
    N: int -> Número total de nodos.

    Retorna:
    u_new: ndarray -> Vector de u al tiempo t^{n+1}.
    v_new: ndarray -> Vector de v al tiempo t^{n+1}.
    """

    # Paso 00: calcular k1 en (u^n, v^n)
    k1u, k1v = pendientes(u, v, c, gamma, dx, N)

    # Paso 01: calcular estado intermedio con k1 y evaluar k2
    u2 = u + (dt / 2) * k1u
    v2 = v + (dt / 2) * k1v
    k2u, k2v = pendientes(u2, v2, c, gamma, dx, N)

    # Paso 02: calcular estado intermedio con k2 y evaluar k3
    u3 = u + (dt / 2) * k2u
    v3 = v + (dt / 2) * k2v
    k3u, k3v = pendientes(u3, v3, c, gamma, dx, N)

    # Paso 03: calcular estado al final del intervalo con k3 y evaluar k4
    u4 = u + dt * k3u
    v4 = v + dt * k3v
    k4u, k4v = pendientes(u4, v4, c, gamma, dx, N)

    # Paso 04: combinar las 4 pendientes -> actualizar u y v
    u_new = u + (dt / 6) * (k1u + 2 * k2u + 2 * k3u + k4u)
    v_new = v + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)


    return u_new, v_new

def resolver_sistema(x, N, L, c, gamma, dx, dt, t_final):
    """
    Resuelve el sisema completo de transporte advectivo + relajación desde
    t = 0 hasta t = t_final usando RK4.

    Parámetros:
    x: ndarray -> Vector de posiciones nodales.
    N: int -> Número total de nodos.
    L: float -> Longitud del dominio [m].
    c: float -> Velocidad de transporte [m/s].
    gamma: float -> Tasa de relajación [1/s].
    dx: float -> Paso espacial [m].
    dt: float -> Paso temporal [s].
    t_final: float -> Tiempo final de simulación [s].

    Retorna:
    t: ndarray -> Vector de tiempos.
    u_hist: list[ndarray] -> Historial de u (solo instantes guardados).
    v_hist: list[ndarray] -> Historial de v (solo instantes guardados).
    t_hist: list[float] -> Tiempos correspondientes a u_hist y v_hist.
    """

    from discretizacion import condiciones_iniciales

    # Paso 00: crear vector de tiempo
    t = np.arange(0, t_final + dt, dt)
    Nt = len(t)

    # Paso 01: condiciones iniciales
    u, v = condiciones_iniciales(x, L)

    # Paso 02: definir instantes a gaurdar para graficar (t=0, 12, 30 s)
    t_guardar = [0, 12, 30]
    u_hist = []
    v_hist = []
    t_hist = []

    # Paso 03: guardar condición espacial
    u_hist.append(u.copy())
    v_hist.append(v.copy())
    t_hist.append(0.0)

    # Paso 04: avanzar en el tiempo con RK4
    for n in range(Nt - 1):
        u, v = rk4_paso(u, v, c, gamma, dx, dt, N)

        # Guardar si el tiempo actual coincide (aproximadamente) con un
        # instante de interés
        t_actual = t[n + 1]
        for tg in t_guardar:
            if tg > 0 and abs(t_actual - tg) < dt / 2:
                u_hist.append(u.copy())
                v_hist.append(v.copy())
                t_hist.append(t_actual)

    return t, u_hist, v_hist, t_hist
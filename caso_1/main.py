"""
main.py

Script principal del Caso 1: Sistema acoplado de transporte advectivo y
relajación, resuelto mediante Diferencias Finitas Centradas (CFD) e 
integración temporal Runge-Kutta de cuarto orden (RK4).

Sistema de ecuaciones:
    du/dt = -c  du/dx
    dv/dt = gamma * (u - v)

Para correr:
    python main.py
"""

import numpy as np
import matplotlib

from discretizacion import crear_dominio
from solver_rk4 import resolver_sistema
from graficas import (graficar_perfiles, graficar_efecto_malla, 
                      graficar_efecto_cfl)


# PARTE 1: Parámetros base del problema (los dados en el taller)

L = 5.0         # Longitud del dominio [m]
c = 1.0         # Velocidad de transporte [m/s]
gamma = 1.0     # Tasa de relajación [1/s]
dx = 0.125      # Paso espacial base
dt = 1e-4       # Paso temporal base [s]
tf = 30.0       # Tiempo final [s]

print("—" * 60)
print(" CASO 1  Transporte advectivo + Relajación")
print("—" * 60)

print(f"Parámetros: L = {L} m | c = {c} m/s | gamma = {gamma} 1/s")
print(f"Malla base: dx = {dx} m | dt = {dt} s")
cfl_base = c * dt / dx
print(f"Número de Courant (CFL) base: {cfl_base:.4f}")
print()


# PARTE 2: SOlución con parámetros base

print(">> Ejecutando la simulación con parámetros base")
x, N = crear_dominio(L, dx)
_, u_hist, v_hist, t_hist = resolver_sistema(x, N, L, c, gamma, dx, dt, tf)

print(f"    Instantes guardados: {t_hist}")
graficar_perfiles(x, u_hist, v_hist, t_hist, nombre_figura="caso1_perfiles_base")
print()


# PARTE 3: Experimento 1 - Efecto del tamaño de malla Δx

print(">> Experimento 1: Efecto del tamaño de malla Δx")

dx_lista = [0.25, 0.125, 0.0625, 0.03125] # mallas: gruesa -> fina
resultados_malla = []

for dxi in dx_lista:
    xi, Ni = crear_dominio(L, dxi)
    # Se mantiene dt pequeño para que la estabilidad no interfiera
    dti = 1e-4
    _, u_h, v_h, t_h = resolver_sistema(xi, Ni, L, c, gamma, dxi, dti, tf)

    # Se guarda el perfil en t ≈ 12 (índice 1 de t_hist)
    idx_t12 = 1 if len(t_h) > 1 else 0
    resultados_malla.append({
        "dx": dxi,
        "x": xi,
        "u": u_h[idx_t12],
        "v": v_h[idx_t12],
        "t": t_h[idx_t12]
    })
    print(f"    dx = {dxi:.5f} m | N = {Ni} nodos | completado.")

graficar_efecto_malla(resultados_malla, t_objetivo=12.0, L=L)
print()


# PARTE 4: Experimento 2 - Efecto del número de Courant (CFL = c * Δt/Δx)

print(">> Experimento 2: Efecto del número de Courant CFL")

# Se fija dx base y se varía dt para obtener distintos CFL 
dx_fijo = 0.125
cfl_lista = [0.1, 0.5, 1.5, 2.8]    # CFL = 2.8 debería mostrar inestabilidad

resultados_cfl = []

for cfl_val in cfl_lista:
    dti = cfl_val * dx_fijo / c
    xi, Ni = crear_dominio(L, dx_fijo)

    try:
        _, u_h, v_h, t_h = resolver_sistema(xi, Ni, L, c, gamma, dx_fijo, dti, tf)
        idx_t12 = 1 if len(t_h) > 1 else 0
        u_plot = u_h[idx_t12]
        v_plot = v_h[idx_t12]
        t_plot = t_h[idx_t12]
    except Exception:
        # Si explota, se guarda NaN para visualizarlo
        u_plot = np.full(Ni, np.nan)
        v_plot = np.full(Ni, np.nan)
        t_plot = 12.0

    resultados_cfl.append({
        "cfl": cfl_val,
        "dt":  dti,
        "x":   xi,
        "u":   u_plot,
        "v":   v_plot,
    })
    print(f"    CFL = {cfl_val:.2f} | dt = {dti:.5f} s | completado")

graficar_efecto_cfl(resultados_cfl, t_objetivo=12.0, L=L)
print()
 
print("—" * 60)
print("  Caso 1 finalizado. Figuras generadas:")
print("    - caso1_perfiles_base.png")
print("    - efecto_malla.png")
print("    - efecto_cfl.png")
print("—" * 60)

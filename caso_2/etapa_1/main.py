"""
main.py

Script principal — Caso 2, Etapa 1:
Conducción de calor estacionario en placa plana 2D.

Ecuación gobernante: Laplace  ∇^2 T = 0
Método numérico: Diferencias Finitas Centradas (CFD) 2D.
Validación: Solución analítica por separación de variables

Para correr:
    python main.py

Taller 02 - Modelación Matemática
Universidad Nacional de Colombia
"""

import numpy as np

from dominio        import crear_malla, solucion_analitica
from solver_laplace import resolver_laplace, calcular_error
from graficas import (graficar_campo_temperatura,
                             graficar_comparacion,
                             graficar_curvas_error)



# PARTE 1: Parámetros del problema

a   = 1.0 # longitud en x [m]
b   = 1.0 # longitud en y [m]
T_l = 300.0 # temperatura baja [K]  — bordes izq, inf, sup
T_h = 450.0 # temperatura alta [K]  — borde derecho

print("—" * 60)
print("  CASO 2 — Etapa 1: Conducción Estacionaria 2D")
print("—" * 60)
print(f"  Dominio: {a} m x {b} m")
print(f"  T_alta (borde der) = {T_h} K  |  T_baja (otros) = {T_l} K")
print()


# PARTE 2: Solución base con malla dx = dy = 0.05 m

dx = 0.05
dy = 0.05

print(f">> Solución base con dx = dy = {dx} m")
x, y, XX, YY, Nx, Ny = crear_malla(a, b, dx, dy)
print(f"   Malla: {Nx} x {Ny} nodos  ({(Nx-2)*(Ny-2)} incógnitas interiores)")

# Paso 00: solución numérica
T_num = resolver_laplace(Nx, Ny, dx, dy,
                         T_izq=T_l, T_der=T_h,
                         T_inf=T_l, T_sup=T_l)

# Paso 01: solución analítica
T_anal = solucion_analitica(XX, YY, a, b, T_l, T_h, N_terms=101)

# Paso 02: errores
err_max, err_L2, err_rel = calcular_error(T_num, T_anal)
print(f"   Error máximo  = {err_max:.4f} K")
print(f"   Error L2      = {err_L2:.4f} K")
print(f"   Error relativo = {err_rel:.4f} %")
print()

# Paso 03: graficar comparación
graficar_comparacion(XX, YY, T_num, T_anal,
                     nombre_figura="caso2_etapa1_comparacion")

# Paso 04: graficar solo numérica (réplica de Figura 2 del taller)
graficar_campo_temperatura(XX, YY, T_num,
    titulo=f"Campo de temperatura — CFD ($\\Delta x = {dx}$ m)",
    nombre_figura="caso2_etapa1_campo_numerico",
    T_min=T_l, T_max=T_h)

# Paso 05: graficar analítica
graficar_campo_temperatura(XX, YY, T_anal,
    titulo="Campo de temperatura — Solución analítica",
    nombre_figura="caso2_etapa1_campo_analitico",
    T_min=T_l, T_max=T_h)



# PARTE 3: Experimento — efecto del tamaño de malla y curvas de error

print(">> Experimento: curvas de error vs tamaño de malla")

dx_lista    = [0.2, 0.1, 0.05, 0.025, 0.0125]
errores_max = []
errores_L2  = []

for dxi in dx_lista:
    xi, yi, XXi, YYi, Nxi, Nyi = crear_malla(a, b, dxi, dxi)

    T_numi  = resolver_laplace(Nxi, Nyi, dxi, dxi,
                               T_izq=T_l, T_der=T_h,
                               T_inf=T_l, T_sup=T_l)
    T_anali = solucion_analitica(XXi, YYi, a, b, T_l, T_h, N_terms=101)

    emax, eL2, _ = calcular_error(T_numi, T_anali)
    errores_max.append(emax)
    errores_L2.append(eL2)

    print(f"   dx = {dxi:.4f} m  |  N = {Nxi}x{Nyi}  |  "
          f"err_max = {emax:.5f} K  |  err_L2 = {eL2:.5f} K")

graficar_curvas_error(dx_lista, errores_max, errores_L2,
                      nombre_figura="caso2_etapa1_convergencia")

print()
print("—" * 60)
print("  Etapa 1 finalizada. Figuras generadas:")
print("    - caso2_etapa1_comparacion.png")
print("    - caso2_etapa1_campo_numerico.png")
print("    - caso2_etapa1_campo_analitico.png")
print("    - caso2_etapa1_convergencia.png")
print("—" * 60)
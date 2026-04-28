"""
graficas.py

Módulo de visualización para la Etapa 2: transitorio con fuente interna.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors


# para inferno pero bonito
colors = plt.cm.inferno(np.linspace(0.07, 1, 256))
new_inferno = mcolors.ListedColormap(colors)


def graficar_snapshots(XX, YY, T_snapshots, t_hist, omega, nombre_figura):
    """
    Grilla de mapas de temperatura en diferentes instantes de tiempo.

    Parámetros:
    
    XX, YY: ndarray -> Mallas de coordenadas.
    T_snapshots: list[ndarray] -> Campos T en cada instante.
    t_hist: list[float -> Tiempos correspondientes.
    omega: float -> Frecuencia usada [rad/s].
    nombre_figura: str -> Nombre para guardar.
    """
    n = len(T_snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=True)

    T_min = min(Ti.min() for Ti in T_snapshots)
    T_max = max(Ti.max() for Ti in T_snapshots)

    for idx, (ax, T, t) in enumerate(zip(axes, T_snapshots, t_hist)):
        cf = ax.contourf(XX, YY, T, levels=40, cmap=new_inferno,
                         vmin=T_min, vmax=T_max)
        cs = ax.contour(XX, YY, T, levels=6, colors="white",
                        linewidths=0.6, alpha=0.6)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
        ax.set_title(f"$t = {t:.2f}$ s", fontsize=10)
        ax.set_xlabel("x [m]", fontsize=9)
        if idx == 0:
            ax.set_ylabel("y [m]", fontsize=9)
        ax.set_aspect("equal")
        fig.colorbar(cf, ax=ax, shrink=0.85).set_label("T [°C]", fontsize=8)

    plt.suptitle(f"Evolución del campo de temperatura — $\\omega = {omega:.4f}$ rad/s",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{nombre_figura}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  >> Figura guardada: {nombre_figura}.png")

def graficar_snapshots_saturado(XX, YY, T_snapshots, t_hist, omega,
                                 T_sat_min=27.0, T_sat_max=180.0,
                                 nombre_figura="snapshots_saturado"):
    """
    Igual que graficar_snapshots pero con escala de color fija entre 
    T_sat_min y T_sat_max, permitiendo visualizar la dinámica del borde 
    Dirichlet sin que la fuente domine la escala.
    """
    n = len(T_snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=True)

    for idx, (ax, T, t) in enumerate(zip(axes, T_snapshots, t_hist)):

        # se crean niveles manualmente para que sean exactos entre min y max
        niveles = np.linspace(T_sat_min, T_sat_max, 40)

        cf = ax.contourf(XX, YY, T, levels=niveles, cmap=new_inferno,
                         extend="both") # apra manejar los valores fuera del rango
        cs = ax.contour(XX, YY, levels=np.linspace(T_sat_min, T_sat_max, 6),
                        colors="white", linewidths=0.6, alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
        ax.set_title(f"$t = {t:.2f}$ s", fontsize=10)
        ax.set_xlabel("x [m]", fontsize=9)
        if idx == 0:
            ax.set_ylabel("y [m]", fontsize=9)
        ax.set_aspect("equal")
        fig.colorbar(cf, ax=ax, shrink=0.85).set_label("T [°C]", fontsize=8)

    plt.suptitle(
        f"Campo de temperatura (escala saturada [{T_sat_min:.0f}, {T_sat_max:.0f}]°C)"
        f" — $\\omega = {omega:.4f}$ rad/s",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{nombre_figura}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  >> Figura guardada: {nombre_figura}.png")


def graficar_sensores(t_vec, T_sensores, sensores_labels, omega,
                      T_min_ref=27, T_max_ref=50, nombre_figura="sensores"):
    """
    Evolución temporal de la temperatura en los sensores predefinidos.
    Usa dos subplots: bordes Neumann (escala acotada) e interiores.

    Parámetros:
    
    t_vec: ndarray -> Vector de tiempos [s].
    T_sensores: ndarray -> (n_sens, Nt) temperaturas en sensores.
    sensores_labels: list[str] -> Etiquetas de cada sensor.
    omega: float -> Frecuencia usada [rad/s].
    T_min_ref: float -> Límite inferior de temperatura [°C].
    T_max_ref: float -> Límite superior de temperatura [°C].
    nombre_figura: str -> Nombre para guardar.
    """

    estilos = ["-", "-", "-."]
    colores = ["#FF0062", "#8800FF", "#8FF522", "#FF005D", "#9154ca"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot izquierdo: sensores en bordes Neumann (primeros 3)
    ax = axes[0]
    for s in range(3):

        ax.plot(t_vec, T_sensores[s], linestyle=estilos[s], color=colores[s], 
                linewidth=1.8, label=sensores_labels[s])
    ax.axhline(T_max_ref, color="red",  linestyle="--",
               linewidth=1.2, alpha=0.8, label=f"Límite sup. {T_max_ref}°C")
    ax.axhline(T_min_ref, color="navy", linestyle="--",
               linewidth=1.2, alpha=0.8, label=f"Límite inf. {T_min_ref}°C")
    ax.axhspan(T_min_ref, T_max_ref, alpha=0.1, color="#9ddbf4",
               label="Rango permitido")
    ax.set_xlabel("t [s]", fontsize=11)
    ax.set_ylabel("T [°C]", fontsize=11)
    ax.set_title("Sensores en bordes Neumann", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    # Subplot derecho: sensores interiores (últimos 2)
    ax = axes[1]
    for s in range(3, len(sensores_labels)):
        ax.plot(t_vec, T_sensores[s], "-", color=colores[s],
                linewidth=1.8, label=sensores_labels[s])
    ax.set_xlabel("t [s]", fontsize=11)
    ax.set_ylabel("T [°C]", fontsize=11)
    ax.set_title("Sensores en nodos interiores", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

    plt.suptitle(f"Evolución temporal en sensores — $\\omega = {omega:.4f}$ rad/s",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{nombre_figura}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  >> Figura guardada: {nombre_figura}.png")


def graficar_prom_neumann(t_vec, T_prom_neu, omega,
                           T_min_ref=27, T_max_ref=50,
                           nombre_figura="prom_neumann"):
    """
    Evolución temporal de la temperatura promedio en los bordes Neumann.

    Parámetros:
    
    t_vec: ndarray -> Vector de tiempos [s].
    T_prom_neu: ndarray -> Temperatura promedio en bordes Neumann [°C].
    omega: float -> Frecuencia usada [rad/s].
    nombre_figura: str -> Nombre para guardar.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(t_vec, T_prom_neu, "#007f9f", linewidth=2, label="$\\bar{T}$ bordes Neumann")
    ax.axhline(T_max_ref, color="#ff005d",  linestyle="--",
               linewidth=1.5, label=f"Límite sup. {T_max_ref}°C")
    ax.axhline(T_min_ref, color="navy", linestyle="--",
               linewidth=1.5, label=f"Límite inf. {T_min_ref}°C")
    ax.axhspan(T_min_ref, T_max_ref, alpha=0.1, color="#9ddbf4",
               label="Rango permitido")

    T_max_obs = T_prom_neu.max()
    T_min_obs = T_prom_neu.min()
    cumple    = (T_min_obs >= T_min_ref) and (T_max_obs <= T_max_ref)

    ax.set_title(
        f"Temperatura promedio en bordes Neumann — $\\omega = {omega:.4f}$ rad/s\n"
        f"T_max = {T_max_obs:.2f}°C  |  T_min = {T_min_obs:.2f}°C  |  "
        f"Condición: {'✓ CUMPLE' if cumple else '✗ NO CUMPLE'}",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlabel("t [s]", fontsize=11)
    ax.set_ylabel("$\\bar{T}$ [°C]", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{nombre_figura}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  >> Figura guardada: {nombre_figura}.png")
    return cumple, T_min_obs, T_max_obs


def graficar_efecto_alpha(resultados_alpha, nombre_figura="efecto_alpha"):
    """
    Compara la temperatura promedio Neumann para distintos valores de α.

    Parámetros:
    
    resultados_alpha: list[dict] -> keys: alpha, t_vec, T_prom.
    nombre_figura: str -> Nombre para guardar.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colores = ["#00ffff", "#fc5a50", "#7fff00", "#c20078"]

    for idx, res in enumerate(resultados_alpha):
        ax.plot(res["t_vec"], res["T_prom"], "-",
                color=colores[idx % len(colores)],
                linewidth=1.8,
                label=f"$\\alpha = {res['alpha']:.2e}$ m²/s")

    ax.axhline(50, color="red",  linestyle="--", linewidth=1.2,
               alpha=0.7, label="Límite sup. 50°C")
    ax.axhline(27, color="blue", linestyle="--", linewidth=1.2,
               alpha=0.7, label="Límite inf. 27°C")
    ax.axhspan(27, 50, alpha=0.07, color="#79bfdb", label="Rango permitido")

    ax.set_xlabel("t [s]", fontsize=11)
    ax.set_ylabel("$\\bar{T}$ bordes Neumann [°C]", fontsize=11)
    ax.set_title("Efecto de la difusividad térmica $\\alpha$ sobre el transitorio",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{nombre_figura}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  >> Figura guardada: {nombre_figura}.png")

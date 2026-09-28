"""
Simulation de trafic routier : modèle de Nagel-Schreckenberg.

Automate cellulaire stochastique sur une route circulaire (conditions
périodiques), vectorisé avec NumPy, avec une zone de travaux à vitesse
limitée.

  Phase 1 : animation en temps réel (route, diagramme espace-temps,
            vitesse moyenne, débit). Fermer la fenêtre pour passer à la suite.
  Phase 2 : analyse statistique (diagramme fondamental, instabilité,
            effet du freinage aléatoire).
  Phase 3 : comparaison de trois régimes sur des diagrammes espace-temps
            (fluide, bouchons fantômes, congestion).

Projet M1 Physique, CY Cergy Paris Université (février 2026).

Utilisation :
    python trafic.py             # animation interactive, puis statistiques et comparaison
    python trafic.py --sauver    # enregistre le GIF et les figures dans figures/
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

SAUVER = "--sauver" in sys.argv

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

L = 400         # Longueur de la route (bords périodiques)
DENSITE = 0.3   # Fraction des cases occupées par des voitures
V_MAX = 5       # Vitesse maximale autorisée (cases par pas de temps)
P_FREIN = 0.3   # Probabilité de freinage aléatoire (facteur humain)
DT = 150        # Nombre de pas gardés en mémoire pour le diagramme espace-temps

# Zone de travaux : vitesse limitée entre deux cases
DEBUT_TRAVAUX = 280
FIN_TRAVAUX = 300
V_TRAVAUX = 2

# Variables globales
position_repere = -1  # Position de la voiture "repère" (-1 = pas encore définie)
matrice_temps = np.full((DT, L), -1)  # Historique du trafic (diagramme espace-temps)


def creer_route(L, densite):
    """Crée la route initiale : -1 = case vide, sinon vitesse de la voiture."""
    global position_repere

    route = np.full(L, -1, dtype=int)
    n_voitures = int(L * densite)

    # On place les voitures au hasard, avec une vitesse aléatoire
    indices = np.random.choice(range(L), n_voitures, replace=False)
    route[indices] = np.random.randint(0, V_MAX + 1, size=n_voitures)

    # La première voiture tirée sert de repère (tracée en noir)
    if n_voitures > 0:
        position_repere = indices[0]

    return route


# ---------------------------------------------------------------------------
# Règles de Nagel-Schreckenberg
# ---------------------------------------------------------------------------

def next_step(route, pos_repere):
    """Calcule l'état de la route au pas de temps suivant (t+1)."""

    # Positions et vitesses des voitures
    pos = np.where(route > -1)[0]
    if len(pos) == 0:
        return route, 0, 0, pos_repere
    v = route[pos]

    # Distance libre devant chaque voiture : np.roll(pos, -1) donne la
    # voiture suivante, le modulo gère la route circulaire, le -1 compte
    # uniquement les cases vides
    voiture_suivante = np.roll(pos, -1)
    distance = (voiture_suivante - pos) % L - 1

    # 1. Accélération : +1 sans dépasser V_MAX
    v = np.minimum(v + 1, V_MAX)

    # Zone de travaux : vitesse limitée à V_TRAVAUX
    in_zone = (pos >= DEBUT_TRAVAUX) & (pos <= FIN_TRAVAUX)
    v[in_zone] = np.minimum(v[in_zone], V_TRAVAUX)

    # 2. Sécurité : on ne dépasse pas la distance libre devant soi
    v = np.minimum(v, distance)

    # 3. Freinage aléatoire (facteur humain), seulement si on roule
    r = np.random.rand(len(v))
    freiner = (r < P_FREIN) & (v > 0)
    v = np.maximum(v - freiner, 0)

    # 4. Déplacement
    nouvelle_route = np.full(L, -1, dtype=int)
    nouvelles_pos = (pos + v) % L
    nouvelle_route[nouvelles_pos] = v

    # Suivi de la voiture repère
    idx_repere_local = np.where(pos == pos_repere)[0]
    if len(idx_repere_local) > 0:
        nouv_pos_repere = nouvelles_pos[idx_repere_local[0]]
    else:
        nouv_pos_repere = pos_repere

    # Statistiques
    v_moy = np.mean(v)
    flux = DENSITE * v_moy  # débit J = densité x vitesse moyenne

    return nouvelle_route, v_moy, flux, nouv_pos_repere


# ---------------------------------------------------------------------------
# Phase 1 : animation
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(12, 10))
fig.canvas.manager.set_window_title("Phase 1 : Observation (fermez pour voir les stats)")

# Grille 3 lignes x 2 colonnes, la ligne du milieu est deux fois plus haute
gs = fig.add_gridspec(3, 2, height_ratios=[0.5, 2, 1], hspace=0.4)
ax_route = fig.add_subplot(gs[0, :])
ax_spacetime = fig.add_subplot(gs[1, :])
ax_vitesse = fig.add_subplot(gs[2, 0])
ax_flux = fig.add_subplot(gs[2, 1])

# Couleurs : blanc (vide), rouge (v=0) ... bleu (v=5), noir (voiture repère)
colors = ["white", "red", "orange", "yellow", "#ADFF2F", "green", "blue", "black"]
bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 10.5]
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

route = creer_route(L, DENSITE)

# Route en temps réel
img_route = ax_route.imshow(route[np.newaxis, :], aspect="auto", cmap=cmap, norm=norm)
ax_route.axvspan(DEBUT_TRAVAUX, FIN_TRAVAUX, color="gray", alpha=0.7)
ax_route.set_title(f"Simulation temps réel (zone grise : vitesse limitée à {V_TRAVAUX})")
ax_route.set_yticks([])
ax_route.set_xticks([])

# Diagramme espace-temps
img_spacetime = ax_spacetime.imshow(matrice_temps, aspect="auto", cmap=cmap, norm=norm, origin="upper")
ax_spacetime.set_title("Diagramme espace-temps (en noir : voiture repère)")
ax_spacetime.set_ylabel("Temps")
ax_spacetime.set_xlabel("Position")
ax_spacetime.axvspan(DEBUT_TRAVAUX, FIN_TRAVAUX, color="gray", alpha=0.7)

# Courbes de vitesse et de débit
x_data, y_vitesse, y_flux = [], [], []

line_v, = ax_vitesse.plot([], [], "b-", lw=2)
ax_vitesse.set_xlim(0, 300)
ax_vitesse.set_ylim(0, V_MAX)
ax_vitesse.set_title("Vitesse moyenne")
ax_vitesse.grid(True)

line_f, = ax_flux.plot([], [], "m-", lw=2)
ax_flux.set_xlim(0, 300)
ax_flux.set_ylim(0, 1.0)
ax_flux.set_title("Débit (flux)")
ax_flux.grid(True)


def animate(frame):
    global route, position_repere, matrice_temps

    # 1. Un pas de simulation
    route, v, j, position_repere = next_step(route, position_repere)

    # 2. Diagramme espace-temps : on décale l'historique d'une ligne vers le
    #    haut et on ajoute l'état actuel en bas (voiture repère forcée en noir)
    matrice_temps = np.roll(matrice_temps, -1, axis=0)
    ligne_actuelle = route.copy()
    if position_repere != -1:
        ligne_actuelle[position_repere] = 10
    matrice_temps[-1, :] = ligne_actuelle

    # 3. Courbes du bas (défilement au-delà de 300 pas)
    x_data.append(frame)
    y_vitesse.append(v)
    y_flux.append(j)
    if frame > 300:
        ax_vitesse.set_xlim(frame - 300, frame)
        ax_flux.set_xlim(frame - 300, frame)

    # 4. Mise à jour de l'affichage
    img_route.set_data(ligne_actuelle[np.newaxis, :])
    img_spacetime.set_data(matrice_temps)
    line_v.set_data(x_data, y_vitesse)
    line_f.set_data(x_data, y_flux)

    return img_route, img_spacetime, line_v, line_f


if SAUVER:
    os.makedirs("figures", exist_ok=True)
    # On part de l'état initial aléatoire : le diagramme espace-temps se
    # remplit et on voit les bouchons se former, puis le régime établi.
    # Une image tous les 3 pas pour garder un GIF raisonnable.
    def trois_pas(k):
        animate(3 * k)
        animate(3 * k + 1)
        return animate(3 * k + 2)

    ani = animation.FuncAnimation(fig, trois_pas, frames=150, blit=False)
    ani.save("figures/trafic.gif", writer=animation.PillowWriter(fps=10), dpi=45)
    plt.close(fig)
    print("Animation enregistrée : figures/trafic.gif")
else:
    ani = animation.FuncAnimation(fig, animate, interval=10, blit=False, cache_frame_data=False)
    plt.show()  # Le programme attend ici que la fenêtre soit fermée


# ---------------------------------------------------------------------------
# Phase 2 : analyse physique
# ---------------------------------------------------------------------------

def calcul_stat(L_sim, densite_sim, P_FREIN_sim, steps=300, travaux=False, historique=None):
    """
    Même dynamique que next_step, sans affichage, pour mesurer des moyennes.
    Renvoie la vitesse moyenne, le flux et l'écart-type des vitesses,
    mesurés sur la seconde moitié de la simulation (régime stationnaire).
    Si une liste `historique` est fournie, l'état de la route y est ajouté
    à chaque pas (pour tracer un diagramme espace-temps).
    """
    route = np.full(L_sim, -1, dtype=int)
    n = int(L_sim * densite_sim)
    pos = np.random.choice(range(L_sim), n, replace=False)
    route[pos] = np.random.randint(0, V_MAX + 1, n)

    v_moy_samples = []
    v_std_samples = []

    for i in range(steps):
        pos = np.where(route > -1)[0]
        if len(pos) == 0:
            break
        v = route[pos]
        dist = (np.roll(pos, -1) - pos) % L_sim - 1
        v = np.minimum(v + 1, V_MAX)

        if travaux:
            in_zone = (pos >= DEBUT_TRAVAUX) & (pos <= FIN_TRAVAUX)
            v[in_zone] = np.minimum(v[in_zone], V_TRAVAUX)

        v = np.minimum(v, dist)
        freiner = (np.random.rand(len(v)) < P_FREIN_sim) & (v > 0)
        v = np.maximum(v - freiner, 0)

        new_route = np.full(L_sim, -1, dtype=int)
        new_route[(pos + v) % L_sim] = v
        route = new_route
        if historique is not None:
            historique.append(route.copy())

        # On attend la moitié de la simulation avant de mesurer
        if i > steps // 2:
            v_moy_samples.append(np.mean(v))
            v_std_samples.append(np.std(v))

    v_moy_final = np.mean(v_moy_samples)
    flux_final = densite_sim * v_moy_final
    std_final = np.mean(v_std_samples)
    return v_moy_final, flux_final, std_final


np.random.seed(0)  # graine fixe : courbes reproductibles

# Expérience A : balayage en densité
densities = np.linspace(0.05, 0.95, 20)
flux_res, vitesse_res, std_res = [], [], []
for d in densities:
    v, f, s = calcul_stat(L, d, P_FREIN, travaux=False)
    flux_res.append(f)
    vitesse_res.append(v)
    std_res.append(s)

# Expérience B : effet de la probabilité de freinage, à densité fixée
# (0.15, proche de la densité critique)
probas_frein = np.linspace(0, 1.0, 20)
densite_fixe = 0.15
vitesse_vs_proba = []
for p in probas_frein:
    v, _, _ = calcul_stat(L, densite_fixe, p, travaux=False)
    vitesse_vs_proba.append(v)

rho_c = densities[int(np.argmax(flux_res))]
print(f"Débit maximal J = {max(flux_res):.3f} atteint pour une densité ρ ≈ {rho_c:.2f}")

fig2, axs = plt.subplots(2, 2, figsize=(14, 10))
fig2.canvas.manager.set_window_title("Phase 2 : Analyse physique")
plt.subplots_adjust(hspace=0.3, wspace=0.3)

axs[0, 0].plot(densities, flux_res, "o-", color="purple")
axs[0, 0].set_title("1. Diagramme fondamental (flux)")
axs[0, 0].set_xlabel(r"Densité ($\rho$)")
axs[0, 0].set_ylabel(r"Flux ($J$)")
axs[0, 0].grid(True)

axs[0, 1].plot(densities, vitesse_res, "o-", color="blue")
axs[0, 1].set_title("2. Vitesse moyenne vs densité")
axs[0, 1].set_xlabel(r"Densité ($\rho$)")
axs[0, 1].set_ylabel("Vitesse")
axs[0, 1].grid(True)

axs[1, 0].plot(densities, std_res, "o-", color="red", lw=2)
axs[1, 0].set_title(r"3. Instabilité ($\sigma_v$)")
axs[1, 0].set_xlabel(r"Densité ($\rho$)")
axs[1, 0].set_ylabel("Écart-type des vitesses")
axs[1, 0].grid(True)

axs[1, 1].plot(probas_frein, vitesse_vs_proba, "s-", color="green", lw=2)
axs[1, 1].set_title("4. Impact du facteur humain")
axs[1, 1].set_xlabel(r"Probabilité de freinage ($P_{frein}$)")
axs[1, 1].set_ylabel("Vitesse moyenne")
axs[1, 1].grid(True)
axs[1, 1].text(0.1, 0.2, f"Densité fixée à {densite_fixe}", transform=axs[1, 1].transAxes,
               bbox=dict(facecolor="white", alpha=0.8))


# ---------------------------------------------------------------------------
# Phase 3 : comparaison de trois régimes (sans zone de travaux)
# ---------------------------------------------------------------------------

# Même densité avec et sans freinage aléatoire, puis une densité élevée.
# On ne garde que les 300 derniers pas (régime établi).
regimes = [
    (0.12, 0.0, "Fluide : conduite parfaite"),
    (0.12, P_FREIN, "Bouchons fantômes : même densité, freinage aléatoire"),
    (0.35, P_FREIN, "Congestion : densité élevée"),
]
np.random.seed(1)  # graine fixe : figure reproductible

fig3, axs3 = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
fig3.canvas.manager.set_window_title("Phase 3 : Comparaison des régimes")
for ax, (d, p, titre) in zip(axs3, regimes):
    historique = []
    v, _, _ = calcul_stat(L, d, p, steps=600, historique=historique)
    ax.imshow(np.array(historique[-300:]), aspect="auto", cmap=cmap, norm=norm)
    ax.set_title(f"{titre}\n" + rf"$\rho$ = {d}, $P_{{frein}}$ = {p}, vitesse moyenne = {v:.2f}", fontsize=10)
    ax.set_xlabel("Position")
axs3[0].set_ylabel("Temps")
fig3.tight_layout()

if SAUVER:
    fig2.savefig("figures/analyse.png", dpi=80)
    fig3.savefig("figures/comparaison.png", dpi=70)
    print("Figures enregistrées : figures/analyse.png, figures/comparaison.png")
else:
    plt.show()

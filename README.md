# Snake Rainbow DQN

Projet Python modulaire pour Snake combinant un agent Rainbow DQN (PyTorch, Gymnasium) et un planner hamiltonien pour obtenir des full clears fiables.

Le projet contient maintenant deux voies separees :

- `Rainbow DQN` : apprentissage par reinforcement learning, lisible et stable.
- `Planner` : IA symbolique de planification pour finir la grille de facon fiable.

## Architecture

```text
snake_rainbow/
  config.py
  train.py
  evaluate.py
  requirements.txt
  README.md
  env/
    snake_env.py
  agent/
    network.py
    planner.py
    replay_buffer.py
    rainbow_agent.py
    utils.py
  utils/
    metrics.py
    plotting.py
    seed.py
```

## Roles des fichiers

- `config.py` : hyperparametres, rewards, chemins, tailles d'observation et support C51.
- `train.py` : boucle d'entrainement Rainbow, checkpoints, diagnostics et evaluations deterministes.
- `evaluate.py` : evaluation visuelle/headless en mode `rainbow`, `planner` ou `hybrid`.
- `env/snake_env.py` : environnement Snake compatible Gymnasium et rendu Pygame.
- `agent/network.py` : reseau Rainbow avec Dueling, NoisyLinear et tete C51.
- `agent/replay_buffer.py` : Prioritized Experience Replay et n-step returns.
- `agent/rainbow_agent.py` : Double DQN distributionnel, target network, save/load.
- `agent/planner.py` : IA de planification basee sur cycle hamiltonien + BFS de securite.
- `agent/utils.py` : projection C51, beta PER et hard update.
- `utils/metrics.py` : logs CSV des episodes.
- `utils/plotting.py` : courbes d'entrainement.
- `utils/seed.py` : seeds Python, NumPy, PyTorch et Gymnasium.

## Choix importants

Le mode d'observation par defaut est revenu a `rich`. C'est la base la plus stable du projet : 30 features compactes avec dangers, direction, nourriture relative, distances, espace atteignable et accessibilite de la queue.

Le mode `grid` reste disponible, mais il est experimental ici. Il donne plus d'information brute, mais il a montre des runs plus instables avec ce reseau MLP.

Le support C51 est maintenant `[-50, 300]` au lieu de `[-20, 20]`. Avec `+10` par pomme, l'ancien support ecrasait trop vite les gros retours et rendait les bons episodes presque indistinguables pour la tete distributionnelle.

Le shaping de securite est desactive par defaut. Il reste dans l'environnement, mais il avait ajoute du bruit aux derniers runs. La reward de base reste simple :

- `+10` quand le serpent mange.
- `+10` quand la grille est terminee.
- `-20` sur collision.
- `-20` sur timeout sans nourriture.
- `-0.01` par pas.
- `+0.02` si le serpent se rapproche de la nourriture, `-0.02` s'il s'en eloigne.

## Installation

```powershell
git clone https://github.com/Kawamoux/SnakeRainbowDQN.git
cd SnakeRainbowDQN
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
```

## Entrainement Rainbow DQN

Run propre depuis zero, recommande pour retrouver de bons resultats RL :

```powershell
.\.venv\Scripts\python.exe .\train.py --episodes 3000 --observation-mode rich --diagnostics --eval-interval 25 --eval-episodes 50
```

Reprendre exactement un entrainement la ou il s'est arrete :

```powershell
.\.venv\Scripts\python.exe .\train.py --checkpoint .\artifacts\checkpoints\latest_checkpoint.pt --episodes 5000 --diagnostics --eval-interval 25 --eval-episodes 50
```

Important : avec `--checkpoint`, `--episodes 5000` veut dire "aller jusqu'a l'episode 5000 au total", pas "ajouter 5000 episodes".

Warm-start depuis un bon modele, mais en repartant avec optimizer/replay buffer propres :

```powershell
.\.venv\Scripts\python.exe .\train.py --episodes 3000 --observation-mode rich --warm-start-checkpoint .\artifacts\checkpoints\best_model_rich_before_grid.pt --diagnostics --eval-interval 25 --eval-episodes 50
```

## Evaluation

Evaluer le modele Rainbow sauvegarde :

```powershell
.\.venv\Scripts\python.exe .\evaluate.py --controller rainbow --checkpoint .\artifacts\checkpoints\best_model.pt --episodes 10
```

Evaluer Rainbow sans affichage :

```powershell
.\.venv\Scripts\python.exe .\evaluate.py --controller rainbow --checkpoint .\artifacts\checkpoints\best_model.pt --episodes 100 --headless
```

Finir la grille avec l'IA planificatrice, sans checkpoint :

```powershell
.\.venv\Scripts\python.exe .\evaluate.py --controller planner --episodes 10
```

Test headless du planner :

```powershell
.\.venv\Scripts\python.exe .\evaluate.py --controller planner --episodes 100 --headless
```

Mode hybride conservateur : meme trajectoire que le planner par defaut, avec possibilite de tester des propositions Rainbow via `--planner-shortcuts` :

```powershell
.\.venv\Scripts\python.exe .\evaluate.py --controller hybrid --checkpoint .\artifacts\checkpoints\best_model.pt --episodes 10
```

Les raccourcis du planner existent avec `--planner-shortcuts`, mais ils sont experimentaux. Pour maximiser les full clears, garde le mode `planner` ou `hybrid` sans raccourcis.

## Resultat de verification

Test local effectue apres ajout du planner :

```text
Full clears: 10/10
Mean score: 141.00
Min score: 141
Max score: 141
```

Sur une grille `12x12`, le score maximum est `141`, car le serpent commence avec une longueur de `3` et il reste `144 - 3 = 141` pommes a manger.

## Limites connues

- Le Rainbow DQN reste du reinforcement learning : il peut devenir tres bon, mais il ne garantit pas mathematiquement le full clear a chaque episode.
- Le planner est volontairement conservateur. Il finit la grille, mais il prend souvent plusieurs milliers de steps.
- Le mode `hybrid` reste volontairement conservateur par defaut. Les shortcuts sont plus rapides, mais moins fiables.
- L'entrainement n'est pas vectorise, donc les longs runs restent plus lents qu'un setup parallele.
- Le warning `pkg_resources` vient de Pygame/setuptools et n'empeche pas le projet de tourner.

## Ameliorations futures

- Ajouter un script d'imitation learning pour entrainer un reseau a copier le planner.
- Ajouter un environnement vectorise pour accelerer Rainbow.
- Tester un reseau CNN pour le mode `grid`.
- Sauvegarder automatiquement les runs dans des dossiers nommes pour eviter d'ecraser les bons checkpoints.
- Ajouter TensorBoard ou Weights & Biases pour suivre les evaluations.

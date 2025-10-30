
from collections import deque
import heapq
from PIL import Image, ImageDraw
import numpy as np
import time

class MazeAStar:
    def __init__(self, image_path):
        """Charger et analyser l'image du labyrinthe"""
        print("📂 Chargement de l'image...")
        self.image_path = image_path
        self.original_image = Image.open(image_path)
        self.maze = self._process_image()
        self.height = len(self.maze)
        self.width = len(self.maze[0])
        print(f"✅ Labyrinthe chargé: {self.width}x{self.height} pixels")

        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def _process_image(self):
        """Convertir l'image en matrice binaire"""
        img_gray = self.original_image.convert('L')
        img_array = np.array(img_gray)
        maze = (img_array < 128).astype(int)
        return maze

    def find_first_open_cell(self):
        """Trouver la première cellule ouverte dans le labyrinthe"""
        print("\n🔍 Recherche d'une cellule de départ...")

        corners = [
            (10, 10),
            (self.width - 10, 10),
            (10, self.height - 10),
            (self.width - 10, self.height - 10)
        ]

        for x, y in corners:
            if 0 <= x < self.width and 0 <= y < self.height:
                if self.maze[y][x] == 0:
                    print(f"✅ Cellule trouvée dans un coin: ({x}, {y})")
                    return (x, y)

        for y in range(5, self.height, 5):
            for x in range(5, self.width, 5):
                if self.maze[y][x] == 0:
                    print(f"✅ Première cellule libre trouvée: ({x}, {y})")
                    return (x, y)

        raise Exception("❌ Aucune cellule libre trouvée dans le labyrinthe!")

    def find_center(self):
        """Trouver le centre du labyrinthe"""
        print("\n🎯 Recherche du centre...")

        center_x = self.width // 2
        center_y = self.height // 2
        max_radius = max(self.width, self.height) // 2

        for radius in range(0, max_radius, 2):
            for angle in range(0, 360, 15):
                rad = np.radians(angle)
                dx = int(radius * np.cos(rad))
                dy = int(radius * np.sin(rad))

                x = center_x + dx
                y = center_y + dy

                if (0 <= x < self.width and
                    0 <= y < self.height and
                    self.maze[y][x] == 0):
                    print(f"✅ Centre trouvé: ({x}, {y})")
                    return (x, y)

        raise Exception("❌ Aucun centre accessible trouvé!")

    def is_valid(self, x, y):
        """Vérifier si la position est valide"""
        return (0 <= x < self.width and
                0 <= y < self.height and
                self.maze[y][x] == 0)

    def heuristic(self, point, goal):
        """Distance de Manhattan"""
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    def a_star(self, start, goal):
        """Algorithme A*"""
        print(f"\n⚡ Application de l'algorithme A*...")
        print(f"   Départ: {start}")
        print(f"   Arrivée: {goal}")

        start_time = time.time()

        counter = 0
        open_set = []
        heapq.heappush(open_set, (0, counter, start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        closed_set = set()
        nodes_explored = 0

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                end_time = time.time()
                path = self._reconstruct_path(came_from, goal)

                stats = {
                    'length': len(path),
                    'time': end_time - start_time,
                    'nodes_explored': nodes_explored,
                    'success': True
                }

                print(f"✅ Chemin trouvé!")
                print(f"   📏 Longueur: {len(path)} pas")
                print(f"   ⏱️  Temps: {stats['time']:.4f} secondes")
                print(f"   🔢 Nœuds explorés: {nodes_explored}")

                return path, stats

            if current in closed_set:
                continue

            closed_set.add(current)
            nodes_explored += 1
            x, y = current

            for dx, dy in self.directions:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)

                if not self.is_valid(nx, ny) or neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)

                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))

        end_time = time.time()
        stats = {
            'length': -1,
            'time': end_time - start_time,
            'nodes_explored': nodes_explored,
            'success': False
        }

        print(f"❌ Aucun chemin trouvé!")
        return None, stats

    def _reconstruct_path(self, came_from, current):
        """Reconstruire le chemin"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def create_animated_gif(self, path, start, goal, output_path='solution.gif',
                           frame_skip=5, duration=50, final_pause=2000):
        """
        Créer un GIF animé montrant la progression du chemin

        Args:
            path: Le chemin à animer
            start: Point de départ
            goal: Point d'arrivée
            output_path: Nom du fichier GIF
            frame_skip: Nombre de pas à sauter entre chaque frame (pour réduire la taille)
            duration: Durée de chaque frame en ms
            final_pause: Durée de pause sur la dernière frame en ms
        """
        print(f"\n🎬 Génération du GIF animé...")

        frames = []

        # Calculer les indices pour les frames
        path_length = len(path)
        frame_indices = list(range(0, path_length, frame_skip))

        # S'assurer que la dernière position est incluse
        if frame_indices[-1] != path_length - 1:
            frame_indices.append(path_length - 1)

        print(f"   📊 {len(frame_indices)} frames à générer...")

        for i, end_idx in enumerate(frame_indices):
            # Créer une nouvelle image pour chaque frame
            frame_img = self.original_image.convert('RGB')
            draw = ImageDraw.Draw(frame_img)

            # Dessiner le chemin parcouru jusqu'à maintenant
            current_path = path[:end_idx + 1]

            if len(current_path) > 1:
                for j in range(len(current_path) - 1):
                    x1, y1 = current_path[j]
                    x2, y2 = current_path[j + 1]
                    draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=3)

            # Position actuelle (tête du chemin) en jaune
            if current_path:
                cx, cy = current_path[-1]
                r = 6
                draw.ellipse([(cx-r, cy-r), (cx+r, cy+r)],
                           fill=(255, 255, 0), outline=(255, 255, 0))

            # Marquer le départ en VERT
            sx, sy = start
            r = 8
            draw.ellipse([(sx-r, sy-r), (sx+r, sy+r)],
                        fill=(0, 255, 0), outline=(0, 255, 0))

            # Marquer l'arrivée en BLEU
            gx, gy = goal
            draw.ellipse([(gx-r, gy-r), (gx+r, gy+r)],
                        fill=(0, 0, 255), outline=(0, 0, 255))

            frames.append(frame_img)

            if (i + 1) % 10 == 0:
                print(f"   ⏳ Progression: {i+1}/{len(frame_indices)} frames")

        # Sauvegarder le GIF
        print(f"   💾 Sauvegarde du GIF...")

        # Créer la liste des durées (dernière frame plus longue)
        durations = [duration] * (len(frames) - 1) + [final_pause]

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,  # 0 = boucle infinie
            optimize=False
        )

        print(f"✅ GIF animé sauvegardé: {output_path}")
        print(f"   🎞️  Frames: {len(frames)}")
        print(f"   ⏱️  Durée par frame: {duration}ms")
        print(f"   🔄 Lecture en boucle activée")
        print(f"   🟢 Vert = Départ")
        print(f"   🔴 Rouge = Chemin parcouru")
        print(f"   🟡 Jaune = Position actuelle")
        print(f"   🔵 Bleu = Arrivée")

    def visualize_solution(self, path, start, goal, output_path='solution.png'):
        """Créer une image statique avec le chemin tracé"""
        print(f"\n🎨 Génération de l'image statique...")

        result_img = self.original_image.convert('RGB')
        draw = ImageDraw.Draw(result_img)

        if path and len(path) > 1:
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=3)

            sx, sy = start
            r = 8
            draw.ellipse([(sx-r, sy-r), (sx+r, sy+r)],
                        fill=(0, 255, 0), outline=(0, 255, 0))

            gx, gy = goal
            draw.ellipse([(gx-r, gy-r), (gx+r, gy+r)],
                        fill=(0, 0, 255), outline=(0, 0, 255))

        result_img.save(output_path)
        print(f"✅ Image statique sauvegardée: {output_path}")

        return result_img

    def solve_auto(self, output_gif='solution.gif', output_png='solution.png',
                   frame_skip=5, duration=50):
        """Résolution automatique avec GIF et image statique"""
        print("\n" + "="*60)
        print("🤖 RÉSOLUTION AUTOMATIQUE DU LABYRINTHE")
        print("="*60)

        try:
            start = self.find_first_open_cell()
            goal = self.find_center()

            path, stats = self.a_star(start, goal)

            if stats['success']:
                # Générer l'image statique
                self.visualize_solution(path, start, goal, output_png)

                # Générer le GIF animé
                self.create_animated_gif(path, start, goal, output_gif,
                                        frame_skip=frame_skip, duration=duration)

                print("\n" + "="*60)
                print("✅ RÉSOLUTION RÉUSSIE!")
                print("="*60)
                print(f"📊 STATISTIQUES:")
                print(f"   • Longueur du chemin: {stats['length']} pas")
                print(f"   • Temps de calcul: {stats['time']:.4f} secondes")
                print(f"   • Nœuds explorés: {stats['nodes_explored']}")
                print("="*60)

                return path, stats
            else:
                print("\n❌ Impossible de trouver un chemin!")
                return None, stats

        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            return None, None

    def solve_manual(self, start, goal, output_gif='solution.gif',
                    output_png='solution.png', frame_skip=5, duration=50):
        """Résolution avec coordonnées manuelles"""
        print("\n" + "="*60)
        print("🎯 RÉSOLUTION AVEC COORDONNÉES MANUELLES")
        print("="*60)

        if not self.is_valid(start[0], start[1]):
            print(f"❌ Point de départ {start} invalide (dans un mur)")
            return None, None

        if not self.is_valid(goal[0], goal[1]):
            print(f"❌ Point d'arrivée {goal} invalide (dans un mur)")
            return None, None

        path, stats = self.a_star(start, goal)

        if stats['success']:
            # Générer l'image statique
            self.visualize_solution(path, start, goal, output_png)

            # Générer le GIF animé
            self.create_animated_gif(path, start, goal, output_gif,
                                    frame_skip=frame_skip, duration=duration)

            print("\n" + "="*60)
            print("✅ RÉSOLUTION RÉUSSIE!")
            print("="*60)
            print(f"📊 STATISTIQUES:")
            print(f"   • Longueur du chemin: {stats['length']} pas")
            print(f"   • Temps de calcul: {stats['time']:.4f} secondes")
            print(f"   • Nœuds explorés: {stats['nodes_explored']}")
            print("="*60)

            return path, stats
        else:
            print("\n❌ Impossible de trouver un chemin!")
            return None, stats


def main():
    """Programme principal"""

    image_path = 'labyrinthe.png'

    print("🚀 Démarrage du résolveur de labyrinthe...")
    print(f"📁 Fichier: {image_path}\n")

    try:
        solver = MazeAStar(image_path)

        # OPTION 1: Résolution automatique avec GIF
        print("\n🔄 Mode automatique activé...")
        path, stats = solver.solve_auto(
            output_gif='solution.gif',
            output_png='solution.png',
            frame_skip=5,      # Réduire pour plus de fluidité (mais fichier plus gros)
            duration=50        # ms entre chaque frame
        )

        # OPTION 2: Mode manuel avec GIF
        # Décommentez et ajustez ces lignes:
        """
        print("\n🔧 Mode manuel...")
        start = (50, 50)
        goal = (213, 216)
        path, stats = solver.solve_manual(
            start, goal,
            output_gif='solution.gif',
            output_png='solution.png',
            frame_skip=3,
            duration=50
        )
        """

        if path:
            print("\n✨ Fichiers générés:")
            print("   📄 solution.png - Image statique du chemin complet")
            print("   🎬 solution.gif - Animation de la progression")

    except FileNotFoundError:
        print(f"\n❌ ERREUR: Le fichier '{image_path}' n'existe pas!")
        print("📝 Placez votre image 'labyrinthe.png' dans le dossier du script")

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")


if __name__ == "__main__":
    main()
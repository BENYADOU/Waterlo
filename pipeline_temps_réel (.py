# =====================================================================
# Script de capture vidéo, application de watermark et envoi réseau UDP
# =====================================================================

# ==== IMPORTS ====
import cv2
import asyncio
import time
import torch
import subprocess
from apply_watermark import Generator, apply_watermark


# ==== PARAMÈTRES GÉNÉRAUX ====
CAMERA_ID = 0
FPS = 30                           # Fréquence cible d’acquisition (30 images/s)
WIDTH, HEIGHT = 640, 480           # Résolution de la capture vidéo
BUFFER_SIZE = 30                   # Taille d’un buffer (30 frames ≈ 1 seconde)
ALPHA = 0.03                       # Intensité du watermark appliqué
TARGET_SIZE = 256                  # Taille cible des images pour le modèle
WEIGHTS_PATH = "weights/G.pt"      # Chemin du modèle générateur pré-entraîné

# Adresse du récepteur (PC-2) pour la transmission UDP
TARGET_IP   = "192.168.1.2"
TARGET_PORT = 1236

ffmpeg_proc = None  # Processus FFmpeg (sera lancé dynamiquement)


# ==== INITIALISATION DU MODÈLE ====
# Choix du périphérique : GPU si disponible, sinon CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du générateur
generator = Generator().to(device)

# Chargement des poids du modèle (suppression du préfixe "module." si présent)
state_dict = torch.load(WEIGHTS_PATH, map_location=device)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
generator.load_state_dict(state_dict)
generator.eval()  # Mode évaluation (pas de backpropagation)


# ==== BUFFERS ====
# Deux buffers circulaires pour gérer la capture et le traitement en parallèle
buffer_A, buffer_B = [], []
current_buffer, next_buffer = buffer_A, buffer_B

# Dictionnaires pour suivre les frames et leurs timestamps
traitement_buffer = {}  # Images après watermark
timestamps = {}         # Timestamps de capture (pour calculer la latence)


# ==== FONCTION : OUVERTURE D’UN PIPE FFmpeg ====
def open_ffmpeg_sink(ip, port, w, h, fps, bitrate_mbps=40):
    """
    Ouvre un processus FFmpeg en sous-tâche qui :
    - prend les images brutes générées par OpenCV (BGR),
    - les encode en H.264,
    - les envoie en flux temps réel via UDP au PC-2.
    """
    # Construction de l’URL de sortie UDP
    url = f"udp://{ip}:{port}?pkt_size=1316&fifo_size=1000000&overrun_nonfatal=1"
    # Taille du buffer d’encodage (en Mb), proportionnel au bitrate choisi
    buf_mbps = max(2, bitrate_mbps // 10)

    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "warning",   # Mode silencieux (pas d’input utilisateur)
        "-f", "rawvideo", "-pix_fmt", "bgr24",          # Format brut attendu en entrée (images OpenCV)
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",    # Taille, FPS, et entrée via stdin
        "-an",                                          # Pas d’audio
        "-c:v", "libx264",                              # Codec H.264
        "-preset", "ultrafast",                         # Encodage rapide (sacrifice compression)
        "-tune", "zerolatency",                         # Mode faible latence
        "-profile:v", "baseline", "-level", "3.1",      # Profil H.264 compatible matériel
        "-bf", "0",                                     # Pas de B-frames (évite la latence)
        "-g", str(fps), "-keyint_min", str(fps),        # GOP = 1 seconde (1 keyframe/sec)
        "-x264-params", "scenecut=0:rc-lookahead=0:sync-lookahead=0:"
                        "nal-hrd=cbr:force-cfr=1:repeat-headers=1",
        "-b:v", f"{bitrate_mbps}M",                     # Bitrate vidéo cible
        "-maxrate", f"{bitrate_mbps}M",                 # Débit maximum autorisé
        "-bufsize", f"{buf_mbps}M",                     # Taille du buffer de régulation
        "-pix_fmt", "yuv420p",                          # Format standard pour compatibilité
        "-mpegts_flags", "+resend_headers+initial_discontinuity", # Headers répétés + synchro
        "-flush_packets", "1",                          # Envoi immédiat des paquets
        "-muxpreload", "0", "-muxdelay", "0",           # Pas de mise en tampon → latence minimale
        "-f", "mpegts", url                             # Format de sortie : MPEG-TS via UDP
    ]

    # Lance FFmpeg en sous-processus, prêt à recevoir les images brutes via stdin
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=10**7)


# ==== CAPTURE ====
async def capture_loop():
    """
    Capture les images de la caméra en temps réel et les stocke dans un buffer.
    Un double-buffer est utilisé pour éviter la perte d’images :
    - next_buffer : se remplit avec les nouvelles images capturées,
    - current_buffer : est traité par la boucle de traitement.
    """
    global current_buffer, next_buffer
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)    # Résolution largeur
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)  # Résolution hauteur
    cap.set(cv2.CAP_PROP_FPS, FPS)              # Fréquence cible (30 FPS)

    frame_id = 0  # Identifiant unique de chaque frame
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break  # Arrêt si la caméra ne fournit plus d’images

        # Sauvegarde du timestamp pour calculer la latence plus tard
        timestamps[frame_id] = int(start * 1000)

        # Stockage de l’image brute dans le buffer en cours de remplissage
        next_buffer.append((frame.copy(), frame_id))
        frame_id += 1

        # Si le buffer atteint 30 images, on l’échange avec celui en cours de traitement
        if len(next_buffer) >= BUFFER_SIZE:
            current_buffer, next_buffer = next_buffer, []

        # Pause adaptée pour maintenir ~30 FPS stables
        await asyncio.sleep(max(0, 1.0 / FPS - (time.time() - start)))

    cap.release()
    next_buffer.append((None, None))  # Signal de fin de capture


# ==== TRAITEMENT ====
async def traitement_loop():
    """
    Applique le modèle de watermark (Generator) sur les images du buffer.
    Utilisation d’un thread séparé via asyncio.to_thread pour éviter de bloquer
    la capture et l’affichage (car l’inférence PyTorch est lourde).
    """
    global current_buffer
    while True:
        if current_buffer:
            frame, frame_id = current_buffer.pop(0)
            if frame is None:  # Signal de fin
                traitement_buffer[frame_id] = None
                break

            # Application du watermark en tâche séparée (non bloquante)
            img_traitee = await asyncio.to_thread(
                apply_watermark, generator, frame.copy(), device, ALPHA, TARGET_SIZE
            )

            # Sauvegarde de l’image brute et de l’image filigranée
            traitement_buffer[frame_id] = (frame, img_traitee)
        else:
            # Petite pause si le buffer est vide (évite 100% CPU)
            await asyncio.sleep(0.001)


# ==== AFFICHAGE & ENVOI ====
async def affichage_loop():
    """
    Affiche les images brutes et watermarkées en local + envoie le flux watermarké
    vers PC-2 via FFmpeg (UDP).
    Ajoute aussi des overlays (FPS, latence) pour suivre les performances.
    """
    global ffmpeg_proc
    cv2.namedWindow("Brute", cv2.WINDOW_AUTOSIZE)   # Fenêtre image brute
    cv2.namedWindow("Watermark", cv2.WINDOW_NORMAL) # Fenêtre image watermarkée

    current_id = 0  # Index de la prochaine frame à afficher/envoyer
    frame_count, last_time = 0, time.time()
    presentation_timer = time.time()

    while True:
        if current_id in traitement_buffer:
            pair = traitement_buffer.pop(current_id)
            if pair is None:  # Signal de fin
                break

            frame_brute, frame_watermarked = pair

            # Démarrage de FFmpeg au premier envoi
            if ffmpeg_proc is None:
                ffmpeg_proc = open_ffmpeg_sink(TARGET_IP, TARGET_PORT, WIDTH, HEIGHT, FPS)

            # Mise à l’échelle si nécessaire
            if frame_watermarked.shape[:2] != (HEIGHT, WIDTH):
                frame_watermarked = cv2.resize(frame_watermarked, (WIDTH, HEIGHT))

            # Envoi de l’image filigranée à FFmpeg (qui l’encode et la pousse en UDP)
            try:
                ffmpeg_proc.stdin.write(frame_watermarked.tobytes())
            except (BrokenPipeError, Exception):
                ffmpeg_proc = None  # Si problème, relancer FFmpeg plus tard

            # Calcul latence : temps entre capture et affichage
            now_ms = int(time.time() * 1000)
            delta_ms = now_ms - timestamps.get(current_id, now_ms)

            # Calcul FPS affiché (nombre d’images envoyées par seconde)
            frame_count += 1
            if time.time() - last_time >= 1.0:
                fps = frame_count / (time.time() - last_time)
                frame_count, last_time = 0, time.time()
            else:
                fps = 0.0

            # Ajout HUD : FPS et latence
            view = cv2.resize(frame_watermarked, (1152, 864))
            cv2.putText(view, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(view, f"Latence: {delta_ms}ms", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Affichage local
            cv2.imshow("Brute", frame_brute)
            cv2.imshow("Watermark", view)

            if cv2.waitKey(1) == 27:  # Quitter avec ESC
                break

            current_id += 1
        else:
            await asyncio.sleep(0.001)

    # Nettoyage final
    if ffmpeg_proc is not None:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait(timeout=2)
    cv2.destroyAllWindows()


# ==== MAIN ====
async def main():
    """
    Fonction principale : lance en parallèle
    - la capture vidéo,
    - le traitement watermark,
    - l’affichage et l’envoi réseau.
    """
    await asyncio.gather(
        capture_loop(),
        traitement_loop(),
        affichage_loop()
    )

if __name__ == "__main__":
    asyncio.run(main())

# =====================================================================
# Script de détection du watermark (modèle Bob) à partir d’un flux vidéo
# =====================================================================

import cv2
import torch
import numpy as np
import torch.nn as nn
import time
from collections import deque


# ==== BLOCS U-NET ====
class UNetDown(nn.Module):
    """Bloc d’encodage : convolution + normalisation + activation"""
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Bloc de décodage avec skip connection"""
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1)


# ==== MODÈLE BOB (Détecteur) ====
class Bob(nn.Module):
    """
    Modèle de détection basé sur U-Net simplifié :
    - encodeur seulement (4 downsampling),
    - sortie = heatmap 1 canal avec Sigmoid.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(Bob, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(512, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        return self.final(d4)


# ==== FONCTION DE DÉTECTION ====
def detect_watermark(frame, model, device, target_size=256):
    """
    Applique Bob sur une image et retourne un overlay avec heatmap colorée.
    """
    # Prétraitement
    img = cv2.resize(frame, (target_size, target_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    # Inférence
    with torch.no_grad():
        output = model(img_tensor)
        heatmap = output[0, 0].cpu().numpy()
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

        # Colorisation de la heatmap
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Détection simple avec seuil sur intensité moyenne
        if heatmap_colored.mean() > 125:
            print("Watermark détectée !", heatmap_colored.mean())

        # Fusion overlay
        overlay = cv2.addWeighted(frame, 0.5, heatmap_colored, 0.5, 0)

    return overlay


# ==== COMPTEUR FPS ====
class RateCounter:
    """Compteur sur fenêtre glissante (par défaut 1 seconde)."""
    def __init__(self, window_s: float = 1.0):
        self.window = window_s
        self.tstamps = deque()

    def tick(self, t: float = None):
        if t is None:
            t = time.perf_counter()
        self.tstamps.append(t)
        cutoff = t - self.window
        while self.tstamps and self.tstamps[0] < cutoff:
            self.tstamps.popleft()

    def fps(self) -> float:
        return len(self.tstamps) / self.window if self.tstamps else 0.0


# ==== MAIN ====
def main():
    # Flux vidéo en entrée (UDP)
    stream_url = "udp://192.168.1.1:1234?overrun_nonfatal=1&fifo_size=50000000"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chargement du modèle Bob
    model = Bob().to(device)
    state_dict = torch.load("weights/B.pt", map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Ouverture du flux vidéo
    print("Ouverture du flux :", stream_url)
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"❌ Impossible d’ouvrir {stream_url}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ Flux ouvert: {w}x{h} @ {reported_fps} fps (valeur reportée, peut être inexacte)")

    cv2.namedWindow("Détection DeepFake - Bob", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Détection DeepFake - Bob", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Compteurs FPS
    recv_meter = RateCounter(window_s=1.0)   # Frames reçues
    proc_meter = RateCounter(window_s=1.0)   # Tours de boucle
    last_log = time.perf_counter()

    # Boucle principale
    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            proc_meter.tick(t0)
            print("⚠️ Frame invalide, on saute…")
            if cv2.waitKey(1) == 27:
                break
            continue

        recv_meter.tick(t0)

        # Détection watermark
        result = detect_watermark(frame, model, device)

        # Calcul FPS + latence
        recv_fps = recv_meter.fps()
        t1 = time.perf_counter()
        proc_meter.tick(t1)
        proc_fps = proc_meter.fps()
        dt_ms = (t1 - t0) * 1000.0

        # Overlay des métriques
        line1 = f"Recv: {recv_fps:4.1f} fps | Proc: {proc_fps:4.1f} fps"
        line2 = f"Frame time: {dt_ms:6.1f} ms"
        cv2.putText(result, line1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(result, line2, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Détection DeepFake - Bob", result)

        # Log console toutes les 2s
        now = t1
        if now - last_log > 2.0:
            print(line1, "|", line2)
            last_log = now

        if cv2.waitKey(1) == 27:  # ESC pour quitter
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Détection terminée.")


if __name__ == "__main__":
    print("▶️  Lancement détection réseau…")
    main()

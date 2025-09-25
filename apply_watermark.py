# =====================================================================
# Script d’application du watermark avec le modèle Generator (U-Net)
# =====================================================================

import argparse
import cv2
import torch
import numpy as np
import os
import torch.nn as nn


# ==== BLOCS DE L’ARCHITECTURE U-NET ====
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
    """Bloc de décodage : déconvolution + normalisation + activation + skip connection"""
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
        # Concaténation avec la sortie correspondante de l’encodeur
        return torch.cat((x, skip_input), 1)


# ==== GENERATOR (U-Net) ====
class Generator(nn.Module):
    """
    Modèle principal Generator basé sur un U-Net :
    - encodeur : réduit progressivement la taille de l’image
    - décodeur : reconstruit l’image et insère le watermark
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
        # Encodeur
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        # Décodeur
        self.up1 = UNetUp(512, 256, dropout=0.5)
        self.up2 = UNetUp(512, 128)
        self.up3 = UNetUp(256, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        return self.final(u3)


# ==== CHARGEMENT DES MODÈLES ====
class Models:
    """Classe utilitaire pour charger les poids d’un modèle"""
    def __init__(self, G, B):
        self.G = G
        self.B = B
    def load(self, weights):
        self.G.load_state_dict(torch.load(weights, map_location="cpu"))
        self.G.eval()


# ==== PRÉ/POST-TRAITEMENT ====
def preprocess_frame(frame, target_size=256):
    """Redimensionne et convertit une image OpenCV en tenseur PyTorch"""
    frame_resized = cv2.resize(frame, (target_size, target_size))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
    return frame_tensor


def postprocess_frame(tensor, original_size):
    """Convertit un tenseur PyTorch en image OpenCV BGR"""
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
    frame_bgr = cv2.resize(frame_bgr, original_size)
    return frame_bgr


def apply_watermark(generator, frame, device, alpha, target_size=256):
    """
    Applique le watermark sur une image :
    - passage dans le modèle Generator,
    - ajout du watermark pondéré par alpha,
    - clampage entre 0 et 1 pour rester dans l’espace valide.
    """
    original_size = (frame.shape[1], frame.shape[0])  # (width, height)
    frame_tensor = preprocess_frame(frame, target_size).to(device)
    with torch.no_grad():
        watermark = generator(frame_tensor)
        watermarked = frame_tensor + alpha * watermark
        watermarked = torch.clamp(watermarked, 0, 1)
    return postprocess_frame(watermarked, original_size)


# ==== MAIN SCRIPT ====
def main(args):
    # Vérifie que le dossier de sortie existe
    if args.output is not None:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(in_channels=3, out_channels=3).to(device)
    models = Models(generator, None)
    models.load(args.weights)

    # Choix de la source vidéo (webcam ou fichier vidéo)
    if args.input_type == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.input)

    # Préparation de la sauvegarde vidéo si demandé
    out_writer = None
    if args.output is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Vidéo sauvegardée dans: {args.output} | Taille: {width}x{height} | FPS: {fps}")
        out_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        if not out_writer.isOpened():
            print("Erreur: Impossible d'ouvrir le fichier vidéo de sortie!")
            exit(1)

    # Boucle principale : lecture + watermark + affichage
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        watermarked_frame = apply_watermark(generator, frame, device, args.alpha, target_size=args.target_size)
        watermarked_frame = cv2.resize(watermarked_frame, (width, height))

        cv2.imshow('Watermarked', watermarked_frame)

        if out_writer:
            out_writer.write(watermarked_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()


# ==== EXECUTION EN LIGNE DE COMMANDE ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_size", type=int, default=256, help="Taille d'entrée pour le modèle (256 ou 512)")
    parser.add_argument("--weights", type=str, required=True, help="Chemin des poids du Generator")
    parser.add_argument("--input_type", choices=["webcam", "video"], default="webcam", help="Source : webcam ou vidéo")
    parser.add_argument("--input", type=str, help="Chemin de la vidéo (si input_type=video)")
    parser.add_argument("--output", type=str, help="Chemin de sortie pour la vidéo watermarked (ex: results/video_watermarked.avi)")
    parser.add_argument("--alpha", type=float, default=0.005, help="Force du watermark")
    args = parser.parse_args()
    main(args)


# ================================================================
#  eval_all_models.py
#  Įklijuok į naują Colab cell ir paleisk.
#  Generuoja confusion matrix ir F1 diagramas visiems modeliams.
# ================================================================

# ── Importai ────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.models.video as video_models
from sklearn.metrics import f1_score, multilabel_confusion_matrix

try:
    import timm
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "timm", "-q"])
    import timm

# ── Konstantos ──────────────────────────────────────────────────
LABELS      = ["y_stand","y_lying_down","y_foraging","y_drinking_water","y_rumination"]
LABEL_NAMES = ["stand","lying_down","foraging","drinking_water","rumination"]
NUM_CLASSES = 5
IMG_SIZE    = 224
CLIP_SIZE   = 8        # kadrų skaičius klipe — pakeisk jei kitaip naudojai
BATCH_SIZE  = 16
SEED        = 42
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

MANIFEST    = "/content/cow_behavior/manifest_colab.csv"
DRIVE_OUT   = Path("/content/drive/MyDrive/Project/cow_behavior/outputs")
DRIVE_OUT.mkdir(parents=True, exist_ok=True)

PATHS = {
    "ResNet-18":     "/content/drive/MyDrive/Project/cow_behavior/models/cow_behavior_resnet_4class_best.pth",
    "R3D-18 v1":     "/content/drive/MyDrive/cbvd5_data/outputs_yolo_r3d/yolo_r3d_best_f1.pth",
    "R3D-18 ultra":  "/content/drive/MyDrive/cbvd5_data/outputs_upgraded/best_tuned_f1.pth",
    "VideoViT":      "/content/drive/MyDrive/cbvd5_data/outputs/videovit_best.pth",
}

print("Device:", DEVICE)

# ════════════════════════════════════════════════════════════════
#  MODELIŲ KLASĖS  (atkuriamos pagal state_dict struktūrą)
# ════════════════════════════════════════════════════════════════

# ── ResNet-18 (kadrų modelis) ───────────────────────────────────
class ResNet18Model(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        base.fc = nn.Linear(base.fc.in_features, NUM_CLASSES)
        # perkeliame viską kaip tiesiogiai (be wrapper'io)
        self.__dict__["_modules"] = base.__dict__["_modules"]
        self.fc = base.fc

    def forward(self, x):
        return self._forward_impl(x)

# Paprasčiau — tiesiog naudojame resnet18 tiesiogiai
def build_resnet18():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m

# ── R3D-18 v1 (backbone.fc) ─────────────────────────────────────
# State dict: backbone.stem.*, backbone.layer*, backbone.fc.*
class R3DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = video_models.r3d_18(weights=None)
        self.backbone.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        return self.backbone(x)

# ── R3D-18 ultra (backbone + custom head) ───────────────────────
# State dict: backbone.*, head.1 (512→256), head.4 (256→5)
class R3DUpgradedModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = video_models.r3d_18(weights=None)
        # pašaliname originalų fc
        base.fc = nn.Identity()
        self.backbone = base
        self.head = nn.Sequential(
            nn.Dropout(0.5),          # head.0
            nn.Linear(512, 256),      # head.1
            nn.ReLU(),                # head.2
            nn.Dropout(0.3),          # head.3
            nn.Linear(256, NUM_CLASSES),  # head.4
        )

    def forward(self, x):
        feat = self.backbone(x)   # (B, 512)
        return self.head(feat)

# ── VideoViT ────────────────────────────────────────────────────
# State dict: vit.*, temporal_attn.*, norm.*, head.1 (768→5)
class VideoViTModel(nn.Module):
    def __init__(self, num_frames=CLIP_SIZE):
        super().__init__()
        self.num_frames = num_frames
        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0
        )
        hidden = 768
        self.temporal_attn = nn.MultiheadAttention(
            hidden, num_heads=8, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Dropout(0.5),              # head.0
            nn.Linear(hidden, NUM_CLASSES),  # head.1
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        # apdorojame kiekvieną kadrą atskirai
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        feats = self.vit(x)               # (B*T, 768)
        feats = feats.reshape(B, T, -1)   # (B, T, 768)
        attn_out, _ = self.temporal_attn(feats, feats, feats)
        feats = self.norm(feats + attn_out)
        cls = feats.mean(dim=1)           # (B, 768)
        return self.head(cls)

# ════════════════════════════════════════════════════════════════
#  DATASET KLASĖS
# ════════════════════════════════════════════════════════════════

# Test split (70/15/15 — tas pats kaip train skripte)
def make_test_split(df):
    videos = df["video_id"].dropna().unique()
    rng = np.random.default_rng(SEED)
    rng.shuffle(videos)
    n = len(videos)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    test_videos = videos[n_train + n_val:]
    return df[df["video_id"].isin(test_videos)].copy()

# ── Kadrų dataset (ResNet-18) ────────────────────────────────────
class FrameDataset(Dataset):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["frame_path"]).convert("RGB")
        img = self.tf(img)
        y   = torch.tensor(row[LABELS].values.astype(np.float32))
        return img, y

# ── Klipų dataset (R3D-18, VideoViT) ────────────────────────────
class ClipDataset(Dataset):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    def __init__(self, df, clip_size=CLIP_SIZE):
        # grupuojam pagal video_id ir renkam klipus
        self.samples = []
        self.clip_size = clip_size
        for vid, grp in df.groupby("video_id"):
            grp = grp.sort_values("frame_path").reset_index(drop=True)
            # sliding window su žingsniu clip_size//2
            step = max(1, clip_size // 2)
            for start in range(0, len(grp) - clip_size + 1, step):
                chunk = grp.iloc[start:start + clip_size]
                labels = chunk[LABELS].values.astype(np.float32).max(axis=0)
                self.samples.append((
                    chunk["frame_path"].tolist(),
                    labels
                ))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        paths, labels = self.samples[idx]
        frames = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            frames.append(self.tf(img))
        clip = torch.stack(frames, dim=1)  # (C, T, H, W)
        return clip, torch.tensor(labels)

# ════════════════════════════════════════════════════════════════
#  INFERENCE
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(model, loader, threshold=0.5):
    model.eval()
    all_probs, all_targets = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        probs = torch.sigmoid(model(x)).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.numpy())
    probs   = np.vstack(all_probs)
    targets = np.vstack(all_targets)
    preds   = (probs >= threshold).astype(int)
    return probs, preds, targets

def get_metrics(preds, targets):
    macro = f1_score(targets, preds, average="macro", zero_division=0)
    micro = f1_score(targets, preds, average="micro", zero_division=0)
    per   = f1_score(targets, preds, average=None,    zero_division=0).tolist()
    return macro, micro, per

# ════════════════════════════════════════════════════════════════
#  GRAFIKAI
# ════════════════════════════════════════════════════════════════

def plot_confusion(preds, targets, title, save_path):
    cms = multilabel_confusion_matrix(targets, preds)
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(18, 3.8))
    fig.suptitle(title, fontsize=13, y=1.02)
    for i, ax in enumerate(axes):
        cm = cms[i]
        ax.imshow(cm, cmap=plt.cm.Blues)
        ax.set_title(LABEL_NAMES[i], fontsize=10, pad=6)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred 0","Pred 1"], fontsize=8)
        ax.set_yticklabels(["True 0","True 1"], fontsize=8)
        thr = cm.max() / 2
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r,c]),
                        ha="center", va="center", fontsize=12,
                        color="white" if cm[r,c] > thr else "black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✓ {save_path.name}")

def plot_f1_comparison(results, save_path):
    names  = list(results.keys())
    macro  = [results[m]["macro"] for m in names]
    micro  = [results[m]["micro"] for m in names]
    x = np.arange(len(names)); w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, macro, w, label="Macro F1", color="#378ADD",
                zorder=3, edgecolor="white")
    b2 = ax.bar(x + w/2, micro, w, label="Micro F1", color="#1D9E75",
                zorder=3, edgecolor="white")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.1); ax.set_ylabel("F1 balas", fontsize=11)
    ax.set_title("Macro ir Micro F1 palyginimas", fontsize=13)
    ax.legend(fontsize=10); ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✓ {save_path.name}")

def plot_per_class(results, save_path):
    names  = list(results.keys())
    colors = ["#888780", "#7F77DD", "#378ADD", "#1D9E75"]
    x = np.arange(NUM_CLASSES)
    w = 0.20
    offsets = np.linspace(
        -(len(names)-1)/2, (len(names)-1)/2, len(names)
    ) * w

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, color) in enumerate(zip(names, colors)):
        vals = results[name]["per_class"]
        bars = ax.bar(x + offsets[i], vals, w, label=name,
                      color=color, zorder=3, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            if h > 0.05:
                ax.annotate(f"{h:.2f}",
                            xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(LABEL_NAMES, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel("F1 balas", fontsize=11)
    ax.set_title("Per-klasės F1 palyginimas", fontsize=13)
    ax.legend(fontsize=10); ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✓ {save_path.name}")

# ════════════════════════════════════════════════════════════════
#  PAGRINDINIS BLOKAS
# ════════════════════════════════════════════════════════════════

print("\n[Duomenys] Įkeliamas manifest...")
df = pd.read_csv(MANIFEST)
df = df.dropna(subset=["video_id","frame_path"]).copy()
test_df = make_test_split(df)
print(f"  Test kadrų: {len(test_df)}")

frame_loader = DataLoader(
    FrameDataset(test_df),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)
clip_loader = DataLoader(
    ClipDataset(test_df, clip_size=CLIP_SIZE),
    batch_size=4, shuffle=False, num_workers=2
)
print(f"  Frame batches: {len(frame_loader)} | Clip batches: {len(clip_loader)}")

results = {}

# ── 1. ResNet-18 ─────────────────────────────────────────────────
print("\n[1/4] ResNet-18...")
m_rn = build_resnet18().to(DEVICE)
cp = torch.load(PATHS["ResNet-18"], map_location=DEVICE)
state = cp["model_state_dict"] if "model_state_dict" in cp else cp
m_rn.load_state_dict(state)
_, preds, targets_frame = run_inference(m_rn, frame_loader)
macro, micro, per = get_metrics(preds, targets_frame)
results["ResNet-18"] = {"macro": macro, "micro": micro, "per_class": per,
                        "preds": preds, "targets": targets_frame}
print(f"  Macro F1: {macro:.4f} | Micro F1: {micro:.4f}")
plot_confusion(preds, targets_frame, "Confusion matrices – ResNet-18",
               DRIVE_OUT / "confusion_resnet18.png")

# ── 2. VideoViT ──────────────────────────────────────────────────
print("\n[2/4] VideoViT...")
m_vit = VideoViTModel().to(DEVICE)
cp = torch.load(PATHS["VideoViT"], map_location=DEVICE, weights_only=False)
m_vit.load_state_dict(cp["model_state_dict"])
_, preds, targets_clip = run_inference(m_vit, clip_loader)
macro, micro, per = get_metrics(preds, targets_clip)
results["VideoViT"] = {"macro": macro, "micro": micro, "per_class": per,
                       "preds": preds, "targets": targets_clip}
print(f"  Macro F1: {macro:.4f} | Micro F1: {micro:.4f}")
plot_confusion(preds, targets_clip, "Confusion matrices – VideoViT",
               DRIVE_OUT / "confusion_videovit.png")

# ── 3. R3D-18 v1 ─────────────────────────────────────────────────
print("\n[3/4] R3D-18 v1...")
m_r3d = R3DModel().to(DEVICE)
cp = torch.load(PATHS["R3D-18 v1"], map_location=DEVICE, weights_only=False)
m_r3d.load_state_dict(cp["model_state_dict"])
_, preds, _ = run_inference(m_r3d, clip_loader)
macro, micro, per = get_metrics(preds, targets_clip)
results["R3D-18 v1"] = {"macro": macro, "micro": micro, "per_class": per,
                        "preds": preds, "targets": targets_clip}
print(f"  Macro F1: {macro:.4f} | Micro F1: {micro:.4f}")
plot_confusion(preds, targets_clip, "Confusion matrices – R3D-18 v1",
               DRIVE_OUT / "confusion_r3d_v1.png")

# ── 4. R3D-18 ultra ──────────────────────────────────────────────
print("\n[4/4] R3D-18 ultra...")
m_ult = R3DUpgradedModel().to(DEVICE)
cp = torch.load(PATHS["R3D-18 ultra"], map_location=DEVICE, weights_only=False)
m_ult.load_state_dict(cp["model_state_dict"])
_, preds, _ = run_inference(m_ult, clip_loader)
macro, micro, per = get_metrics(preds, targets_clip)
results["R3D-18 ultra"] = {"macro": macro, "micro": micro, "per_class": per,
                           "preds": preds, "targets": targets_clip}
print(f"  Macro F1: {macro:.4f} | Micro F1: {micro:.4f}")
plot_confusion(preds, targets_clip, "Confusion matrices – R3D-18 ultra",
               DRIVE_OUT / "confusion_r3d_ultra.png")

# ── Suvestinės diagramos ──────────────────────────────────────────
print("\n[Grafikai] Piešiamos suvestinės diagramos...")
plot_f1_comparison(results, DRIVE_OUT / "f1_comparison.png")
plot_per_class(results,     DRIVE_OUT / "per_class_f1.png")

# ── Galutinė suvestinė ────────────────────────────────────────────
print("\n" + "="*55)
print(f"{'Modelis':<18} {'Macro F1':>10} {'Micro F1':>10}")
print("="*55)
for name, r in results.items():
    print(f"{name:<18} {r['macro']:>10.4f} {r['micro']:>10.4f}")
print("="*55)
print(f"\nVisi failai išsaugoti → {DRIVE_OUT}")

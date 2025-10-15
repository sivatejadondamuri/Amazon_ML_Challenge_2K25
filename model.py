# train_multimodal.py
# Multimodal training: Image(CNN) + Text (MPNet embeddings + PCA) + Tabular -> Fusion NN
# Saves: fusion_model.pt, pca_embeddings.pkl, price_scaler.pkl, target_encoder.pkl, text_embed_cache.npy

import os
import re
import math
import time
import random
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from category_encoders import TargetEncoder

# ----------------- CONFIG -----------------
CSV_PATH = r"d:\student_resource\dataset\train.csv"
IMG_DIR = r"d:\student_resource\dataset\product_images"   # folder with downloaded images
OUT_DIR = r"d:\student_resource\outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# artifacts
EMBED_CACHE = os.path.join(OUT_DIR, "text_embeddings_mpnet.npy")
PCA_PATH = os.path.join(OUT_DIR, "pca_embeddings.pkl")
SCALER_PATH = os.path.join(OUT_DIR, "price_scaler.pkl")
TARGET_ENCODER_PATH = os.path.join(OUT_DIR, "target_encoder.pkl")
FUSION_WEIGHTS = os.path.join(OUT_DIR, "fusion_model.pt")
TEXT_EMBED_MODEL = "all-mpnet-base-v2"

# training config
SAMPLE_LIMIT = None   # set to e.g. 5000 for quick testing; else None to use all
BATCH_SIZE = 64
NUM_WORKERS = 4
IMAGE_SIZE = 224
EPOCHS = 6
LR = 3e-4
WEIGHT_DECAY = 1e-5
RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 1
PCA_DIM = 512   # reduced dimension for text embeddings
# ------------------------------------------

# ---------------- Helper functions ----------------
def img_filename_from_url(url):
    try:
        return Path(str(url)).name
    except:
        return ""

# Dataset class (safe to define at top-level)
class MultimodalDataset(Dataset):
    def __init__(self, df, text_feats_scaled, transform=None, device="cpu"):
        self.df = df.reset_index(drop=True)
        self.text_feats = text_feats_scaled.astype(np.float32)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # text/tabular features
        text_feat = self.text_feats[idx]   # numpy float32

        # image
        img_path = row.get("image_path", "")
        img_exists = bool(img_path and os.path.exists(img_path) and os.path.getsize(img_path) > 0)
        if img_exists:
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (127, 127, 127))
        else:
            # neutral gray image (so after normalization it's near zero)
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (127, 127, 127))

        # Always pass a PIL image into the transform pipeline (train/eval transforms expect PIL)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # target (log1p)
        price = float(row["price"])
        target = math.log1p(price)
        return img.float(), torch.from_numpy(text_feat), torch.tensor(target, dtype=torch.float32)

# Model definition (safe at top-level)
IMG_EMB_DIM = 512
# TEXT_FEAT_DIM will be set after we compute text_features_scaled in main
FUSION_HIDDEN = 512

class FusionModel(nn.Module):
    def __init__(self, img_emb_dim=IMG_EMB_DIM, text_dim=512, hidden_dim=FUSION_HIDDEN):
        super().__init__()
        # pretrained resnet34 backbone
        # keep 'pretrained=True' for backward compatibility; warnings expected on newer torchvision
        self.backbone = models.resnet34(pretrained=True)
        # replace final fc with projection to img_emb_dim
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_f, img_emb_dim)
        # optionally freeze early layers for faster training
        for name, param in self.backbone.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        # small text projection (if needed)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # fusion head
        self.fusion = nn.Sequential(
            nn.Linear(img_emb_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)   # output log1p(price)
        )

    def forward(self, img, text_feat):
        # img: (B,3,H,W), text_feat: (B, text_dim)
        img_emb = self.backbone(img)            # (B, img_emb_dim)
        text_proj = self.text_proj(text_feat)   # (B, hidden_dim)
        x = torch.cat([img_emb, text_proj], dim=1)
        out = self.fusion(x).squeeze(1)
        return out

# ---------------- MAIN ----------------
if __name__ == "__main__":
    import nltk
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()  # required on Windows EXEs, harmless otherwise

    # reproducible
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(RANDOM_STATE)

    print("Device:", DEVICE)

    # ------------- load csv & basic preprocessing -------------
    df = pd.read_csv(CSV_PATH, on_bad_lines="skip")
    df = df.dropna(subset=["catalog_content", "price"]).reset_index(drop=True)
    print("Total rows in CSV (after dropping null text/price):", len(df))

    if SAMPLE_LIMIT:
        df = df.sample(SAMPLE_LIMIT, random_state=RANDOM_STATE).reset_index(drop=True)
        print("Using SAMPLE_LIMIT rows:", len(df))

    # image filename/path
    df["image_filename"] = df["image_link"].apply(img_filename_from_url)
    df["image_path"] = df["image_filename"].apply(lambda fn: os.path.join(IMG_DIR, fn) if fn else "")
    df["has_image"] = df["image_path"].apply(lambda p: os.path.exists(p) and os.path.getsize(p) > 0)
    num_images = df["has_image"].sum()
    print(f"Found {num_images:,} images available in {IMG_DIR} (out of {len(df):,} rows).")

    # ------------- text cleaning -------------
    try:
        nltk.data.find("corpora/stopwords")
    except Exception:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except Exception:
        nltk.download("wordnet", quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stop_words = set(stopwords.words("english"))
    lemm = WordNetLemmatizer()

    def clean_text(text):
        s = str(text).lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        tokens = [lemm.lemmatize(w) for w in s.split() if w not in stop_words]
        return " ".join(tokens)

    tqdm.pandas()
    df["catalog_content_clean"] = df["catalog_content"].progress_apply(clean_text)

    # ------------- Text embeddings (SentenceTransformer) -------------
    embeddings = None
    if os.path.exists(EMBED_CACHE):
        try:
            embeddings = np.load(EMBED_CACHE)
            if embeddings.shape[0] != len(df):
                print("Cached embeddings length mismatch; regenerating embeddings.")
                os.remove(EMBED_CACHE)
                embeddings = None
            else:
                print("Loaded cached text embeddings from", EMBED_CACHE)
        except Exception:
            embeddings = None

    if embeddings is None:
        print("Creating sentence embeddings (this may take time)...")
        embedder = SentenceTransformer(TEXT_EMBED_MODEL, device=DEVICE)
        embeddings = embedder.encode(df["catalog_content_clean"].tolist(),
                                     batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        np.save(EMBED_CACHE, embeddings)
        del embedder
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print("Saved text embeddings to", EMBED_CACHE)

    print("Text embeddings shape:", embeddings.shape)

    # ---------------- train/val split (indices) ----------------
    from sklearn.model_selection import train_test_split
    idxs = np.arange(len(df))
    train_idxs, val_idxs = train_test_split(idxs, test_size=0.15, random_state=RANDOM_STATE)
    train_df = df.iloc[train_idxs].reset_index(drop=True)
    val_df = df.iloc[val_idxs].reset_index(drop=True)
    print("Train size:", len(train_df), "Val size:", len(val_df))

    # ---------------- PCA on text embeddings: fit on train only ----------------
    train_embeddings = embeddings[train_idxs]
    if os.path.exists(PCA_PATH):
        try:
            pca = joblib.load(PCA_PATH)
            print("Loaded PCA from", PCA_PATH)
        except Exception:
            pca = None
    else:
        pca = None

    if pca is None:
        pca = PCA(n_components=PCA_DIM, random_state=RANDOM_STATE)
        pca.fit(train_embeddings)
        joblib.dump(pca, PCA_PATH)
        print("Fitted and saved PCA ->", PCA_PATH)

    embeddings_reduced = pca.transform(embeddings)  # transform all rows
    print("Reduced embeddings shape:", embeddings_reduced.shape)

    # ---------------- Tabular features (Value, Unit, etc.) ----------------
    TEXT_COL = "catalog_content"
    TARGET_COL = "price"
    exclude_cols = {TEXT_COL, "catalog_content_clean", TARGET_COL, "sample_id", "image_link", "image_filename", "image_path", "has_image"}
    tabular_cols = [c for c in df.columns if c not in exclude_cols]

    numeric_cols = df[tabular_cols].select_dtypes(include=[np.number]).columns.tolist() if len(tabular_cols) > 0 else []
    cat_cols = [c for c in tabular_cols if c not in numeric_cols] if len(tabular_cols) > 0 else []

    print("Tabular numeric columns:", numeric_cols)
    print("Tabular categorical columns:", cat_cols)

    # TargetEncoder: fit on train only, transform full df
    te = None
    if len(cat_cols) > 0:
        te = TargetEncoder(cols=cat_cols)
        te.fit(train_df[cat_cols], train_df[TARGET_COL])
        df[cat_cols] = te.transform(df[cat_cols])
        joblib.dump(te, TARGET_ENCODER_PATH)
        print("Target encoder saved:", TARGET_ENCODER_PATH)

    # Impute numeric missing values using train medians and apply to full df
    if len(numeric_cols) > 0:
        for c in numeric_cols:
            median_val = train_df[c].median()
            df[c] = df[c].fillna(median_val)
        tab_numeric = df[numeric_cols].to_numpy()
    else:
        tab_numeric = None

    tab_cat = df[cat_cols].to_numpy() if len(cat_cols) > 0 else None

    if tab_numeric is None and tab_cat is None:
        tabular_features = None
        print("No tabular features found; using text embeddings only.")
    elif tab_numeric is not None and tab_cat is not None:
        tabular_features = np.hstack([tab_numeric, tab_cat])
    elif tab_numeric is not None:
        tabular_features = tab_numeric
    else:
        tabular_features = tab_cat

    # ---------------- Combine text reduced + tabular into text_feature_vector ----------------
    if tabular_features is not None:
        text_features = np.hstack([embeddings_reduced, tabular_features])
    else:
        text_features = embeddings_reduced.copy()

    print("Combined text+tabular feature shape:", text_features.shape)

    # ---------------- StandardScaler: fit on train only ----------------
    train_text_features = text_features[train_idxs]
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            print("Loaded scaler:", SCALER_PATH)
        except Exception:
            scaler = None
    else:
        scaler = None

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(train_text_features)
        joblib.dump(scaler, SCALER_PATH)
        print("Fitted and saved scaler:", SCALER_PATH)

    text_features_scaled = scaler.transform(text_features)
    train_text_feats = text_features_scaled[train_idxs]
    val_text_feats = text_features_scaled[val_idxs]

    print("Text features scaled. Shape:", text_features_scaled.shape)

    # ---------------- Image transforms ----------------
    train_img_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_img_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ---------------- build dataset and dataloaders ----------------
    train_ds = MultimodalDataset(train_df, train_text_feats, transform=train_img_transform, device=DEVICE)
    val_ds = MultimodalDataset(val_df, val_text_feats, transform=eval_img_transform, device=DEVICE)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ---------------- Model setup ----------------
    TEXT_FEAT_DIM = text_features_scaled.shape[1]
    model = FusionModel(img_emb_dim=IMG_EMB_DIM, text_dim=TEXT_FEAT_DIM, hidden_dim=FUSION_HIDDEN)
    model = model.to(DEVICE)
    print("Model created. Params:", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable")

    # ---------------- optimizer / loss / scheduler ----------------
    criterion = nn.MSELoss()    # on log1p(price)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)

    # ---------------- training loop ----------------
    best_val_r2 = -1e9
    best_epoch = -1
    patience = 4
    no_improve = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS} train", ncols=120)
        for imgs, text_feats, targets in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            text_feats = text_feats.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            preds = model(imgs, text_feats)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            pbar.set_postfix(train_loss=np.mean(train_losses))
        scheduler.step()

        # validation
        model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for imgs, text_feats, targets in val_dl:
                imgs = imgs.to(DEVICE, non_blocking=True)
                text_feats = text_feats.to(DEVICE, non_blocking=True)
                preds = model(imgs, text_feats)
                all_preds.append(preds.cpu().numpy())
                all_trues.append(targets.numpy())

        all_preds = np.concatenate(all_preds).ravel()
        all_trues = np.concatenate(all_trues).ravel()
        pred_price = np.expm1(all_preds)
        true_price = np.expm1(all_trues)

        mae = mean_absolute_error(true_price, pred_price)
        rmse = math.sqrt(mean_squared_error(true_price, pred_price))
        r2 = r2_score(true_price, pred_price)
        den = (np.abs(true_price) + np.abs(pred_price))
        den = np.where(den == 0, 1e-8, den)
        smape = np.mean(2.0 * np.abs(pred_price - true_price) / den) * 100

        print(f"Epoch {epoch} VAL -> MAE: {mae:.4f} RMSE: {rmse:.4f} R2: {r2:.4f} SMAPE: {smape:.2f}%")

        if r2 > best_val_r2:
            best_val_r2 = r2
            best_epoch = epoch
            # save full model state_dict
            torch.save(model.state_dict(), FUSION_WEIGHTS)
            print("Saved best fusion weights to:", FUSION_WEIGHTS)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping. No improvement for", patience, "epochs.")
                break

    total_time = time.time() - start_time
    print(f"Training finished in {total_time/60:.2f} minutes. Best val R2: {best_val_r2:.4f} at epoch {best_epoch}")

    # ---------------- Save metadata/scalers ----------------
    joblib.dump(pca, PCA_PATH)
    joblib.dump(scaler, SCALER_PATH)
    if te is not None:
        joblib.dump(te, TARGET_ENCODER_PATH)
    print("Saved PCA, scaler, encoder to outputs.")

    # ---------------- Save predictions on full CSV (optional) ----------------
    print("Generating predictions on full dataset (rows used)...")
    full_ds = MultimodalDataset(df, text_features_scaled, transform=eval_img_transform, device=DEVICE)
    full_dl = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model.load_state_dict(torch.load(FUSION_WEIGHTS, map_location=DEVICE))
    model.eval()
    preds_all = []
    with torch.no_grad():
        for imgs, text_feats, _ in tqdm(full_dl):
            imgs = imgs.to(DEVICE, non_blocking=True)
            text_feats = text_feats.to(DEVICE, non_blocking=True)
            out = model(imgs, text_feats).cpu().numpy().ravel()
            preds_all.append(out)

    preds_all = np.concatenate(preds_all)
    pred_prices = np.expm1(preds_all)

    results_df = pd.DataFrame({
        "sample_id": df.get("sample_id"),
        "predicted_price": pred_prices,
        "actual_price": df["price"]
    })
    out_csv = os.path.join(OUT_DIR, "train_multimodal_predictions.csv")
    results_df.to_csv(out_csv, index=False)
    print("Saved predictions to:", out_csv)

    mae_f = mean_absolute_error(results_df["actual_price"], results_df["predicted_price"])
    rmse_f = math.sqrt(mean_squared_error(results_df["actual_price"], results_df["predicted_price"]))
    r2_f = r2_score(results_df["actual_price"], results_df["predicted_price"])
    den = (np.abs(results_df["actual_price"]) + np.abs(results_df["predicted_price"]))
    den = np.where(den == 0, 1e-8, den)
    smape_f = np.mean(2.0 * np.abs(results_df["predicted_price"] - results_df["actual_price"]) / den) * 100

    print("\nFinal metrics on used rows:")
    print(f"MAE: {mae_f:.4f}, RMSE: {rmse_f:.4f}, R2: {r2_f:.4f}, SMAPE: {smape_f:.2f}%")
    print("All done. Artifacts in:", OUT_DIR)

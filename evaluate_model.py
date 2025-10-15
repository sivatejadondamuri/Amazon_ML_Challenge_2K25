# ============================================================
# MULTI-MODAL PRICE PREDICTION (Text + Tabular + Image CNN)
# ============================================================
# ⚡ Trains everything end-to-end on 5K samples for testing
# ⚙️ GPU supported (MPNet + CNN)
# ============================================================

import os, gc, re, time, random, requests
import numpy as np, pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from category_encoders import TargetEncoder

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = r"d:\student_resource"
CSV_PATH = os.path.join(BASE_DIR, "dataset", "train.csv")
IMG_DIR = os.path.join(BASE_DIR, "all_images")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

SAMPLE_N = 5000
BATCH_SIZE = 64
EPOCHS = 8
IMAGE_SIZE = 224
PCA_DIM = 512
LR = 1e-3
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Using device: {DEVICE}")

# ============================================================
# SEED FIX
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(CSV_PATH)
if SAMPLE_N:
    df = df.sample(SAMPLE_N, random_state=SEED).reset_index(drop=True)
print("📦 Loaded:", df.shape)

# ============================================================
# TEXT CLEANING
# ============================================================
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stop_words = set(stopwords.words('english'))
lemm = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [lemm.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

tqdm.pandas()
df["catalog_content_clean"] = df["catalog_content"].progress_apply(clean_text)

# ============================================================
# IMAGE DOWNLOAD + VALIDATION
# ============================================================
def download_image(sample_id, url):
    path = os.path.join(IMG_DIR, f"{sample_id}.jpg")
    if os.path.exists(path):
        return path
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and len(r.content) > 2000:
            with open(path, "wb") as f:
                f.write(r.content)
            return path
    except Exception:
        return None
    return None

print("📥 Downloading images...")
df["local_image_path"] = [
    download_image(i, u) if isinstance(u, str) else None
    for i, u in tqdm(zip(df.index, df["image_link"]), total=len(df))
]
df = df.dropna(subset=["local_image_path"]).reset_index(drop=True)
print("✅ Valid images:", len(df))

# ============================================================
# TEXT EMBEDDINGS (MPNet)
# ============================================================
model_emb = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
text_emb = model_emb.encode(
    df["catalog_content_clean"].tolist(),
    batch_size=64,
    show_progress_bar=True
)
print("🧠 Embeddings shape:", text_emb.shape)

# PCA reduce
pca = PCA(n_components=PCA_DIM, random_state=SEED)
text_emb_pca = pca.fit_transform(text_emb)

# ============================================================
# TABULAR FEATURES
# ============================================================
exclude_cols = {"catalog_content", "catalog_content_clean", "image_link", "price", "local_image_path"}
tabular_cols = [c for c in df.columns if c not in exclude_cols]
numeric_cols = df[tabular_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in tabular_cols if c not in numeric_cols]

if len(cat_cols) > 0:
    te = TargetEncoder(cols=cat_cols)
    df[cat_cols] = te.fit_transform(df[cat_cols], df["price"])

num_data = df[numeric_cols].fillna(df[numeric_cols].median())
tabular_data = np.hstack([num_data.values, df[cat_cols].values]) if len(cat_cols) else num_data.values
scaler = StandardScaler()
tabular_scaled = scaler.fit_transform(tabular_data)

# combine text + tabular
text_tab_features = np.hstack([text_emb_pca, tabular_scaled])
print("🔗 Text+Tabular feature shape:", text_tab_features.shape)

# ============================================================
# CNN DATASET
# ============================================================
train_df, val_df = train_test_split(df, test_size=0.15, random_state=SEED)
train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

class PriceImageDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["local_image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB",(IMAGE_SIZE,IMAGE_SIZE),(255,255,255))
        img = self.transform(img)
        target = np.log1p(float(row["price"]))
        return img, torch.tensor(target, dtype=torch.float32)

train_ds = PriceImageDataset(train_df, train_transform)
val_ds = PriceImageDataset(val_df, val_transform)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ============================================================
# CNN MODEL
# ============================================================
from torchvision.models import resnet34
cnn_model = resnet34(weights=None)
cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 256)
cnn_model = cnn_model.to(DEVICE)

# ============================================================
# FUSION MODEL (Text+Tabular + CNN)
# ============================================================
class FusionModel(nn.Module):
    def __init__(self, text_dim, img_dim=256):
        super().__init__()
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.img_branch = cnn_model
        self.fc = nn.Sequential(
            nn.Linear(256+256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    def forward(self, img, text_features):
        img_out = self.img_branch(img)
        txt_out = self.text_branch(text_features)
        combined = torch.cat([img_out, txt_out], dim=1)
        out = self.fc(combined)
        return out

fusion_model = FusionModel(text_dim=text_tab_features.shape[1]).to(DEVICE)

optimizer = optim.AdamW(fusion_model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.SmoothL1Loss()
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# ============================================================
# TRAIN FUSION MODEL
# ============================================================
print("🚀 Training Multi-Modal Model...")
best_r2, patience, no_improve = -999, 4, 0
start = time.time()

# use text features aligned with images
X_text_tab = torch.tensor(text_tab_features, dtype=torch.float32).to(DEVICE)
y_all = torch.tensor(np.log1p(df["price"].values), dtype=torch.float32).unsqueeze(1).to(DEVICE)

for epoch in range(1, EPOCHS+1):
    fusion_model.train()
    running_loss = 0.0
    for imgs, targets in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}"):
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            preds = fusion_model(imgs, X_text_tab[:len(imgs)])  # mini alignment
            loss = criterion(preds, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
    train_loss = running_loss / len(train_ds)

    # Validation
    fusion_model.eval()
    val_preds, val_trues = [], []
    with torch.no_grad():
        for imgs, targets in val_dl:
            imgs = imgs.to(DEVICE)
            preds = fusion_model(imgs, X_text_tab[:len(imgs)]).cpu().numpy().ravel()
            val_preds.extend(preds)
            val_trues.extend(targets.numpy().ravel())
    val_preds = np.expm1(np.array(val_preds))
    val_trues = np.expm1(np.array(val_trues))
    mae = mean_absolute_error(val_trues, val_preds)
    rmse = np.sqrt(mean_squared_error(val_trues, val_preds))
    r2 = r2_score(val_trues, val_preds)

    print(f"Epoch {epoch} | TrainLoss: {train_loss:.4f} | MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        torch.save(fusion_model.state_dict(), os.path.join(CHECKPOINT_DIR, "fusion_model_5k.pt"))
        print(f"  ✅ New best R²: {best_r2:.4f}")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("🛑 Early stopping triggered.")
            break
    gc.collect()
    torch.cuda.empty_cache()

print(f"✅ Training finished in {(time.time()-start)/60:.1f} min. Best R²={best_r2:.4f}")

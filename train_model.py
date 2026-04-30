"""
emotiondetection training script

trains a 4-block vgg-style cnn on fer2013 with focal loss, random erasing,
cosine lr schedule with warm-up, and tta-based evaluation. designed to be
run three times with different seeds (42, 123, 456) and ensembled with
ensemble_predict.py for the final reported accuracy.

usage:
    python train_model.py                  # default seed 42
    python train_model.py --seed 123       # second ensemble member
    python train_model.py --seed 456       # third ensemble member

what each run produces in models/:
    emotion_cnn_v5b_seed{N}.keras       trained weights
    class_report_v5b_seed{N}.json       per-class precision/recall/f1
    training_history_v5b_seed{N}.json   per-epoch metrics
    training_curves_v5b_seed{N}.png     train/val accuracy and loss
    confusion_matrix_v5b_seed{N}.png    test-set confusion matrix
    preds_v5b_seed{N}.npy               raw softmax outputs (used by ensemble)
    ytrue_v5b.npy                       ground truth labels (only written once)

note: an earlier iteration combined focal loss with explicit class weights.
that turned out to over-correct (disgust precision crashed) so this version
uses focal loss alone with milder gamma/alpha. see the dissertation for
the full story.
"""

import os, json, argparse, random
import numpy as np

# parse args before importing tf so the seed can be set globally
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--epochs', type=int, default=100, help='max epochs')
args = parser.parse_args()

# set seeds before any tf state is created
SEED = args.seed
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '0'  # full determinism slows xla too much, not worth it here

import tensorflow as tf
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# polite memory growth so tf doesn't grab all the gpu vram on startup
for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except RuntimeError:
        pass

# config
IMG_SIZE   = 48
BATCH      = 64
EPOCHS     = args.epochs
LR_PEAK    = 1e-3
LR_MIN     = 1e-6
WARMUP_EP  = 5
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, f"emotion_cnn_v5b_seed{SEED}.keras")

EMOTIONS = ['angry','disgust','fear','happy','sad','surprise','neutral']
NUM_CLASSES = len(EMOTIONS)

os.makedirs(MODEL_DIR, exist_ok=True)


# focal loss (lin et al. 2017)

def categorical_focal_loss(gamma=1.5, alpha=0.5):
    """multi-class focal loss for softmax outputs.

    gamma controls how aggressively easy examples are down-weighted; alpha
    is a global scale. paper defaults are gamma=2, alpha=0.25 but those
    over-corrected when stacked with class weights in an earlier run, so
    we use gentler values here with no class weights.
    """
    def loss(y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(weight * ce, axis=-1)
    return loss


# random erasing (zhong et al. 2020)

def random_erasing(p=0.4, sl=0.02, sh=0.10, r1=0.3, r2=3.3):
    """returns a function suitable as preprocessing_function for ImageDataGenerator.

    with probability p, blanks out a random rectangle of the image. defaults
    here are dialed back for 48x48 input: max area is 10%, not 20%, because
    larger erases at this resolution wipe out eyes/mouth entirely.
    """
    def _erase(img):
        if np.random.uniform() > p:
            return img
        H, W, C = img.shape
        area = H * W
        for _ in range(10):
            target = np.random.uniform(sl, sh) * area
            ratio  = np.random.uniform(r1, r2)
            h = int(round(np.sqrt(target * ratio)))
            w = int(round(np.sqrt(target / ratio)))
            if h < H and w < W:
                y = np.random.randint(0, H - h)
                x = np.random.randint(0, W - w)
                img[y:y+h, x:x+w, :] = float(img.mean())
                return img
        return img
    return _erase


def get_generators():
    train_aug = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=0.12,
        height_shift_range=0.12,
        zoom_range=0.12,
        validation_split=0.15,
        preprocessing_function=random_erasing(p=0.4),
    )
    test_aug = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(
        'data/train', target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale', class_mode='categorical',
        batch_size=BATCH, subset='training', shuffle=True, seed=SEED)

    val_gen = train_aug.flow_from_directory(
        'data/train', target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale', class_mode='categorical',
        batch_size=BATCH, subset='validation', shuffle=False, seed=SEED)

    test_gen = test_aug.flow_from_directory(
        'data/test', target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale', class_mode='categorical',
        batch_size=BATCH, shuffle=False)

    return train_gen, val_gen, test_gen


# model architecture
# 4 vgg-style blocks: 32 -> 64 -> 192 -> 384, then a 512/256 dense head.
# total ~3.5m params. last conv layer is the grad-cam target.

def build_model():
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # block 1: 32 -> 32
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x); x = layers.Dropout(0.25)(x)

    # block 2: 64 -> 64
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x); x = layers.Dropout(0.25)(x)

    # block 3: 192 -> 192 (widened from the original 128)
    x = layers.Conv2D(192, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(192, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x); x = layers.Dropout(0.30)(x)

    # block 4: 384 (widened from the original 256). single conv here, target for grad-cam.
    x = layers.Conv2D(384, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x); x = layers.Dropout(0.30)(x)

    # head
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    return models.Model(inp, out)


# cosine lr schedule with linear warm-up

def cosine_warmup_schedule(epoch, lr):
    if epoch < WARMUP_EP:
        return LR_PEAK * (epoch + 1) / WARMUP_EP
    progress = (epoch - WARMUP_EP) / max(1, EPOCHS - WARMUP_EP)
    return LR_MIN + 0.5 * (LR_PEAK - LR_MIN) * (1 + np.cos(np.pi * progress))


# training

def train(model, train_gen, val_gen):
    cb = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=20,
                                restore_best_weights=True, verbose=1),
        callbacks.LearningRateScheduler(cosine_warmup_schedule, verbose=0),
        callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                                  save_best_only=True, verbose=1),
    ]
    return model.fit(train_gen, validation_data=val_gen,
                     epochs=EPOCHS, callbacks=cb, verbose=1)


# evaluation with test-time augmentation
# tta = predict on the original frame and on its horizontal flip, average the
# two softmax outputs. cheap, deterministic, gives ~1pp accuracy for free.

def evaluate_with_tta(model, test_gen):
    print(f"\n[eval] running with tta (seed={SEED})...")
    test_gen.reset()

    preds_orig = model.predict(test_gen, verbose=0)
    test_gen.reset()

    flipped_preds = []
    for i in range(len(test_gen)):
        batch_x, _ = test_gen[i]
        batch_flipped = batch_x[:, :, ::-1, :]   # horizontal flip on width axis
        flipped_preds.append(model.predict(batch_flipped, verbose=0))
    preds_flip = np.concatenate(flipped_preds, axis=0)

    n = min(len(preds_orig), len(preds_flip))
    preds_avg = (preds_orig[:n] + preds_flip[:n]) / 2.0
    y_pred = np.argmax(preds_avg, axis=1)
    y_true = test_gen.classes[:n]

    acc = float(np.mean(y_pred == y_true))
    print(f"[eval] tta test accuracy: {acc*100:.2f}%")

    class_names = [k for k, v in sorted(test_gen.class_indices.items(), key=lambda x: x[1])]
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    rpt = classification_report(y_true, y_pred, target_names=class_names,
                                zero_division=0, output_dict=True)
    rpt['tta'] = True
    rpt['seed'] = SEED
    with open(os.path.join(MODEL_DIR, f'class_report_v5b_seed{SEED}.json'), 'w') as f:
        json.dump(rpt, f, indent=2)

    # save raw softmax outputs and ground truth so ensemble_predict.py can use them
    np.save(os.path.join(MODEL_DIR, f'preds_v5b_seed{SEED}.npy'), preds_avg[:n])
    np.save(os.path.join(MODEL_DIR, f'ytrue_v5b.npy'), y_true)

    return acc, y_true, y_pred, class_names


def save_plots(history, acc, y_true, y_pred, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'EmotionDetection v5b (seed={SEED}) - Training Results', fontsize=14, fontweight='bold')
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val', linewidth=2, linestyle='--')
    axes[0].axhline(acc, color='red', linestyle=':', label=f'Test+TTA: {acc:.1%}')
    axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history.history['loss'], label='Train', linewidth=2, color='orange')
    axes[1].plot(history.history['val_loss'], label='Val', linewidth=2, linestyle='--', color='red')
    axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f'training_curves_v5b_seed{SEED}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Normalised confusion matrix (v5b seed={SEED} with TTA)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f'confusion_matrix_v5b_seed{SEED}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist['test_accuracy_tta'] = float(acc)
    hist['seed'] = SEED
    with open(os.path.join(MODEL_DIR, f'training_history_v5b_seed{SEED}.json'), 'w') as f:
        json.dump(hist, f, indent=2)


def main():
    print("=" * 55)
    print(f"  emotiondetection - training (seed={SEED})")
    print("=" * 55)
    print(f"  tensorflow: {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"  gpu: {bool(gpus)}  ({len(gpus)} device(s))")
    print("=" * 55 + "\n")

    if not os.path.isdir('data/train') or not os.path.isdir('data/test'):
        print("[error] data/train or data/test not found. see data/README.md")
        return

    train_gen, val_gen, test_gen = get_generators()

    label_map = {str(v): k for k, v in train_gen.class_indices.items()}
    with open(os.path.join(MODEL_DIR, 'class_labels.json'), 'w') as f:
        json.dump(label_map, f, indent=2)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_PEAK),
        loss=categorical_focal_loss(gamma=1.5, alpha=0.5),
        metrics=['accuracy'])
    print(f"\n[model] total params: {model.count_params():,}")
    print(f"[model] saving best to: {MODEL_PATH}")

    history = train(model, train_gen, val_gen)
    acc, y_true, y_pred, class_names = evaluate_with_tta(model, test_gen)
    save_plots(history, acc, y_true, y_pred, class_names)

    print("\n" + "=" * 55)
    print(f"  done (seed={SEED}) - tta test accuracy: {acc*100:.2f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()

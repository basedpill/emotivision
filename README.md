# EmotionDetection

real-time facial emotion recognition from webcam video, with a live grad-cam attention overlay. custom cnn trained from scratch on fer2013, deployed as a flask web app.

## quick start

```bash
pip install -r requirements.txt

# put fer2013 at data/train/<class>/ and data/test/<class>/ (see data/README.md)
python train_model.py --seed 42
python train_model.py --seed 123
python train_model.py --seed 456

# combine the three runs into the ensemble
python ensemble_predict.py

# run the web app (loads the deployed model in models/emotion_cnn.keras)
python webapp/app.py
# then open http://localhost:5000
```

debug mode: `FLASK_DEBUG=true python webapp/app.py`

## tests

```bash
pip install pytest
pytest tests/
```

## folder structure

```
data/
  train/   angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
  test/    same structure
models/
  emotion_cnn.keras                   model used by the webapp (seed 123)
  emotion_cnn_v5b_seed{42,123,456}.keras   per-seed training outputs
  class_labels.json                   index -> class name mapping
  class_report.json                   ensemble per-class precision/recall/f1
  class_report_v5b_seed{N}.json       per-seed reports
  training_history.json               per-epoch metrics for the deployed model
  training_curves.png                 train/val accuracy + loss
  confusion_matrix.png                ensemble confusion matrix
  preds_v5b_seed{N}.npy               raw softmax outputs (used by ensemble)
  ytrue_v5b.npy                       ground truth labels for the test set
webapp/
  app.py                              flask server: /, /analyze, /gradcam, /health, /stats
  templates/index.html                single-page frontend
tests/
  test_app.py                         pytest suite for routes + utilities
train_model.py                        full training pipeline
ensemble_predict.py                   averages the per-seed predictions
```

## results

3-model ensemble with test-time augmentation on fer2013 (7178-image combined test split):

| metric           | value |
|------------------|------:|
| test accuracy    | 67.6% |
| macro f1         |  0.63 |
| weighted f1      |  0.67 |

per-class:

| emotion  | precision | recall | f1   |
|----------|----------:|-------:|-----:|
| happy    |      0.88 |   0.89 | 0.89 |
| surprise |      0.76 |   0.82 | 0.79 |
| neutral  |      0.55 |   0.77 | 0.64 |
| angry    |      0.60 |   0.66 | 0.63 |
| sad      |      0.57 |   0.52 | 0.54 |
| disgust  |      0.71 |   0.42 | 0.53 |
| fear     |      0.63 |   0.30 | 0.41 |

per-seed individual accuracies (with TTA):
- seed 42:  66.19%
- seed 123: 67.01%   <-- this is the deployed single model
- seed 456: 66.97%

## training notes

- 4-block vgg-style cnn, ~3.5m params per ensemble member
- focal loss with gamma=1.5, alpha=0.5 (no class weights, those over-corrected when stacked with focal)
- adam, lr 1e-3 with cosine annealing and 5-epoch warm-up
- random erasing (p=0.4, max area 10% of image)
- horizontal flip + small geometric augmentation
- 100 epochs max with patience-20 early stopping on val accuracy
- ~25 minutes per seed on an rtx 3050 laptop gpu

## hardware

trained and developed on:
- nvidia rtx 3050 laptop gpu (4 gb vram)
- ubuntu 22.04 in wsl2 on windows
- tensorflow 2.21 with bundled cuda 12.x

## dataset 

this repo does not include the FER2013 dataset itself (/data is gitignored). FER2013 was released by Pierre-Luc Carrier and Aaron Courville as part of the ICML 2013 Challenges in Representation Learning, and is commonly distributed via Kaggle. check the current Kaggle listing for FER2013's license and usage terms before redistributing it or using it commercially as this repo only assumes you'll source it yourself and drop it into data/train and data/test.

import math, re, glob, os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tf2onnx
from sklearn.metrics import classification_report, f1_score
import random

# --------------------------------------
# FIJAR SEMILLAS PARA REPRODUCIBILIDAD
# --------------------------------------
SEED = 42

import random
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# DATA_PATH prepara para leer imágenes de 224x224 del conjunto de datos
DATA_PATH = "/PATH_TO_DATA/flower-classification-with-tpus/tfrecords-jpeg-224x224"
IMAGE_SIZE = [224, 224]
BATCH_SIZE = 24
EPOCHS = 8
AUTO = tf.data.AUTOTUNE

# -----------------------------
# TFRecords
# -----------------------------

TRAINING_FILENAMES = glob.glob(f"{DATA_PATH}/train/*.tfrec")
VALIDATION_FILENAMES = glob.glob(f"{DATA_PATH}/val/*.tfrec")
TEST_FILENAMES = glob.glob(f"{DATA_PATH}/test/*.tfrec")

# -----------------------------
# Clases
# -----------------------------

CLASSES = [
    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'wild geranium',
    'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle',
    'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower',
    'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily',
    'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy',
    'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william',
    'carnation', 'garden phlox', 'love in the mist', 'cosmos', 'alpine sea holly', 'ruby-lipped cattleya',
    'cape flower', 'great masterwort', 'siam tulip', 'lenten rose',
    'barberton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower',
    'marigold', 'buttercup', 'daisy', 'common dandelion',
    'petunia', 'wild pansy', 'primula', 'sunflower', 'lilac hibiscus', 'bishop of llandaff',
    'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia',
    'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy',
    'osteospermum', 'spring crocus', 'iris', 'windflower', 'tree poppy',
    'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower',
    'lotus', 'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine',
    'desert-rose', 'tree mallow', 'magnolia', 'cyclamen', 'watercress', 'canna lily', 'hippeastrum',
    'bee balm', 'pink quill', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia',
    'bromelia', 'blanket flower', 'trumpet creeper', 'blackberry lily', 'common tulip', 'wild rose'
]

# -------------------------------------------
# Normalización y uso del conjunto de datos
# -------------------------------------------

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])          
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "class": tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "id": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum

def load_dataset(filenames, labeled=True, ordered=False):
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    if not ordered:
        options = tf.data.Options()
        options.experimental_deterministic = True
        dataset = dataset.with_options(options)
    if labeled:
        dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else:
        dataset = dataset.map(read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    return dataset

def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label

def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat().shuffle(2048, seed=SEED).batch(BATCH_SIZE).prefetch(AUTO)
    return dataset

def get_validation_dataset():
    dataset = load_dataset(VALIDATION_FILENAMES)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTO)
    return dataset

def get_test_dataset():
    dataset = load_dataset(TEST_FILENAMES, labeled=False)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTO)
    return dataset

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

# -----------------------------
# Count images and steps
# -----------------------------

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
VALIDATION_STEPS = -(-NUM_VALIDATION_IMAGES // BATCH_SIZE)  # round up
TEST_STEPS = -(-NUM_TEST_IMAGES // BATCH_SIZE)

print(f'Dataset: {NUM_TRAINING_IMAGES} training, {NUM_VALIDATION_IMAGES} validation, {NUM_TEST_IMAGES} test')
print(f'Steps per epoch: {STEPS_PER_EPOCH}, Validation steps: {VALIDATION_STEPS}')

# data dump
print("Training data shapes:")
for image, label in get_training_dataset().take(3):
    print(image.numpy().shape, label.numpy().shape)
print("Training data label examples:", label.numpy())

print("Validation data shapes:")
for image, label in get_validation_dataset().take(3):
    print(image.numpy().shape, label.numpy().shape)
print("Validation data label examples:", label.numpy())

print("Test data shapes:")
for image, idnum in get_test_dataset().take(3):
    print(image.numpy().shape, idnum.numpy().shape)
print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string

#-----------------------------
# Modelo base EfficientNetB0
#-----------------------------

# Carga el modelo preentrenado EfficientNetB0 con pesos de ImageNet
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False # Congela los pesos

# Construye una nueva capa secuencial encima del modelo base
model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(CLASSES), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(get_training_dataset(),
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=EPOCHS,
                    validation_data=get_validation_dataset(),
                    validation_steps=VALIDATION_STEPS)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

#--------------------
# Precisión
#--------------------

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', marker='.')
plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='.')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

#--------------------
# Perdida
#--------------------

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', marker='.')
plt.plot(epochs_range, val_loss, label='Validation Loss', marker='.')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plt.show()

#----------
# F1 score
#----------

y_true = []
y_pred = []

val_dataset = get_validation_dataset()

for images, labels in val_dataset:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(labels.numpy())

print("\n=== Macro F1 score evaluation ===")
print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))

f1_macro = f1_score(y_true, y_pred, average='macro')
print(f"F1 Score (macro): {f1_macro:.4f}")

# -------------------------------------
# Guarda el modelo en el estandar ONNX
# -------------------------------------

os.makedirs("dir_exit", exist_ok=True)

route_keras = os.path.join("dir_exit", "flowers.keras")
model.save(route_keras)

input_signature = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
route_onnx = os.path.join("dir_exit", "flowers.onnx")

model_keras = tf.keras.models.load_model(route_keras)

model_keras.output_names = ["output"]

model_proto, _ = tf2onnx.convert.from_keras(model_keras,
                                            input_signature=input_signature,
                                            output_path=route_onnx)

print(f"Modelo guardado en dir_exit:\n- Keras: {route_keras}\n- ONNX: {route_onnx}")

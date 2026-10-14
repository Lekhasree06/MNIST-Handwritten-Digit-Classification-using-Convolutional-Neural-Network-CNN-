import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# ---------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# ---------------------------------------------------------
# 3. Visualize Samples
# ---------------------------------------------------------
plt.figure(figsize=(8, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i].squeeze(), cmap="gray")
    plt.title(f"Label: {y_train[i]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 4. Build CNN
# ---------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.30),

    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------------------------------------
# 5. Train
# ---------------------------------------------------------
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# ---------------------------------------------------------
# 6. Evaluate
# ---------------------------------------------------------
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# ---------------------------------------------------------
# 7. Training Curves
# ---------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 8. Confusion Matrix
# ---------------------------------------------------------
predictions = model.predict(x_test, verbose=0)
y_pred = np.argmax(predictions, axis=1)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=np.arange(10)
)

fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("MNIST CNN Confusion Matrix")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 9. Sample Predictions
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")

    predicted = y_pred[i]
    actual = y_test[i]

    plt.title(f"Actual: {actual} | Pred: {predicted}")
    plt.axis("off")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 10. Save Model
# ---------------------------------------------------------
model.save("mnist_cnn.keras")
print("Model saved as mnist_cnn.keras")

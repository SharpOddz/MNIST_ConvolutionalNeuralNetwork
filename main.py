import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#Setting random seed for reproducibility
tf.random.set_seed(23)

#Loading in MNIST dataset
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#Preprocessing: Normalization and Reshaping to (28, 28, 1)
x_train_full = x_train_full.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

#Train/val split
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.1, random_state=23, shuffle=True
)

print(f'Training data shape: {x_train.shape}')
print(f'Validation data shape: {x_val.shape}')

#Model hyperparameters
learning_rate = 0.001
epochs = 20
batch_size = 32
dropout_rate = 0.2
reg_type = 'None'

#Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)

#Model Architecture
model = tf.keras.models.Sequential([
    #Convolutional Block 1
    layers.Conv2D(8, (3, 3), input_shape=(28, 28, 1), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    #Convolutional Block 2
    layers.Conv2D(16, (3, 3), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    #Fully Connected Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(dropout_rate),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

#Progress Bar + history tracking + callbacks
history = model.fit(
    x_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, lr_reduction],
    verbose=1
)

#Evaluation
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print(f'\nRestored Model Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
print(f'Restored Model Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')

#Calculate best epoch for plotting
best_epoch = np.argmin(history.history['val_loss'])

#Loss and accuracy plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch + 1}')
axes[0].set_title('Sparse Categorical Cross Entropy (SCCE) Loss vs Epoch')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('SCCE Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['accuracy'], label='Training Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch + 1}')
axes[1].set_title('Accuracy vs Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

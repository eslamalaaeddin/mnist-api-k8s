# train_model.py

import tensorflow as tf

def train_and_save_model(model_path='mnist_model.h5'):
    """Trains a simple Keras model on MNIST and saves it."""
    print("INFO: Starting MNIST model training...")
    
    # 1. تحميل البيانات
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 2. تطبيع البيانات (Normalization)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 3. بناء الموديل (Simple Sequential Model)
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10)
    ])

    # 4. التجميع والتدريب
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    # التدريب على 5 Epochs فقط للسرعة
    model.fit(x_train, y_train, epochs=5, verbose=0) 
    
    # 5. حفظ الموديل
    model.save(model_path)
    print(f"INFO: Model saved successfully to {model_path}")
    print("INFO: Training complete.")

if __name__ == '__main__':
    train_and_save_model()
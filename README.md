# Redes-Neurais
Redes-Neurais
Preparação do Ambiente

Passos iniciais no Google Colab:

Abra o Colab: https://colab.research.google.com

Crie um novo notebook.

Importe as bibliotecas principais:

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os


Faça o upload ou download do seu dataset.

Se usar gatos e cachorros, baixe o dataset oficial.

Caso queira usar seu próprio dataset, organize as imagens em pastas:

dataset/
    train/
        classe1/
        classe2/
    validation/
        classe1/
        classe2/

3. Transfer Learning – Passo a Passo

Carregar o modelo pré-treinado

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # Congela os pesos para não treinar novamente


Adicionar camadas finais (head)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)  # 2 classes: gato e cachorro

model = Model(inputs=base_model.input, outputs=predictions)


Compilar o modelo

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


Gerar dados de treinamento e validação

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('dataset/train', target_size=(224,224), batch_size=32, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory('dataset/validation', target_size=(224,224), batch_size=32, class_mode='categorical')


Treinar o modelo

history = model.fit(train_generator, validation_data=validation_generator, epochs=10)


Salvar modelo treinado (opcional)

model.save('modelo_transfer_learning.h5')


Visualizar resultados

plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

4. Documentação no GitHub

No repositório público, organize os arquivos assim:

meu-projeto-transfer-learning/
│
├─ README.md          # Descrição completa do projeto
├─ transfer-learning.ipynb  # Notebook do Colab
├─ /images            # Capturas de tela do treinamento ou resultados
└─ dataset/           # (Opcional se dataset for pequeno; ou link para download)


Dicas para o README.md:

Explique o que é Transfer Learning

Descreva seu dataset

Documente cada passo do notebook

Inclua resultados e gráficos

Explique possíveis melhorias ou experimentos futuros

# -*- coding: utf-8 -*-

"""
eng : we are loading libraries
tr : Kütüphaneleri yüklüyoruz
"""
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt

# eng : How many classes we have, we find this.
# tr : Kaç tane sınıfımız var bunu buluyoruz.
from glob import glob

"""
eng : train - test path
tr : train - test dosyalarımızın olduğu dosya yollarımız
"""
train_path = "fruits-360/Training/" 
test_path = "fruits-360/Test/"

"""
eng : We visualize one of the pictures.
tr : Resimlerden bir tanesini görselleştiriyoruz.
"""
img = load_img(train_path + "Apple Braeburn/0_100.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()

"""
eng : from picture to series
tr : resmi diziye çeviriyoruz
"""
x = img_to_array(img) 
print(x.shape) 

"""
eng : we find out how many classes there are
tr : kaç tane sınıfımız olduğunu buluyoruz
"""
className = glob(train_path + '/*') 
numberOfClass = len(className)
print("NumberOfClass: ",numberOfClass)

#%% 
"""
eng : create CNN model
tr : cnn modelimizi oluşturuyoruz
"""

# eng : 32 => number of filters
# tr : 32 => filtre sayısı
model = Sequential()
model.add(Conv2D(32,(3, 3),input_shape = x.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3, 3))) 
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024)) 
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(numberOfClass)) # output 
model.add(Activation("softmax"))


model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])

"""
eng : We specify how many images we will use each time
tr : Her seferde kaç resim kullanacağımızı beliritoyruz
"""
batch_size = 32 

#%%
"""
eng : We are increasing the number of pictures for education.
tr : Eğitim için resim sayımızı çoğaltıyoruz.
"""
train_datagen = ImageDataGenerator(rescale= 1./255,
                   shear_range = 0.3, 
                   horizontal_flip=True, 
                   zoom_range = 0.3) 


test_datagen = ImageDataGenerator(rescale= 1./255) 

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size = x.shape[:2], 
        color_mode= "rgb",
        class_mode= "categorical")

test_generator = test_datagen.flow_from_directory(
        test_path, 
        target_size=x.shape[:2],
        batch_size = batch_size,
        color_mode= "rgb",
        class_mode= "categorical")

hist = model.fit_generator(
        generator = train_generator,
        steps_per_epoch = 1600 // batch_size,
        epochs=100,
        validation_data = test_generator,
        validation_steps = 800 // batch_size) 


#%% 
"""
eng : model evaluation
tr : sonuçlarımızı değerlendiriyoruz.
"""
print(hist.history.keys())
plt.plot(hist.history["loss"], label = "Train Loss")
plt.plot(hist.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"], label = "Train acc")
plt.plot(hist.history["val_accuracy"], label = "Validation acc")
plt.legend()
plt.show()











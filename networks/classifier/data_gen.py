from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input

def create_datagen(train_folder, valid_folder, batch_size, input_size, filename):
    
    if not valid_folder:
        train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.1)
        train_generator=train_datagen.flow_from_directory(train_folder,
                                                         target_size=(input_size,input_size),
                                                         color_mode='rgb',
                                                         batch_size=batch_size,
                                                         class_mode='categorical', 
				                                         shuffle=True,
				                                         subset='training')

        validation_generator=train_datagen.flow_from_directory(train_folder,
                                                         target_size=(input_size,input_size),
                                                         color_mode='rgb',
                                                         batch_size=batch_size,
                                                         class_mode='categorical', 
				                                         shuffle=True,
				                                         subset='validation')
    else:
        train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
        train_generator=train_datagen.flow_from_directory(train_folder,
                                                         target_size=(input_size,input_size),
                                                         color_mode='rgb',
                                                         batch_size=batch_size,
                                                         class_mode='categorical', 
				                                         shuffle=True)

        validation_generator=train_datagen.flow_from_directory(valid_folder,
                                                         target_size=(input_size,input_size),
                                                         color_mode='rgb',
                                                         batch_size=batch_size,
                                                         class_mode='categorical', 
				                                         shuffle=True)				                                     
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    fo = open("labels_101.txt", "w")
    for k,v in labels.items():
        print(v)
        fo.write(v+"\n")
    fo.close()
    return train_generator, validation_generator

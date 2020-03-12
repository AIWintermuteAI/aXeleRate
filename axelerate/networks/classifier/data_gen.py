from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
import os


def create_datagen(train_folder, valid_folder, batch_size, input_size, project_folder, augumentation):
    if augumentation:
        data_gen_args = dict(brightness_range=[0.5,1.5],
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
    else:
        data_gen_args = {}


    
    if not valid_folder:
        train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,validation_split=0.1, **data_gen_args)
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
        train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input, **data_gen_args)
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
    fo = open(os.path.join(project_folder,"labels.txt"), "w")
    for k,v in labels.items():
        print(v)
        fo.write(v+"\n")
    fo.close()
    return train_generator, validation_generator

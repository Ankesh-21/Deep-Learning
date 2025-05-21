# CNN on Classification cat and dog
### Some Steps for Data Import in Colab from Kaggle
- create a api key from profile
- in dataset get the data link

## Download the dataset
```
! mkdir -p ~/.kaggle
! cp kaggle.json ~/.kaggle/
! kaggle datasets download -d salader/cats-vs-dogs
```
## unzipping the zipped datasets
```
import zipfile
```
by zipfile library

## Importing all required library

- tensorflow 
- keras from tensorflow
- Sequential from keras 
- Dense,Conv2D,Maxpooling2D,Flatten from keras.layers

## Generators , label the image 
```
train_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/train',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size = (256,256)
)
```
* In val_ds samely we just change path of directory to ``` '/content/test' ```

## Normalization
- every image here is stored as a numpy array
- the value of the numpy array in range(0 , 255)
- but we need to make it in range(0 , 1)
- to normalize we use:
    - take image and it's label as parameters and then cast it as ``` image = tf.cast(image/255.,tf.float32)```
    - means we divide all the value inside of the numpy array by 255
    - Then replace the modified image back to the original location
    - ``` train_ds = train_ds.map(process)```
    - Here Process is the function take an image and a label and return modified image and label
    - map function pick every image from train dataset and validation dataset and apply the process function on it
    - ``` val_ds = val_ds.map(process)```
## 
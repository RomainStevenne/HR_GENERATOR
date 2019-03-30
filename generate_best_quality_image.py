# for load the image (must be in rgb format)
from PIL import Image

# for creating the NNss
from tensorflow.keras.layers import Activation, BatchNormalization, Input, Flatten, Dense
from tensorflow.keras.layers import UpSampling2D, Conv2D, add, multiply
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# for the data processing
import numpy as np

# the files path
INPUT_FILE = "file.jpg"
OUTPUT_FILE = "newfile.jpg"
NN_FILE = "_gnerator.h5"

# the other vars
WINDOWS_OVERLAPPING = 4
WINDOWS_BORDER = 4
WINDOWS_SIZES = 200

# the generator creation
# the input layer (who extract the data from image)
# one convulution form 3 chanels to 64 chanels
input_layer = Input(shape=(None, None, 3))
generator = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(input_layer)
generator = LeakyReLU(alpha=.1)(generator)

gene_model = generator

# the SPAED blocks (who treat the data from the image to extract the most usefull featurs)
for i in range(16):
    model = generator

    generator = BatchNormalization(momentum = 0.5)(generator)

    model = Conv2D(filters = 32, kernel_size = 9, strides = 1, padding = "same")(input_layer)
    model = LeakyReLU(alpha=.1)(model)

    gamma = Conv2D(filters = 64, kernel_size = 5, strides = 1, padding = "same")(model)
    gamma = LeakyReLU(alpha=.1)(gamma)

    beta = Conv2D(filters = 64, kernel_size = 5, strides = 1, padding = "same")(model)
    beta = LeakyReLU(alpha=.1)(beta)

    generator = multiply([generator, gamma])
    generator = add([generator, beta])

# the end of the residual blocks
# one convolution + add the image befor residuals blocks
generator = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(generator)
generator = BatchNormalization(momentum = 0.5)(generator)
generator = add([gene_model, generator])

# the upsampling blocks (who upscale the image by 2)
# one convolution + upscalling
for i in range(2):
    generator = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = "same")(generator)
    generator = UpSampling2D(size = 2)(generator)
    generator = LeakyReLU(alpha = 0.2)(generator)

# the final layer (who creat the new image)
# one convolution from 64 chanels to 3 chanels
generator = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(generator)
generator = Activation('sigmoid')(generator)

generator = Conv2D(filters = 32, kernel_size = 9, strides = 1, padding = "same")(generator)
generator = LeakyReLU(alpha = .2)(generator)

generator = Conv2D(filters = 64, kernel_size = 1, strides = 1, padding = "same")(generator)
generator = LeakyReLU(alpha = .2)(generator)

generator = Conv2D(filters = 3, kernel_size = 5, strides = 1, padding = "same")(generator)
generator = Activation("sigmoid")(generator)

# end of the graph creation and loading of the weights
generator = Model(inputs = input_layer, outputs = generator)
generator.load_weights("_generator.h5")

# load the image file and reshape color value between 0 and 1
image = Image.open(INPUT_FILE)
print(f"Image format = {image.getbands()}")
image = np.array(image).astype("float32") / 255

# extract the images data for th windows crations
x_size, y_size, _ = image.shape
new_size = x_size * 4, y_size * 4

# caluclate the windows number
x_size //= WINDOWS_SIZES
y_size //= WINDOWS_SIZES
x_size += 1
y_size += 1

# generate an empty new image with the size
# that the new image will have
new_image = []
for i in range(new_size[0]):
    col = [None,] * new_size[1]
    new_image.append(col)

# the loops for pass on all windows
for i in range(x_size):
    for j in range(y_size):
        # define the windows limite
        # on the x axe
        if i == 0:
            begin_border_x = 0
            begin_overlapping_x = 0
            begin_true_x = 0

            end_border_x = WINDOWS_SIZES * (i+1) + WINDOWS_OVERLAPPING + WINDOWS_BORDER
            end_overlapping_x = WINDOWS_SIZES * (i+1) + WINDOWS_OVERLAPPING
            end_true_x = WINDOWS_SIZES * (i+1)

        elif i != x_size -1:
            begin_border_x = WINDOWS_SIZES * (i) - WINDOWS_OVERLAPPING - WINDOWS_BORDER
            begin_overlapping_x = WINDOWS_SIZES * (i) - WINDOWS_OVERLAPPING
            begin_true_x = WINDOWS_SIZES * (i)

            end_border_x = WINDOWS_SIZES * (i+1) + WINDOWS_OVERLAPPING + WINDOWS_BORDER
            end_overlapping_x = WINDOWS_SIZES * (i+1) + WINDOWS_OVERLAPPING
            end_true_x = WINDOWS_SIZES * (i+1)

        else:
            begin_border_x = WINDOWS_SIZES * (i) - WINDOWS_OVERLAPPING - WINDOWS_BORDER
            begin_overlapping_x = WINDOWS_SIZES * (i) - WINDOWS_OVERLAPPING
            begin_true_x = WINDOWS_SIZES * (i)

            end_border_x = new_size[0] // 4
            end_overlapping_x = new_size[0] // 4
            end_true_x = new_size[0] // 4

        # on the y axe
        if j == 0:
            begin_border_y = 0
            begin_overlapping_y = 0
            begin_true_y = 0

            end_border_y = WINDOWS_SIZES * (j+1) + WINDOWS_OVERLAPPING + WINDOWS_BORDER
            end_overlapping_y = WINDOWS_SIZES * (j+1) + WINDOWS_OVERLAPPING
            end_true_y = WINDOWS_SIZES * (j+1)

        elif j != y_size -1:
            begin_border_y = WINDOWS_SIZES * (j) - WINDOWS_OVERLAPPING - WINDOWS_BORDER
            begin_overlapping_y = WINDOWS_SIZES * (j) - WINDOWS_OVERLAPPING
            begin_true_y = WINDOWS_SIZES * (j)

            end_border_y = WINDOWS_SIZES * (j+1) + WINDOWS_OVERLAPPING + WINDOWS_BORDER
            end_overlapping_y = WINDOWS_SIZES * (j+1) + WINDOWS_OVERLAPPING
            end_true_y = WINDOWS_SIZES * (j+1)

        else:
            begin_border_y = WINDOWS_SIZES * (j) - WINDOWS_OVERLAPPING - WINDOWS_BORDER
            begin_overlapping_y = WINDOWS_SIZES * (j) - WINDOWS_OVERLAPPING
            begin_true_y = WINDOWS_SIZES * (j)

            end_border_y = new_size[1] // 4
            end_overlapping_y = new_size[1] // 4
            end_true_y = new_size[1] // 4

        # generate the window
        # based on the data
        # generate before
        window = []

        # cut the window on x
        tmp = image[begin_border_x:end_border_x]
        # cut the window on y
        for t in tmp:
            t = t[begin_border_y:end_border_y]
            # cut the pixels alpha
            col = []
            for pixel in t:
                col.append(np.array([pixel[0], pixel[1], pixel[2]]))
            window.append(col)

        # log
        print(f"Epochs: {i * y_size + j + 1}/{y_size*x_size}; window: [{begin_true_x}, {begin_true_y}] [{end_true_x}, {end_true_y}]; shape: {np.array(window).shape}")

        # predict the new window
        new_window = generator.predict(np.array([window,]))[0]

        # pass throug all pixels of the new_window
        # and push there in the new image

        # the loop
        for k in range(len(new_window)):
            for l in range(len(new_window[0])):

                # calculate the coords
                # on the new image
                x = k + (begin_border_x*4)
                y = l + (begin_border_y*4)

                # get the pixel value
                p = new_window[k][l]
                pixel = np.array([int(p[0]*255), int(p[1]*255), int(p[2]*255)], dtype="uint8")

                # check if the value is on the overlapping
                # interval
                if begin_overlapping_x*4 <= x < end_overlapping_x*4 and \
                    begin_overlapping_y*4 <= y < end_overlapping_y*4:

                    new_image[x][y] = pixel
                    """# check if the pixel is define
                    if new_image[x][y] is not None:
                         # calculate the middle
                         # values between the tow
                         # pixel
                         prev_pix = new_image[x][y]
                         new_pix = prev_pix + pixel
                         new_pix //= 2
                         new_image[x][y] = new_pix
                    else:
                        new_image[x][y] = pixel"""

new = np.array(new_image)
new  = Image.fromarray(new)
new.save(OUTPUT_FILE)

print("save in ", OUTPUT_FILE)

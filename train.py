import csv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Cropping2D, Lambda
import scipy.ndimage
from scipy.misc import pilutil

DATA_FILE_PREFIX = "data/"
SIDE_CAMERA_OFFSET = 0.25


def add_data_prefix(file_name):
    return DATA_FILE_PREFIX + file_name


def load_data(file_path):
    csv_file = open(file_path)
    csv_reader = csv.reader(csv_file)
    image_names = []
    steering_angles = []
    for line in csv_reader:
        if(line[0] == "center"):
            continue    # this is the first line (legend)
        steering_angle = float(line[3])
        center_name = add_data_prefix(line[0])
        left_name = add_data_prefix(line[1])
        right_name = add_data_prefix(line[2])
        image_names.append(center_name)
        steering_angles.append(steering_angle)
        image_names.append(left_name)
        steering_angles.append(steering_angle + SIDE_CAMERA_OFFSET)
        image_names.append(right_name)
        steering_angles.append(steering_angle - SIDE_CAMERA_OFFSET)
    return image_names, steering_angles

def network():
    model = Sequential()
    # removing top and bottom pixels from image
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
    # normalizing
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

if __name__ == "__main__":
    original_image_names, original_steering_angles = load_data("data/driving_log.csv")
#    image = scipy.ndimage.imread(original_image_names[0])
#    pilutil.imshow(image)
#    print(original_image_names[0])
#    print(original_steering_angles[0])

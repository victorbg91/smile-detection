"""
Author: Victor BG
Date: August 3rd, 2019

This script intends to train a smile-recognition neural network.
It first collects its own dataset by requesting a series of pictures with smiles and without smiles.
It then trains a neural network to recognize a smile.
Finally, it launches the webcam and shows a smiley face in the top left corner when a smile is detected.

REFERENCES
List of databases (it was initially intended to use an existing database):
http://www.face-rec.org/databases/
Database 1 (unused - no facial expression change):
http://cbcl.mit.edu/software-datasets/heisele/facerecognition-database.html
Database 2 (unused - they are not labeled):
https://cswww.essex.ac.uk/mv/allfaces/index.html
---> I will collect my own data

For face detection using Haar Cascade:
https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html
Haar Cascade configuration file downloaded from:
https://github.com/opencv/opencv/blob/3.4/data/haarcascades/haarcascade_frontalface_default.xml
"""

# ------- #
# IMPORTS #
# ------- #
import tensorflow as tf
import numpy as np
import glob
import cv2
import os


# ----------------- #
# CREATING DATA SET #
# ----------------- #
def create_data(num_expression, series_name, folder="./data/"):
    # Checking for conflict with series name
    list_conflict = glob.glob(folder + series_name + "*.png")
    if len(list_conflict) > 0:
        answer = input("Conflict detected with this series name.\nDo you want to delete "\
                           +"{} items? [y/N]".format(len(list_conflict)))

        # Delete the conflicting files if 'y' or 'Y' is inputed
        if answer=='y' or answer=='Y':
            for f in list_conflict:
                os.remove(f)
            os.remove(folder + series_name + "_labels.csv")

        # Abort the program otherwise.
        else:
            print("Acquisition aborted")
            return -1

    # Data container
    expression = []

    # Load the Haar cascade for face detection.
    face_cascade = cv2.CascadeClassifier('./config/haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise NameError("Couldn't find the Haar Cascade XML file.")

    # Opening video capture
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise Exception("Could not open webcam")

    # Loop to capture frames
    for ind in range(num_expression):
        # Set the smile-request boolean
        # Still unsure if requesting smiles at random gives better results than with a regular interval.
        # Regular requests makes people less focused, random smile requests makes them confused.
        # Half-half seems to be a good approach.
        # smile_bool = np.random.randint(0, 2)
        # smile_bool = ind % 2
        smile_bool = np.round(ind/num_expression)

        # Display the smile instruction and wait 1 second.
        img = np.zeros((512, 512), np.uint8)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, 'SMILE', (50, 300), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
        if not smile_bool:
            cv2.putText(img, "DON'T", (50, 200), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, str(ind + 1) + "/" + str(num_expression), (50, 450), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Frame', img)
        cv2.waitKey(1000)

        # Read image
        ret, frame = video_capture.read()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect a face with a Haar cascade
        try:
            # Apply the Haar cascade
            faces = face_cascade.detectMultiScale(img_gray)
            fc = faces[0]
            img_face = img_gray[fc[1]:fc[1] + fc[3], fc[0]:fc[0] + fc[2]]

            # Save image and label
            expression.append(smile_bool)
            nom = folder + series_name + "_" + str(ind).zfill(4) + ".png"
            cv2.imwrite(nom, img_face)
        except:
            print("Could not detect a face.")

    # Close webcam and instruction window
    video_capture.release()
    cv2.destroyAllWindows()

    # Save the list of expressions
    nom_fiche = folder + series_name + "_labels.csv"
    np.savetxt(nom_fiche, expression, fmt='%i')


# ---------------- #
# LOADING DATA SET #
# ---------------- #
def load_data(data_dir, series_names, res_out):
    # Loading the labels
    print("Loading the data")
    labels = []
    for sn in series_names:
        file_name = glob.glob(data_dir + sn + "_labels.csv")[0]
        lab = np.genfromtxt(file_name, dtype=float)
        print("Loaded {} labels for series {}".format(len(lab), sn))
        labels = np.concatenate((labels, lab))

    # Loading the images
    file_list = [glob.glob(data_dir + ser_name + "*.png") for ser_name in series_names]
    data = []
    for ind_ser in range(len(file_list)):
        count_img = 0
        for ind in range(len(file_list[ind_ser])):
            # Load an image in B&W
            fname = file_list[ind_ser][ind]
            img_gray = cv2.imread(fname, 0)
            img_resz = cv2.resize(img_gray, res_out)
            data.append(img_resz)
            count_img += 1
        print("Loaded {} images for series {}".format(count_img, fname))

    # Returning the data
    data = np.stack(data, axis=0)
    data = np.array(data) / 255.0
    return data, labels


# -------------------- #
# TRAIN NEURAL NETWORK #
# -------------------- #
def train_network(x, y, outpath, num_test):
    # Randomly defining the testing sample.
    ind_shuffle = np.arange(x.shape[0])
    np.random.shuffle(ind_shuffle)

    xtr = x[ind_shuffle[num_test:], :, :]
    ytr = y[ind_shuffle[num_test:]]
    xte = x[ind_shuffle[:num_test], :, :]
    yte = y[ind_shuffle[:num_test]]

    # Defining, compiling, and training the model.
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=xtr.shape[1:]),
        tf.keras.layers.Dense(5000, activation=tf.nn.relu),
        tf.keras.layers.Dense(100, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(xtr, ytr, epochs=50)
    model.evaluate(xte, yte)

    # Saving the model
    model.save(outpath)


# ------------------ #
# WEBCAM APPLICATION #
# ------------------ #
def webcam_smile(inpath, res_img):
    # Loading the neural network
    model = tf.keras.models.load_model(inpath)

    # Loading the Haar cascade
    face_cascade = cv2.CascadeClassifier('./config/haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise NameError("Couldn't find the Haar Cascade XML file.")

    # Opening webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise Exception("Could not open webcam")

    # Capturing image from webcam and running smile detection
    while True:
        # Read image
        ret, frame = video_capture.read()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face and smile.
        try:
            # Haar cascade
            faces = face_cascade.detectMultiScale(img_gray)
            fc = faces[0]

        except:
            smile_bool = False
            print("Could not detect a face.")

        else:
            # Taking only the ROI and prepare for the neural network
            img_face = img_gray[fc[1]:fc[1] + fc[3], fc[0]:fc[0] + fc[2]]
            img_resz = cv2.resize(img_face, res_img)
            img_norm = img_resz / 255.0
            img_nn = np.array([img_norm])

            # Apply the neural network
            smile_bool = model.predict(img_nn)
            print(smile_bool[0])
            smile_bool = np.round(smile_bool[0])

            # Draw a rectangle around the face.
            cv2.rectangle(frame, (fc[0], fc[1]), (fc[0] + fc[2], fc[1] + fc[3]), (255, 0, 0), 2)

        # Display a smiley if a smile is detected
        font = cv2.FONT_HERSHEY_DUPLEX
        if smile_bool:
            cv2.putText(frame, '=D', (25, 75), font, 2, (0, 0, 255), 1, cv2.LINE_AA)

        # Display instructions to exit
        cv2.putText(frame, "Press 'ESC' to exit", (25, 450), font, 1, (255, 255, 255), 1)

        # Showing the image and catching the escape key
        cv2.imshow("Face detection", frame)
        key_press = cv2.waitKey(16) & 0xFF
        if key_press == 27:
            break

    # Closing the window and webcam
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # To capture a dataset
    # create_data(num_expression=60, series_name="appart5", folder="./data/")

    # Loading the data and training the network
    # x, y = load_data(data_dir="./data/",
    #                  series_names=['appart1', 'appart2', 'appart3', 'appart5',
    #                                'office1', 'office2',
    #                                'jon1', 'jon2'],
    #                  res_out=(64, 64))
    # train_network(x, y, outpath="./model/model.h5", num_test=30)

    # Runnning the webcam application
    webcam_smile(inpath="./model/model.h5", res_img=(64, 64))

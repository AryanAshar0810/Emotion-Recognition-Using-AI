import os
import warnings
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set(color_codes=True)
directory_training = r"C:\Users\Pranav\PycharmProjects\HelloWorld\train/"
dir_training = os.listdir(directory_training)
classes = []

for folder in dir_training:
    classes.append(folder)
    print(folder)

train_counts = []

for folder in dir_training:
    path_of_class = directory_training + folder + "/"
    training_list = []
    count = 0
    for file in os.listdir(path_of_class):
        count += 1

    train_counts.append(count)

print(train_counts)

plt.scatter(classes, train_counts)
plt.plot(classes, train_counts, '-o')
plt.show()

path_of_images = []

for folder in dir_training:
    path_of_class = directory_training + folder + "/"
    for file in os.listdir(path_of_class):
        if file.endswith(".png"):
            final_path = path_of_class + file
            path_of_images.append(final_path)
        break

print(path_of_images)


for file in path_of_images:
    image = mpimg.imread(file)
    plt.figure()
    pos1 = file.find('/')
    pos2 = file.find('/0')
    title = file[pos1 + 1:pos2]
    plt.title(title, fontsize=20)
    plt.imshow(image)
    plt.show()

directory_test = r"C:\Users\Pranav\PycharmProjects\HelloWorld\test/"
dir_test = os.listdir(directory_test)
classes = []

for folder in dir_test:
    classes.append(folder)
    print(folder)

test_counts = []

for folder in dir_test:
    path_of_class = directory_test + folder + "/"
    training_list = []
    count = 0
    for file in os.listdir(path_of_class):
        count += 1

    test_counts.append(count)

print(test_counts)

plt.scatter(classes, test_counts)
plt.plot(classes, test_counts, '-o')
plt.show()

path_of_images = []

for folder in dir_test:
    path_of_class = directory_test + folder + "/"
    for file in os.listdir(path_of_class):
        if file.endswith(".png"):
            final_path = path_of_class + file
            path_of_images.append(final_path)
        break

print(path_of_images)

for file in path_of_images:
    image = mpimg.imread(file)
    plt.figure()
    pos1 = file.find('/')
    pos2 = file.find('/0')
    title = file[pos1+1:pos2]
    plt.title(title, fontsize = 20)
    plt.imshow(image)
    plt.show()
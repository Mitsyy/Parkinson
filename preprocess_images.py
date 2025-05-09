import cv2
import numpy as np
import os
import random

class DataSet:
    
    def __init__(self, location, categories, resize=True,
                 lheight=500, lwidth=500, grayscale=True, shuffled=False,
                 apply=None, count=1000, multiclass=False, enhance=False):
        self.categories = categories
        self.datadir = location  # Dataset folder path
        self.lheight = lheight
        self.lwidth = lwidth
        self.grayscale = grayscale
        self.shuffled = shuffled
        self.multiclass = multiclass
        self.apply = apply
        self.count = count
        self.enhance = enhance
        print("Initializing dataset...")
        self.dataset = self.create_traindata()
        if resize:
            self.dataset = self.resizeIt(self.dataset)

    def resizeIt(self, traindata_array):
        resized_traindata = []
        resized_traindata_temp = []
        for img in traindata_array[0]:
            new_image_array = cv2.resize(img, (self.lheight, self.lwidth))
            resized_traindata_temp.append(np.array(new_image_array))
        array = [np.array(resized_traindata_temp), np.array(traindata_array[1])]
        return array

    def create_traindata(self):
        traindata = []
        for lists in self.categories:
            n = 0
            path = os.path.join(self.datadir, lists)  # "parkinsons_dataset/normal" or "parkinsons_dataset/parkinson"
            print(f"Looking for images in: {path}")
            if not os.path.exists(path):
                print(f"Directory does not exist: {path}")
                continue

            class_num = self.categories.index(lists)  # 0 for "normal", 1 for "parkinson"
            for img in os.listdir(path):
                print(f"Processing image: {img}")
                img_path = os.path.join(path, img)
                if self.grayscale and self.enhance:
                    y = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    y = cv2.resize(y, (512, 512))
                    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5, 5))
                    img_array = clahe.apply(y)
                    img_array = cv2.GaussianBlur(img_array, (3, 3), 1)
                    n += 1
                    print(f"{n} images loaded successfully", end='')

                    if n >= self.count:
                        break

                elif self.enhance:
                    img_array = cv2.imread(img_path)
                    img_array = cv2.resize(img_array, (512, 512))
                    img_yuv_1 = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    img_yuv = cv2.cvtColor(img_yuv_1, cv2.COLOR_RGB2YUV)
                    y, u, v = cv2.split(img_yuv)
                    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5, 5))
                    y = clahe.apply(y)
                    y = cv2.GaussianBlur(y, (3, 3), 1)
                    img_array_1 = cv2.merge((y, u, v))
                    img_array = cv2.cvtColor(img_array_1, cv2.COLOR_YUV2RGB)
                    n += 1
                    print(f"{n} images loaded successfully", end='')

                    if n >= self.count:
                        break
                else:
                    img_array = cv2.imread(img_path)
                    n += 1
                    print(f"{n} images loaded successfully", end='')
                    if n >= self.count:
                        break

                if not self.multiclass:
                    traindata.append([img_array, class_num])
                else:
                    traindata.append([img_array, self.classes(class_num)])

            print(f"Finished loading {n} images from {lists}")
            print(len(traindata))

        if self.shuffled:
            random.shuffle(traindata)
            print("shuffled")

        traindata_img = []
        traindata_lab = []
        for sets in traindata:
            traindata_img.append(sets[0])
            traindata_lab.append(sets[1])
        traindata = [traindata_img, traindata_lab]
        return traindata

    def classes(self, class_num):
        array = [0 for _ in range(len(self.categories))]
        array[class_num] = 1
        return array

if __name__ == "__main__":
    # Provide the location and categories
    location = "parkinsons_dataset"  # Adjust this path if needed
    categories = ['normal', 'parkinson']  # Specify your categories
    
    dataset = DataSet(location=location, categories=categories, resize=True, count=1000, grayscale=True, enhance=True, shuffled=True)
    print("Dataset creation completed.")

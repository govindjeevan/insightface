#! encoding: utf-8

import os
import random
import argparse

class GeneratePairs:
    """
    Generate the pairs.txt file that is used for training face classifier when calling python `src/train_softmax.py`.
    Or others' python scripts that needs the file of pairs.txt.
    Doc Reference: http://vis-www.cs.umass.edu/lfw/README.txt
    """

    def __init__(self, data_dir, pairs_filepath, img_ext):
        """
        Parameter data_dir, is your data directory.
        Parameter pairs_filepath, where is the pairs.txt that belongs to.
        Parameter img_ext, is the image data extension for all of your image data.
        """
        self.data_dir = data_dir
        self.pairs_filepath = pairs_filepath
        self.img_ext = img_ext


    def generate(self):
        for i in range(1):
            self._generate_matches_pairs()
            self._generate_mismatches_pairs()


    def _generate_matches_pairs(self):
        """
        Generate all matches pairs
        """
        for name in os.listdir(data_dir):
            if not os.path.isdir(os.path.join(data_dir,name)):
                continue
            a = []
            for file in os.listdir(os.path.join(data_dir,name)):
                if os.path.isfile(os.path.join(data_dir, name,file)):
                    a.append(file)
            if len(a) > 1:
                with open(self.pairs_filepath, "a") as f:
                    for i in range(4):
                        temp = random.choice(a)
                        temp = temp.split("_")
                        w = "_".join(temp[0:-1]) # This line may vary depending on how your images are named.
                        l = random.choice(a).split("_")[-1].lstrip("0").rstrip(self.img_ext)
                        r = random.choice(a).split("_")[-1].lstrip("0").rstrip(self.img_ext)
                        if l!=r:
                            print(w + "\t" + l + "\t" + r + "\n")
                            f.write(w + "\t" + l + "\t" + r + "\n")
            else:
                print("no in ",name)

    def _generate_mismatches_pairs(self):
        """
        Generate all mismatches pairs
        """
        for i, name in enumerate(os.listdir(self.data_dir)):
            if not os.path.isdir(os.path.join(self.data_dir,name)):
                continue
            if len(os.listdir(os.path.join(self.data_dir, name)))<1:
                continue
            remaining = os.listdir(self.data_dir)
            remaining.remove(name)
            remaining = [f_n for f_n in remaining if f_n != ".DS_Store"]
            # del remaining[i] # deletes the file from the list, so that it is not chosen again
            other_dir = random.choice(remaining)
            if not os.path.isdir(os.path.join(self.data_dir,other_dir)):
                continue
            if len(os.listdir(os.path.join(self.data_dir, other_dir)))<1:
                continue
            with open(self.pairs_filepath, "a") as f: 
                for i in range(1):
                    file1 = random.choice(os.listdir(os.path.join(self.data_dir, name)))
                    # print('first', file1, name)
                    file2 = random.choice(os.listdir(os.path.join(self.data_dir,other_dir)))
                    # print('second', file2, other_dir)
                    if not os.path.isfile(os.path.join(self.data_dir,name,file1)):
                        continue
                    if not os.path.isfile(os.path.join(data_dir,other_dir,file2)):
                        continue
                    number_1 = file1.split("_")[-1].lstrip("0").rstrip(self.img_ext)
                    number_2 = file2.split("_")[-1].lstrip("0").rstrip(self.img_ext)
                    # print(number_1, number_2)
                    # f.write(name + "\t" + file1.split("_")[2].lstrip("0").rstrip(self.img_ext) + "\n")
                    f.write(name + "\t" + number_1 + "\t" + other_dir + "\t" + number_2 + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rename images in the folder according to LFW format: Name_Surname_0001.jpg, Name_Surname_0002.jpg, etc.')
    parser.add_argument('--data-dir', default='', help='Full path to the directory with peeople and their names, folder should denote the Name_Surname of the person')
    parser.add_argument('--txt-file', default='', help='Full path to the directory with peeople and their names, folder should denote the Name_Surname of the person')
    # reading the passed arguments
    args = parser.parse_args()
    data_dir = args.data_dir    # "out_data_crop/"
    pairs_filepath = args.txt_file         # "pairs_1.txt"
    
    img_ext = ".png"
    generatePairs = GeneratePairs(data_dir, pairs_filepath, img_ext)
    generatePairs.generate()
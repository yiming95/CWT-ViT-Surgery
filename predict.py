import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from vit_model import vit_base_patch16_224_in21k as create_model
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from utils import read_split_data
import csv
import argparse
import datetime
import pytz


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    folder_pth_list = ["./data/Processed_dataset/Suturing/" + str(args.test_set) + "/expert/", "./data/Processed_dataset/Suturing/"+ str(args.test_set) + "/intermediate/", "./data/Processed_dataset/Suturing/"+ str(args.test_set) + "/novice/"]

    for folder_path in folder_pth_list:
        for img_path in os.listdir(folder_path):
            img = Image.open(folder_path + img_path)
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)

            json_path = './class_indices.json'
            assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

            json_file = open(json_path, "r")
            class_indict = json.load(json_file)

            # create model
            model = create_model(num_classes=3, has_logits=False).to(device)

            model_weight_path = args.weight_pth

            model.load_state_dict(torch.load(model_weight_path, map_location=device))
            model.eval()

            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                        predict[predict_cla].numpy())

            output = folder_path.split('/')
            output_file = './' + output[-4] + '_' + output[-3] + '_' + output[-2] + '.csv'

            print(output_file)

            with open (output_file, 'a') as csvfile:
                fieldnames = ['trail_name', 'prediction']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                print(writer)
                for i in range(len(predict)):
                    print("class: {:10} prob: {:.3}".format(class_indict[str(i)],
                                                        predict[i].numpy()))

                    writer.writerow({'trail_name':img_path, 'prediction':"class: {:10}, prob: {:.3}".format(class_indict[str(i)],
                                                        predict[i].numpy())})
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set', type=str, default= "test5")
    parser.add_argument('--weight_pth', type=str, default="./best_acc.pth")
    opt = parser.parse_args()
    main(opt)

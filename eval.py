import os
import cv2
import sys
import argparse
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from tqdm import tqdm
from matplotlib import pyplot as plt
import csv
import random

import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

import mlflow

sys.path.insert(0, '..')
from models.Xception import *
from models.CNNDCT import *
from models.A import *

from utils.pilresize import PILResize
from utils.FCRDCT import *
from utils.tsne import *

def read_paths(iut_paths_file, undersampling, subset):
    distribution = dict()
    n_min = None

    
    with open(iut_paths_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            parts = l.rstrip().split('\t')
            iut_path = parts[0]
            label = int(parts[1])

            if (subset and subset not in parts[0]):
                continue
            
            # add to distribution
            if (label not in distribution):
                distribution[label] = [iut_path]
            else:
                distribution[label].append(iut_path)

    for label in distribution:
        if (n_min is None or len(distribution[label]) < n_min):
            n_min = len(distribution[label])

    # undersampling
    iut_paths_labels = []

    for label in distribution:
        ll = distribution[label]

        if (undersampling == 'all'):
            for i in ll:
                iut_paths_labels.append((i, label))
        elif (undersampling == 'min'):
            picked = random.sample(ll, n_min)
            
            for p in picked:
                iut_paths_labels.append((p, label))
        else:
            print('Unsupported undersampling method {}!'.format(undersampling))
            sys.exit()

    return iut_paths_labels

def save_cm(y_true, y_pred, save_path):
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument("--id", type=str, help="run id")

    parser.add_argument("--iut_paths_file", type=str, default="/dataset/iut_files.txt", help="path to the file with paths for image under test") # each line of this file should contain "/path/to/image.ext i", i is an integer represents classes
    parser.add_argument("--image_size", type=int, default=512, help="size of images")

    parser.add_argument("--subset", type=str, help="evaluation on certain subset")
    parser.add_argument("--undersampling", type=str, default='all', choices=['all', 'min'])

    parser.add_argument('--out_dir', type=str, default='out')
    
    parser.add_argument('--model', default='xception', choices=['xception', 'cnndct','cnnpixel','ours'], help='model selection')
    parser.add_argument('--load_path', type=str, help='path to the pretrained model', default="checkpoints/model.pth")
    
    ## DataLad
    parser.add_argument("--repo", type=str, help="repository URL")
    parser.add_argument("--commit", type=str, help="commit hash")
    parser.add_argument("--name", type=str, help="name of the dataset")

    ## MLflow
    parser.add_argument("--workspace", type=str, help="workspace name")
    parser.add_argument("--experiment", type=str, help="experiment name")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # Init mlflow
    mlflow.set_workspace(args.workspace) 
    mlflow.set_experiment(args.experiment)

    mlflow_run = mlflow.start_run(run_name=f"eval-{args.id}")

    mlflow.log_params({
        "run_id": args.id,
        "image_size": args.image_size,
        "iut_paths_file": args.iut_paths_file,
        "subset": args.subset,
        "undersampling": args.undersampling,
        "model": args.model,
        "load_path": args.load_path,
    })

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (args.model == 'xception'):
        model = Xception().to(device)
    elif (args.model == 'cnndct' or args.model == 'cnnpixel'):
        model = CNNDCT(args.image_size).to(device)
    elif (args.model == 'ours'):
        model = Attributor(args.image_size).to(device)
    else:
        print("Unrecognized model %s" % args.model)

    if args.load_path != None and os.path.exists(args.load_path):
        print('Load pretrained model: {}'.format(args.load_path))
        model.load_state_dict(torch.load(args.load_path, map_location=device))
    else:
        print("%s not exist" % args.load_path)
        sys.exit()

    # no training
    model.eval()

    # read paths for data
    if not os.path.exists(args.iut_paths_file):
        print("%s not exists, quit" % args.iut_paths_file)
        sys.exit()

    if (args.subset):
        print("Evaluation on subset {}".format(args.subset))

    iut_paths_labels = read_paths(args.iut_paths_file, args.undersampling, args.subset)

    print("Eval set size is {}!".format(len(iut_paths_labels)))

    # log dataset to MLflow
    mlflow.log_artifact(args.iut_paths_file, "datasets")

    dataframe = pd.DataFrame({
        "input_image_paths": [iut_path for iut_path, _ in iut_paths_labels],
        "labels": [label for _, label in iut_paths_labels]
    })
    dataset = mlflow.data.from_pandas(
        dataframe,
        source=args.repo,
        digest=args.commit[:36],
        name=args.name + ("-test")
    )
    mlflow.log_input(dataset, "test")

    mlflow.log_params({
        "repo": args.repo,
        "commit": args.commit,
        "name": args.name,
    })

    # create/reset output folder
    print("Predicted maps will be saved in :%s" % args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    if (args.subset is None):
        os.makedirs(os.path.join(args.out_dir, 'images'), exist_ok=True)

    # save paths
    if (args.undersampling == 'min'):
        save_path = os.path.join(args.out_dir, 'paths_file_eval.txt')
        with open(save_path, 'w') as f:
            for (iut_path, label) in iut_paths_labels:
                f.write(iut_path + '\t' + str(label) + '\n')

        print('Eval paths file saved to %s' % (save_path))

        mlflow.log_artifact(save_path, "datasets")

    # csv
    if (args.subset is None):
        f_csv = open(os.path.join(args.out_dir, 'pred.csv'), 'w', newline='')
        writer = csv.writer(f_csv)

        header = ['Image', 'Pred', 'True', 'Correct']
        writer.writerow(header)

    # transforms
    if (args.model == 'xception' or args.model == 'cnnpixel'):
        transform = A.Compose([
            A.Normalize(mean=0.0, std=1.0), 
            ToTensorV2()
        ])
    elif (args.model == 'cnndct' or args.model == 'ours'):
        transform = A.Compose([
            A.Normalize(mean=0.0, std=1.0), 
            ToTensorV2(),
            DCT(p = 1.0, log=True, factor=1)
        ])
    else:
        print("Unrecognized model %s" % args.model)

    ## prediction
    y_pred = []
    y_true = []

    for ix, (iut_path, lab) in enumerate(tqdm(iut_paths_labels, mininterval = 60)):
        try:
            img = cv2.cvtColor(cv2.imread(iut_path), cv2.COLOR_BGR2RGB)
        except:
            print('Failed to load image {}'.format(iut_path))
            continue
        if (img is None):
            print('Failed to load image {}'.format(iut_path))
            continue

        # DCT
        img = transform(image = img)['image'].to(device)

        # prediction
        with torch.no_grad():
            out = model(img.unsqueeze(0))
        y = 1 if out > 0.5 else 0
        
        y_pred.append(y)
        y_true.append(lab)

        # write to csv
        if (args.subset is None):
            row = [iut_path, y, lab, y == lab]
            writer.writerow(row)

    ## accuracy
    print("acc%s: %.4f" % ((' (' + args.subset + ')' if args.subset else ''), accuracy_score(y_true, y_pred)))

    ## confusion matrix
    save_path = os.path.join(args.out_dir, 'cm' + ('_' + args.subset if args.subset else '') + '.png')
    save_cm(y_true, y_pred, save_path)

    # log results to MLflow
    mlflow.log_artifact(args.out_dir, "results")

    mlflow.end_run()

    if (args.subset is None): f_csv.close()
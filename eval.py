import os, cv2, sys, argparse, csv, random, json
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
sys.path.insert(0,'..')
from models.Xception import *
from models.CNNDCT import *
from models.A import *
from utils.pilresize import PILResize
from utils.FCRDCT import *
from utils.tsne import *
from pipeline.helpers import write_portable_manifest

def read_paths(iut_paths_file,undersampling,subset):
    distribution={}; n_min=None
    with open(iut_paths_file,'r') as f:
        for l in f.readlines():
            parts=l.rstrip().split('\t'); iut_path=parts[0]; label=int(parts[1])
            if subset and subset not in parts[0]: continue
            distribution.setdefault(label,[]).append(iut_path)
    for label in distribution:
        if n_min is None or len(distribution[label])<n_min: n_min=len(distribution[label])
    values=[]
    for label,ll in distribution.items():
        picked=ll if undersampling=='all' else random.sample(ll,n_min) if undersampling=='min' else None
        if picked is None: raise ValueError(f'Unsupported undersampling method {undersampling}')
        values.extend((p,label) for p in picked)
    return values
def save_cm(y_true,y_pred,save_path):
    plt.figure(); ConfusionMatrixDisplay.from_predictions(y_true,y_pred); plt.tight_layout(); plt.savefig(save_path,dpi=300); plt.close()
def parse_args():
    p=argparse.ArgumentParser(description='Evaluation'); p.add_argument('--id',type=str); p.add_argument('--iut_paths_file',type=str,required=True); p.add_argument('--image_size',type=int,default=512); p.add_argument('--subset',type=str); p.add_argument('--undersampling',default='all',choices=['all','min']); p.add_argument('--out_dir',default='out'); p.add_argument('--model',default='xception',choices=['xception','cnndct','cnnpixel','ours']); p.add_argument('--load_path',default='checkpoints/model.pth'); p.add_argument('--repo'); p.add_argument('--commit'); p.add_argument('--name'); p.add_argument('--dataset_root'); return p.parse_args()
if __name__=='__main__':
    args=parse_args(); device=torch.device('cpu')
    model=Xception().to(device) if args.model=='xception' else CNNDCT(args.image_size).to(device) if args.model in ('cnndct','cnnpixel') else Attributor(args.image_size).to(device)
    if args.load_path and os.path.exists(args.load_path): print('Load pretrained model: {}'.format(args.load_path)); model.load_state_dict(torch.load(args.load_path,map_location=device))
    else: raise FileNotFoundError(args.load_path)
    model.eval()
    if not os.path.exists(args.iut_paths_file): raise FileNotFoundError(args.iut_paths_file)
    iut_paths_labels=read_paths(args.iut_paths_file,args.undersampling,args.subset); print('Eval set size is {}!'.format(len(iut_paths_labels)))
    os.makedirs(args.out_dir,exist_ok=True)
    if args.subset is None: os.makedirs(os.path.join(args.out_dir,'images'),exist_ok=True)
    if args.dataset_root: write_portable_manifest(args.iut_paths_file,Path(args.out_dir)/'test_datalad.txt',args.dataset_root)
    if args.undersampling=='min':
        with open(os.path.join(args.out_dir,'paths_file_eval.txt'),'w') as f:
            for iut_path,label in iut_paths_labels: f.write(iut_path+'\t'+str(label)+'\n')
    f_csv=None; writer=None
    if args.subset is None:
        f_csv=open(os.path.join(args.out_dir,'pred.csv'),'w',newline=''); writer=csv.writer(f_csv); writer.writerow(['Image','Pred','True','Correct'])
    transform=A.Compose([A.Normalize(mean=0.0,std=1.0),ToTensorV2()] if args.model in ('xception','cnnpixel') else [A.Normalize(mean=0.0,std=1.0),ToTensorV2(),DCT(p=1.0,log=True,factor=1)])
    y_pred=[]; y_true=[]
    for iut_path,lab in tqdm(iut_paths_labels,mininterval=60):
        img=cv2.imread(iut_path)
        if img is None: print('Failed to load image {}'.format(iut_path)); continue
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB); img=transform(image=img)['image'].to(device)
        with torch.no_grad(): out=model(img.unsqueeze(0))
        y=int(torch.sigmoid(out).item()>0.5); y_pred.append(y); y_true.append(lab)
        if writer: writer.writerow([iut_path,y,lab,y==lab])
    accuracy=float(accuracy_score(y_true,y_pred)); print('acc%s: %.4f' % ((' ('+args.subset+')' if args.subset else ''),accuracy)); save_cm(y_true,y_pred,os.path.join(args.out_dir,'cm'+('_'+args.subset if args.subset else '')+'.png'))
    if f_csv: f_csv.close()
    result={'execution_id':str(args.id),'accuracy':accuracy,'output_dir':str(Path(args.out_dir).resolve())}; (Path(args.out_dir)/'result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))

import random
import torch.distributed as dist
from tqdm import tqdm
from framework import Trainer
from utils.datasets import prepare_Beijing_dataset, prepare_Porto_dataset
import torch
import os
from tqdm import tqdm
from utils.metrics import IoU


from networks.transformer.transformer import SegTrans

def get_model(model_name):
    if model_name == 'segtrans':
        model = SegTrans(num_classes=1)
    return model


def get_dataloader(args):
    if args.porto == True:
        train_ds, val_ds, test_ds = prepare_Porto_dataset(args)
    else:
        train_ds, val_ds, test_ds = prepare_Beijing_dataset(args)
    if args.dist == False:
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=BATCH_SIZE, num_workers=args.workers)
        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=1, num_workers=args.workers)
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=1, num_workers=args.workers)
    else:
        train_sampler = DistributedSampler(train_ds)
        train_dl = torch.utils.data.DataLoader(train_ds, num_workers=args.workers, batch_size=BATCH_SIZE, sampler=train_sampler)

        val_dl = torch.utils.data.DataLoader(
            val_ds, batch_size=1, num_workers=args.workers)
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=1, num_workers=args.workers)

    return train_dl, val_dl, test_dl


def train(args):
    net = get_model(args.model)
    train_dl, val_dl, test_dl = get_dataloader(args)
    trainer = Trainer(net, None, args.local_rank, args.lr, args.dist, args)
    if args.weight_load_path != '':
        print("load ", args.weight_load_path)
        trainer.solver.load_weights(args.weight_load_path)
    trainer.set_train_dl(train_dl)
    trainer.set_validation_dl(val_dl)
    trainer.set_test_dl(test_dl)
    trainer.set_save_path(WEIGHT_SAVE_DIR)

    trainer.fit(epochs=args.epochs)


def predict(args):
    net = get_model(args.model)
    args.aug = False
    if args.porto:
        _, _, test_ds = prepare_Porto_dataset(args)
    else:
        _, _, test_ds = prepare_Beijing_dataset(args)

    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)
    trainer = Trainer(net, optimizer, args.local_rank, args.lr, args.dist, args)
    if args.weight_load_path != '':
        trainer.solver.load_weights(args.weight_load_path)

    predict_dir = os.path.join(os.path.split(args.weight_load_path)[0], "prediction")
    if not os.path.exists(predict_dir):
        os.mkdir(predict_dir)
    metric = IoU()
    metrics = 0
    iou_list = []
    for i, data in tqdm(enumerate(test_ds)):
        image = data[0]
        pred = trainer.solver.pred_one_image(image)
        pred = pred[0]
        metric_single = metric(torch.tensor(data[1][0, :, :]), pred[0, 0, :, :].cpu() )
        iou_list.append(metric_single[3])
        metrics += metric_single
        pred_filename = os.path.join(predict_dir, f"{i}_pred.png")
        print("[DONE] predicted image: ", pred_filename)
    metric_all = metrics / len(test_ds)
    print(metric_all)
    print(f"aiou = {metric_all[3]:.4f} ,  giou = {(metric_all[4] / metric_all[5]):.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='segtrans')
    parser.add_argument('--lr', '-lr', type=float, default=2e-4)
    parser.add_argument('--name', '-n', type=str, default='')
    parser.add_argument('--batch_size', '-b', type=int, default=2)
    parser.add_argument('--sat_dir', '-sat', type=str,
                        default=r'C:\Users\legen\Documents\001-论文\数据集\BJRoad\train_val\image')
    parser.add_argument('--mask_dir', '-Mask', type=str,
                        default=r'C:\Users\legen\Documents\001-论文\数据集\BJRoad\train_val\mask')
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--test_sat_dir', type=str,
                        default=r'C:\Users\legen\Documents\001-论文\数据集\BJRoad\test\image')
    parser.add_argument('--test_mask_dir', type=str,
                        default=r'C:\Users\legen\Documents\001-论文\数据集\BJRoad\test\mask')
    parser.add_argument('--gps_dir', '-g', type=str,
                        default=r'C:\Users\legen\Documents\001-论文\数据集\BJRoad\train_val\gps')
    parser.add_argument('--gps_type', '-t', type=str, default='image')
    parser.add_argument('--feature_embedding', '-F', type=str, default='')
    parser.add_argument('--gps_augmentation', '-A', type=str, default='')
    parser.add_argument('--test_gps_augmentation', type=str, default='')
    parser.add_argument('--weight_save_dir', '-W',
                        type=str, default='.\weights')
    parser.add_argument('--weight_load_path', '-L', type=str, default='')
    parser.add_argument('--val_size', '-T', type=float, default=0.1)
    parser.add_argument('--use_gpu', '-G', type=bool, default=False)
    parser.add_argument('--gpu_ids', '-N', type=str, default='0')
    parser.add_argument('--workers', '-w', type=int, default=4)
    parser.add_argument('--epochs', '-e', type=int, default=200)
    parser.add_argument('--random_seed', '-r', type=int, default=0)
    parser.add_argument('--eval', '-E', type=str, default="")
    parser.add_argument('--porto', type=bool, default=False)
    parser.add_argument('--split', type=str, default="1")
    parser.add_argument('--porto_img_dir', type=str,
                        default='/data/porto_dataset/images/')
    parser.add_argument('--porto_mask_dir', type=str,
                        default='/data/porto_dataset/mask/')
    parser.add_argument('--porto_gps_dir', type=str,
                        default='/data/porto_dataset/gps/')
    parser.add_argument('--porto_root_dir', type=str,
                        default='/data/porto_dataset/')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist', type=bool, default=False)
    parser.add_argument('--first_stage', type=bool, default=False)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--gpsratio', type=float)

    args = parser.parse_args()
    args.batch_size = int(args.batch_size)
    args.porto = bool(args.porto)
    args.random_seed = random.randint(0, 100000)
    if args.dist == True:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

    if args.porto:
        args.porto_img_dir = args.porto_root_dir + "split_" + args.split + "/images/"
        args.porto_mask_dir = args.porto_root_dir + "split_" + args.split + "/mask/"
        args.porto_gps_dir = args.porto_root_dir + "split_" + args.split + "/gps/"

    if args.use_gpu:
        try:
            gpu_list = [int(s) for s in args.gpu_ids.split(',')]
            print("gpu_list = ", gpu_list)
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        except ValueError:
            raise ValueError(
                'Argument --gpu_ids must be a comma-separated list of integers only')
        BATCH_SIZE = args.batch_size
    else:
        BATCH_SIZE = args.batch_size

    if args.sat_dir == "" and args.gps_dir != "":
        input_channels = "gps_only"
        input_channel_num = 1
    elif args.sat_dir != "" and args.gps_dir == "":
        input_channels = "image_only"
        input_channel_num = 3
    elif args.sat_dir != "" and args.gps_dir != "":
        input_channels = "image_gps"
        input_channel_num = 4
    else:
        print("[ERROR] Both input source are empty!")
        exit(1)

    if args.feature_embedding != "":
        num_embedding = args.feature_embedding.split('-')
        input_channel_num += len(num_embedding)
        if "heading" in num_embedding:
            input_channel_num += 1
        print("[INFO] gps embedding: ", num_embedding)

    if args.porto == True:
        if args.prefix != '':
            WEIGHT_SAVE_DIR = os.path.join(args.weight_save_dir,
                                           f"{args.prefix}_porto_split_{args.split}_{args.model}_{input_channels}_{args.gps_augmentation}")
        else:
            WEIGHT_SAVE_DIR = os.path.join(args.weight_save_dir,
                                       f"porto_split_{args.split}_{args.model}_{input_channels}_{args.gps_augmentation}")
    else:
        if args.prefix != '':
            WEIGHT_SAVE_DIR = os.path.join(args.weight_save_dir,
                                           f"{args.prefix}_{args.model}_{input_channels}_{args.feature_embedding}_{args.gps_augmentation}")
        else:
            WEIGHT_SAVE_DIR = os.path.join(args.weight_save_dir,
                                           f"{args.model}_{input_channels}_{args.feature_embedding}_{args.gps_augmentation}")

    if not os.path.exists(WEIGHT_SAVE_DIR):
        os.mkdir(WEIGHT_SAVE_DIR)

    print("[INFO] input: ", input_channels)
    print("[INFO] channels: ", input_channel_num)

    if args.eval == "":
        train(args)
        print("[DONE] training finished")
    elif args.eval == "predict":
        predict(args)
        print("[DONE] predict finished")
import os
from sklearn.model_selection import train_test_split
from .data_loader import ImageGPSDataset

def prepare_Beijing_dataset(args, aug_sampling_rate=None, aug_precision_rate=None):
    image_list = [x[:-9] for x in os.listdir(args.mask_dir) if x.find('mask.png') != -1]
    test_list = [x[:-9] for x in os.listdir(args.test_mask_dir) if x.find('mask.png') != -1]
    train_list, val_list = train_test_split(image_list, test_size=args.val_size, random_state=args.random_seed)
    train_dataset = ImageGPSDataset(train_list, args.sat_dir, args.mask_dir,
                                    gps_root=args.gps_dir,
                                    gps_typd='data', feature_embedding=args.feature_embedding,
                                    aug_mode=args.gps_augmentation, first_stage=args.first_stage)
    val_dataset = ImageGPSDataset(val_list, args.sat_dir, args.mask_dir,
                                  gps_root=args.gps_dir,
                                  gps_typd='data', feature_embedding=args.feature_embedding,
                                  randomize=False, first_stage=args.first_stage)
    test_dataset = ImageGPSDataset(test_list, args.test_sat_dir, args.test_mask_dir, args.gps_dir,
                                   gps_typd=args.gps_type,
                                   feature_embedding=args.feature_embedding, randomize=False,
                                   aug_mode=args.test_gps_augmentation,
                                   aug_sampling_rate=aug_sampling_rate, aug_precision_rate=aug_precision_rate, first_stage=args.first_stage)
    return train_dataset, val_dataset, test_dataset

def prepare_Porto_dataset(args, aug_sampling_rate=None, aug_precision_rate=None):
    split_index = args.split
    split_dir = os.path.join(args.porto_root_dir, "split_" + split_index)
    train_list=  open(os.path.join(split_dir, 'train.txt'), 'r').readlines()
    val_list=  open(os.path.join(split_dir, 'valid.txt'), 'r').readlines()
    test_list=  open(os.path.join(split_dir, 'test.txt'), 'r').readlines()
    train_list = [x.strip() for x in train_list]
    val_list = [x.strip() for x in val_list]
    test_list = [x.strip() for x in test_list]
    train_dataset = ImageGPSDataset(train_list, args.porto_img_dir, args.porto_mask_dir, args.porto_gps_dir,
                                    gps_typd='image', feature_embedding=args.feature_embedding, aug_mode=args.gps_augmentation, porto=True)
    val_dataset = ImageGPSDataset(val_list, args.porto_img_dir, args.porto_mask_dir, args.porto_gps_dir,
                                  gps_typd='image', feature_embedding=args.feature_embedding, randomize=False, porto=True)
    test_dataset = ImageGPSDataset(test_list, args.porto_img_dir, args.porto_mask_dir, args.porto_gps_dir,
                                   gps_typd='image',
                                   feature_embedding=args.feature_embedding, randomize=False,
                                   aug_mode=args.test_gps_augmentation,
                                   aug_sampling_rate=aug_sampling_rate, aug_precision_rate=aug_precision_rate, porto=True)
    return train_dataset, val_dataset, test_dataset

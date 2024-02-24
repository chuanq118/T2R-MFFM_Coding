import os
import torch.utils.data as data
from .image_augmentation import *
from .gps_render import GPSDataRender, GPSImageRender
import albumentations as A

class ImageGPSDataset(data.Dataset):
    def __init__(self, image_list, sat_root="", mask_root="",
                 gps_root="", sat_type="jpg", mask_type="jpg", gps_typd="data",
                 feature_embedding="", aug_mode="", randomize=True, aug_sampling_rate=None, aug_precision_rate=None, porto=False, aug=False, first_stage=False):
        self.image_list = image_list
        self.sat_root = sat_root
        self.mask_root = mask_root
        self.gps_root = gps_root
        self.aug = aug
        self.first_stage = first_stage
        self.sat_type = "png"
        if self.aug:
            self.sat_type = "jpg"
        self.mask_type = mask_type
        self.randomize = randomize
        self.porto = porto
        if gps_typd == '':
            self.gps_render = None
        elif gps_typd == 'image':
            self.gps_render = GPSImageRender(gps_root)
        elif gps_typd == 'data':
            self.gps_render = GPSDataRender(gps_root, feature_embedding, aug_mode, aug_sampling_rate, aug_precision_rate)

    def _read_image_and_mask(self, image_id):
        if self.porto:
            if self.sat_root != "":
                img = cv2.imread(os.path.join(
                    self.sat_root, "{0}.{1}").format(image_id, self.sat_type))
            else:
                img = None
            mask = cv2.imread(
                os.path.join(
                    self.mask_root,  "{}.png").format(image_id), cv2.IMREAD_GRAYSCALE
            )
            if mask is None: print("[WARN] empty mask: ", image_id)
        else:
            if self.aug == True:
                if self.sat_root != "":
                    img = cv2.imread(os.path.join(
                        self.sat_root, "{0}.{1}").format(image_id, self.sat_type))
                else:
                    img = None
                mask = cv2.imread(
                    os.path.join(
                        self.mask_root, "{}.jpg").format(image_id), cv2.IMREAD_GRAYSCALE
                )
            else:
                if self.sat_root != "":
                    img = cv2.imread(os.path.join(
                        self.sat_root, "{0}_sat.{1}").format(image_id, self.sat_type))
                else:
                    img = None
                mask = cv2.imread(
                    os.path.join(
                        self.mask_root,  "{}_mask.png").format(image_id), cv2.IMREAD_GRAYSCALE
                )
            if mask is None: print("[WARN] empty mask: ", image_id)
        return img, mask

    def _render_gps_to_image(self, image_id):
        if self.aug:
            gps_image = self.gps_render.render(image_id)
        else:
            ix, iy = image_id.split('_')
            gps_image = self.gps_render.render(int(ix), int(iy))
        return gps_image

    def _concat_images(self, image1, image2):
        if image1 is not None and image2 is not None:
            img = np.concatenate([image1, image2], 2)
        elif image1 is None and image2 is not None:
            img = image2
        elif image1 is not None and image2 is None:
            img = image1
        else:
            print("[ERROR] Both images are empty.")
            exit(1)
        return img

    def _data_augmentation(self, sat, mask, gps_image, randomize=True, first_stage=False):
        transform = A.Compose([
            A.OneOf([A.HorizontalFlip(p=0.8),
                     A.VerticalFlip(p=0.8),
                     A.Transpose(p=0.5)], p=0.5),
            A.RandomBrightnessContrast(p = 0.4),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=180, p=0.9),
            A.OneOf([A.GridDistortion(p=0.5),
                     A.ElasticTransform(p=0.5)], p=0.6),
            ], p = 0.9)
        if randomize:
            img = self._concat_images(sat, gps_image)
            trans_im = transform(image=img, mask=mask)
            img, mask = trans_im['image'], trans_im['mask']
            if first_stage:
                img, mask = randomExchange(img, mask, u=1.0)
            else:
                img, mask = randomMask(img, mask, u=1.0)
        else:
            img = self._concat_images(sat, gps_image)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        mask = np.expand_dims(mask, axis=2)
        try:
            img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
            mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
        except Exception as e:
            print(e)
            print(img.shape, mask.shape)
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        return img, mask

    def __getitem__(self, index):
        image_id = self.image_list[index]
        if self.gps_render is not None:
            gps_img = self._render_gps_to_image(image_id)
        else:
            gps_img = None
            print(f"gps_img_{index} is None")
        img, mask = self._read_image_and_mask(image_id)
        img, mask = self._data_augmentation(img, mask, gps_img, self.randomize, self.first_stage)
        img, mask = torch.Tensor(img), torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(self.image_list)


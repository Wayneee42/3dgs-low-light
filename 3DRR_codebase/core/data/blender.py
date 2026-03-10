# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import json
import os
from pathlib import Path

import torch
import torchvision


def build_frame_key(file_path):
    split_name, file_name = file_path.split("/")[-2:]
    return f"{split_name}_{Path(file_name).stem}"


class Blender(torch.utils.data.Dataset):
    def __init__(self, data_cfg, split, load_images=True):
        super().__init__()
        assert split in ["train", "val", "test"]

        self._scene_root = data_cfg.DATA_PATH
        self._bg_color = data_cfg.BACKGROUND_COLOR / 255.0
        self._load_images = load_images
        self._requested_split = split
        self._meta_split = "test" if split == "val" else split
        self._render_split = split
        self._auxiliary_dir = getattr(data_cfg, "AUXILIARY_DIR", "auxiliaries")

        self._img_path_base = os.path.join(self._scene_root, self._meta_split)
        self._meta_path_base = os.path.join(self._scene_root, f"transforms_{self._meta_split}.json")
        self._records, self._data_info = self._load_data()

        if split == "val":
            first_four_keys = list(self._records.keys())[:4]
            self._records = {key: self._records[key] for key in first_four_keys}

        if load_images:
            self._pre_loading_data()

        self._records_keys = list(self._records.keys())
        self._length = len(self._records_keys)

    def __getitem__(self, index):
        frame_key = self._records_keys[index % len(self._records_keys)]
        return self._load_one_record(self._records[frame_key])

    def __len__(self):
        return self._length

    def _load_one_record(self, record):
        one_record_data = {
            "transforms": record["transform_matrix"],
            "infos": {
                "frame_key": record["frame_key"],
                "frame_name": record["frame_name"],
                "frame_stem": record["frame_stem"],
                "split": record["split"],
                "relative_path": record["relative_path"],
            },
            "low_light_image": None,
            "depth": None,
            "prior": None,
        }
        if self._load_images:
            one_record_data["images"] = record["img_tensor"]
            one_record_data["low_light_image"] = record["low_light_tensor"]
            one_record_data["depth"] = record["depth_tensor"]
            one_record_data["prior"] = record["prior_tensor"]
        return one_record_data

    def _load_data(self):
        with open(self._meta_path_base, "rb") as f:
            json_data = json.load(f)
        meta_info = {
            "bg_color": self._bg_color,
            "img_h": int(json_data["h"]),
            "img_w": int(json_data["w"]),
            "fl_x": json_data["fl_x"],
            "fl_y": json_data["fl_y"],
            "cx": json_data["cx"],
            "cy": json_data["cy"],
            "camera_convention": "blender_nerf_synthetic_c2w_opengl",
            "renderer_camera_convention": "opencv_w2c_after_yz_flip",
        }
        records = {}
        for frame in json_data["frames"]:
            relative_path = frame["file_path"]
            frame_key = build_frame_key(relative_path)
            frame_name = os.path.basename(relative_path)
            frame_stem = Path(frame_name).stem
            file_path = os.path.join(self._scene_root, relative_path.replace("/", os.sep))
            transform_matrix = torch.tensor(frame["transform_matrix"]).float()[:3]
            records[frame_key] = {
                "frame_key": frame_key,
                "frame_name": frame_name,
                "frame_stem": frame_stem,
                "split": relative_path.split("/")[-2],
                "relative_path": relative_path,
                "file_path": file_path,
                "img_tensor": None,
                "low_light_tensor": None,
                "depth_tensor": None,
                "prior_tensor": None,
                "transform_matrix": transform_matrix,
            }
        return records, meta_info

    def _pre_loading_data(self):
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor

        def _load_record_assets(key, record):
            image_tensor = load_img(record["file_path"], channel=3).float() / 255.0
            low_light_tensor = self._load_optional_auxiliary(record["frame_key"], "lowlight", channel=3)
            depth_tensor = self._load_optional_auxiliary(record["frame_key"], "depth", channel=1)
            prior_tensor = self._load_optional_auxiliary(record["frame_key"], "prior", channel=1)
            return key, image_tensor[:3], low_light_tensor, depth_tensor, prior_tensor

        print(f"Load data [{self._requested_split}]: [{len(self._records)}].")
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 16)) as executor:
            all_records = list(executor.map(lambda item: _load_record_assets(item[0], item[1]), self._records.items()))
        for key, image, low_light, depth, prior in all_records:
            self._records[key]["img_tensor"] = image
            self._records[key]["low_light_tensor"] = low_light
            self._records[key]["depth_tensor"] = depth
            self._records[key]["prior_tensor"] = prior

    def _load_optional_auxiliary(self, frame_key, modality, channel):
        aux_path = resolve_auxiliary_path(self._scene_root, self._auxiliary_dir, modality, frame_key)
        if aux_path is None:
            return None
        return load_img(aux_path, channel=channel).float() / 255.0



def resolve_auxiliary_path(scene_root, auxiliary_dir, modality, frame_key):
    modality_root = Path(scene_root) / auxiliary_dir / modality
    if not modality_root.exists():
        return None
    for extension in (".png", ".jpg", ".jpeg", ".JPG", ".JPEG"):
        candidate = modality_root / f"{frame_key}{extension}"
        if candidate.exists():
            return str(candidate)
    return None



def load_img(file_name, channel=3):
    if channel == 3:
        mode = torchvision.io.ImageReadMode.RGB
    elif channel == 4:
        mode = torchvision.io.ImageReadMode.RGB_ALPHA
    else:
        mode = torchvision.io.ImageReadMode.GRAY
    image = torchvision.io.read_image(file_name, mode=mode)
    assert image is not None, file_name
    return image

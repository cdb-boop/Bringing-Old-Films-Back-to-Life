import torch
from torch.utils import data as data

try:
    from .VP_code.test import load_model, load_dataset, restore
except:
    from VP_code.test import load_model, load_dataset, restore

import folder_paths

class LoadRestoreOldPhotosModel:
    RETURN_TYPES = ("BOFBTL_MODEL",)
    RETURN_NAMES = ("bofbtl_model",)
    FUNCTION = "restore"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("checkpoints"),),
                "model_type": ("STRING", {"default": "RNN_Swin_4"}),
            },
        }

    def restore(self, model: str, model_type: str):
        model = load_model(model_type, model)
        model.eval()
        return (model,)

class RestoreOldFilms:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "restore"
    OUTPUT_NODE = True
    CATEGORY = "image"

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "bofbtl_model": ("BOFBTL_MODEL",),
                "image": ("IMAGE",),
                "batch_size": ("INT", {"default": 15, "min": 1, "max": 0xffffffffffffffff, "step": 1}),
                "stride": ("INT", {"default": 3, "min": 1, "max": 0xffffffffffffffff, "step": 1}),
                "normalize": (["True", "False"], {"default": "True"}),
            },
        }

    def restore(self, bofbtl_model, image, batch_size, stride, normalize):
        input_dtype = image.dtype
        input_device = image.device

        normalize = True if normalize == "True" else False
        dataset_val = {
            "name": "REDS4",
            "type": "Film_dataset_1",
            "dataroot_gt": r"C:\Users\CDBPC\ComfyUI\custom_nodes\Bringing-Old-Films-Back-to-Life\test_data\data_1\001",
            "dataroot_lq": r"C:\Users\CDBPC\ComfyUI\custom_nodes\Bringing-Old-Films-Back-to-Life\test_data\data_1\001",
            "val_partition": "REDS4",
            "is_train": False,
            "num_frame": 10,
            "gt_size": [
                640,
                368
            ],
            "scale": 1,
            "interval_list": [
                1
            ],
            "random_reverse": False,
            "use_flip": False,
            "use_rot": False,
            "normalizing": normalize,
            "texture_template": "./noise_data"
        }
        frame_loader = load_dataset(dataset_val)

        restored_frames = []
        for frame_batch in frame_loader:
            restored_frames += restore(bofbtl_model, frame_batch, batch_size, stride, normalize)
        restored_frames = torch.stack(restored_frames)
        restored_frames = restored_frames.permute(0, 2, 3, 1)
        for i in range(len(restored_frames)):
            restored_frames[i] = (restored_frames[i] + 1) / 2
        restored_frames = restored_frames.to(input_device, dtype=input_dtype)

        return (restored_frames,)

NODE_CLASS_MAPPINGS = {
    "BOFBTL_LoadRestoreOldFilmsModel": LoadRestoreOldPhotosModel,
    "BOFBTL_RestoreOldFilms": RestoreOldFilms,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BOFBTL_LoadRestoreOldFilmsModel": "Load Restore Old Photos Model",
    "BOFBTL_RestoreOldFilms": "Restore Old Films",
}

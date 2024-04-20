import sys
sys.path.insert(0, "../fl_sim/")


import os
import subprocess
import shutil
from copy import deepcopy
from tqdm import tqdm
import torch
from torchvision.utils import save_image

from models import *
from models.gan import ResNetDiscriminator, StyleVectorizer
from data_funcs import load_data
from utils.logger import Logger
from utils.tasks import get_task_elements
# import utils.gan_utils as gan_utils

WHICH = "ditto"
TASK = "damnist-fedgan"
MODEL_PATHS = {
    "padpaf": "../outputs/id=damnist-padpaf/task=damnist-fedgan/lr=0.001_0.01/seed=123/model/model.pth.tar",
    "ditto": "../outputs/id=damnist-ditto/task=damnist-fedgan/lr=0.001_0.01/seed=123/model/model.pth.tar",
    "ditto_fedprox": "../outputs/id=damnist-ditto-fedprox/task=damnist-fedgan/lr=0.001_0.01/seed=123/model/model.pth.tar",
}


def load_fedgan_all_styles(global_model, fp, device=None):
    state_dict = torch.load(fp, map_location=device)
    # Global modules
    if "model" in state_dict:
        global_model.load_state_dict(state_dict['model'])
    else:
        global_model.contentD.load_state_dict(state_dict['contentD'])
        global_model.G.load_state_dict(state_dict['G'])

    private_modules = []
    for worker_id in range(len(state_dict)):
        if str(worker_id) not in state_dict:
            continue
        # Local modules
        local_state_dict = state_dict[str(worker_id)]
        if 'model' in local_state_dict:
            modules = {
                "id": worker_id,
                "model": local_state_dict['model'],
            }
        else:
            modules = {
                "id": worker_id,
                "styleD": local_state_dict['styleD'],
                "style_map": local_state_dict['style_map'],
            }
        private_modules.append(modules)
    return global_model, private_modules


if __name__ == "__main__":

    Logger.setup_logging("INFO", logfile="")
    Logger()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size = 32

    print("Model choice:", WHICH)
    print("Device =", device)
    print("Getting task elements...")
    init_model, _, _, _, train_dataset, _ = get_task_elements(TASK, 128, "../data/")
    global_model = init_model().to(device)
    global_model, private_modules = load_fedgan_all_styles(global_model, MODEL_PATHS[WHICH], device=device)

    @torch.no_grad()
    def generate_samples(model, m, num_samples=100):
        if "model" in m:
            model.load_state_dict(m["model"])
        else:
            model.style_map.load_state_dict(m["style_map"])
        print(f"Generating {num_samples} samples.")
        data = []
        for _ in tqdm(range(num_samples // batch_size + 1)):
            content_latent = torch.randn(batch_size, model.num_latents).to(device)
            style_latent = torch.randn(batch_size, model.num_latents).to(device)
            fake = model.G(content_latent, cond=model.style_map(style_latent))
            data.append(fake)
        return torch.cat(data)


    def save_samples(data, outdir="samples"):
        os.makedirs(outdir, exist_ok=True)
        print("Saving data to:", outdir)
        for i, data_i in tqdm(enumerate(data)):
            if len(data_i) == 2:
                data_i = data_i[0].float()  # get img only
            save_image(data_i, f"{outdir}/{i:04}.png", normalize=True, range=(-1,1))


    outputs = []
    # Generate and save samples
    for m in private_modules:
        client_dataset = [x for x in train_dataset[m['id']]]  # invoking __getitem__ to apply transform
        print("\n==> Client", m['id'])
        print("Generating samples...")
        client_samples = generate_samples(global_model, m, num_samples=len(client_dataset))
        save_samples(client_dataset, f"samples/{WHICH}_real_samples_{m['id']}")
        save_samples(client_samples, f"samples/{WHICH}_fake_samples_{m['id']}")


        # Run pytorch_fid
        print("Calculating FIDs...")
        command = [
            f"python", "-m",  "pytorch_fid",
            f"samples/{WHICH}_real_samples_{m['id']}",
            f"samples/{WHICH}_fake_samples_{m['id']}",
        ]
        if device != 'cpu':
            command += [f"--device=cuda:0"]
        p = subprocess.run(command, capture_output=True, text=True)
        print(p.stdout)
        outputs.append(p.stdout)

        # Clean directory
        print("Cleaning...")
        shutil.rmtree(f"samples/{WHICH}_real_samples_{m['id']}")
        shutil.rmtree(f"samples/{WHICH}_fake_samples_{m['id']}")

    # Report FIDs
    fids = [float(out.split("FID:")[1]) for out in outputs]  # extract fid from text
    mean_fid = sum(fids) / len(fids)
    print("FIDs =", fids)
    print("Mean FID =", mean_fid)
import json
import os
import time
import torch
from tqdm import tqdm

from lavis.models import load_model_and_preprocess
from PIL import Image
from googletrans import Translator

device = torch.device("cuda")
# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption",
                                                     model_type="base_coco",
                                                     is_eval=True,
                                                     device=device)
# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
# generate caption
print(device)
root = r'F:\image_caption\images'
data = {}
for path in tqdm(os.listdir(root)):
    start = time.time()
    raw_image = Image.open(os.path.join(root, path)).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    result = model.generate({"image": image})
    trans = Translator()
    result = [trans.translate(item, src='en', dest='vi').text for item in result]
    data['path'] = result
open(r'./data.json', 'w', encoding='utf-8').write(json.dumps(data))

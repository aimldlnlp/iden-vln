"""Frozen local CLIP: 2x2 spatial cells per camera and ordered text embeddings."""
from functools import lru_cache
from pathlib import Path
import hashlib
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

CHECKPOINT=Path('/home/aimldl/LH-VLN/data/models/clip-vit-base-patch16')

class SpatialCLIPFeatures:
    HASH_DIM=64
    TEXT_START=111+3*4*512
    INPUT_DIM=TEXT_START+512

    def __init__(self, base, device):
        self.base, self.device = base, device
        self.model=CLIPModel.from_pretrained(CHECKPOINT,local_files_only=True).to(device).eval()
        self.model.requires_grad_(False)
        self.processor=CLIPProcessor.from_pretrained(CHECKPOINT,local_files_only=True)
        assert self.model.config.projection_dim==512

    def __getattr__(self, name):
        return getattr(self.base,name)

    @lru_cache(maxsize=64)
    def encode_text(self, text):
        if not text:
            return np.zeros(512,dtype=np.float32)
        tokens=self.processor.tokenizer(text,return_tensors='pt',padding=True,truncation=True,max_length=77)
        with torch.inference_mode():
            value=self.model.get_text_features(**{k:v.to(self.device) for k,v in tokens.items()})
            value=torch.nn.functional.normalize(value,dim=-1)
        return value[0].cpu().numpy().copy()

    def live_feature(self, images, pos, yaw, previous, context):
        # ponytail: frozen 2x2 cells; denser patch attention if larger cohorts justify it.
        pixels=self.processor(images=images,return_tensors='pt')['pixel_values'].to(self.device)
        with torch.inference_mode():
            patches=self.model.vision_model(pixel_values=pixels).last_hidden_state[:,1:]
            patches=self.model.vision_model.post_layernorm(patches)
            patches=self.model.visual_projection(patches)
            side=int(patches.shape[1]**0.5)
            assert side*side==patches.shape[1]
            grid=patches.transpose(1,2).reshape(3,512,side,side)
            grid=torch.nn.functional.adaptive_avg_pool2d(grid,(2,2))
            grid=torch.nn.functional.normalize(grid,dim=1).flatten().cpu().numpy()
        value=np.concatenate([self.base.live_feature(images,pos,yaw,previous,context)[:111],grid,self.encode_text(context['instruction'])])
        assert value.shape==(self.INPUT_DIM,) and np.isfinite(value).all() and not value[47:111].any()
        return value.tolist()

    def protocol(self, rows):
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CHECKPOINT.iterdir() if p.is_file()}
        return {'encoder':'local pretrained CLIP ViT-B/16, frozen','checkpoint_files_sha256':files,
            'input_dim':self.INPUT_DIM,'visual_features':'L/F/R camera; CLIP projected patch grid pooled to 2x2 spatial cells per camera; cell L2 norm',
            'text_features':'CLIP positional transformer text vector, L2 norm; empty control is zero vector',
            'text_token_limit':77,'raw_text_token_counts':{str(r['index']):len(self.processor.tokenizer(r['policy_context']['instruction'])['input_ids']) for r in rows},
            'limits':'CLIP text mainly English; code-switching/Indonesian may be weak; pretrained corpus overlap with environments not audited; GRU parameter count increases with input dimension; changes visual and text features together.'}

    def check(self):
        from PIL import Image
        image=np.zeros((224,224,3),dtype=np.uint8); image[:,:112,0]=255; image[:,112:,2]=255
        ctx={'instruction':'go left then right','robot':'spot','objects':[],'regions':[],'level':2}
        a=self.live_feature([Image.fromarray(image)]*3,[0,0,0],0,-1,ctx)
        b=self.live_feature([Image.fromarray(image[:,::-1].copy())]*3,[0,0,0],0,-1,ctx)
        assert not np.allclose(a[111:self.TEXT_START],b[111:self.TEXT_START])
        assert not np.allclose(self.encode_text('go left then right'),self.encode_text('go right then left'))
        assert not self.encode_text('').any()
        print('CLIP check passed: spatial reversal and word-order change affect features',flush=True)

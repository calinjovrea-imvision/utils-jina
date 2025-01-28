# %% [code] {"id":"Za86vmBb0bVT","outputId":"c644a74c-fd49-4cc0-e022-c43facb594b9","execution":{"iopub.status.busy":"2025-01-28T18:54:58.661815Z","iopub.execute_input":"2025-01-28T18:54:58.662143Z","iopub.status.idle":"2025-01-28T18:56:11.111385Z","shell.execute_reply.started":"2025-01-28T18:54:58.662118Z","shell.execute_reply":"2025-01-28T18:56:11.110337Z"},"jupyter":{"outputs_hidden":false}}
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL_JINA_NAME = "jinaai/jina-clip-v2"
HUGGINGFACE_CACHE_DIR = "./cache"

# Adjust the processor to resize images to (224, 224)
processor_jina = AutoProcessor.from_pretrained(
    MODEL_JINA_NAME,
    cache_dir=HUGGINGFACE_CACHE_DIR,
    trust_remote_code=True
)

# Custom processing to resize input images
processor_jina.feature_extractor.size = (224, 224)

# The model setup remains the same
model_jina = AutoModel.from_pretrained(
    MODEL_JINA_NAME,
    cache_dir=HUGGINGFACE_CACHE_DIR,
    trust_remote_code=True,
    attn_implementation="eager"
).to("cuda").float()

tokenizer_jina = AutoTokenizer.from_pretrained(
    MODEL_JINA_NAME,
    cache_dir=HUGGINGFACE_CACHE_DIR,
    trust_remote_code=True,
    use_fast=True
)


!git clone https://github.com/openai/CLIP.git
# #https://github.com/openai/CLIP
# #CLIP (Contrastive Language-Image Pre-Training)
# #Learning Transferable Visual Models From Natural Language Supervision
# #Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
# #Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever

!git clone https://github.com/CompVis/taming-transformers

# #https://github.com/CompVis/taming-transformers
# #Taming Transformers for High-Resolution Image Synthesis
# #Patrick Esser, Robin Rombach, Björn Ommer

# %% [code] {"id":"uY8i3Kt4jvY2","outputId":"b0d17f1a-330f-46fc-9545-dbb998ba1c08","execution":{"iopub.status.busy":"2025-01-28T17:11:46.687086Z","iopub.execute_input":"2025-01-28T17:11:46.687403Z","iopub.status.idle":"2025-01-28T17:11:47.161701Z","shell.execute_reply.started":"2025-01-28T17:11:46.687376Z","shell.execute_reply":"2025-01-28T17:11:47.160726Z"},"jupyter":{"outputs_hidden":false}}

# %% [code] {"id":"8wofcATFhlOn","outputId":"3b655fea-8a75-4580-9e9c-0c54106e58e0","execution":{"iopub.status.busy":"2025-01-28T17:11:49.446219Z","iopub.execute_input":"2025-01-28T17:11:49.446553Z","iopub.status.idle":"2025-01-28T17:11:49.631942Z","shell.execute_reply.started":"2025-01-28T17:11:49.446526Z","shell.execute_reply":"2025-01-28T17:11:49.630699Z"},"jupyter":{"outputs_hidden":false}}


# %% [code] {"id":"ompo3S_w7Ej6","outputId":"6ea4d69b-b399-4c69-85f7-217af1603871","execution":{"iopub.status.busy":"2025-01-28T16:57:00.136454Z","iopub.execute_input":"2025-01-28T16:57:00.136715Z","iopub.status.idle":"2025-01-28T16:57:08.740826Z","shell.execute_reply.started":"2025-01-28T16:57:00.136692Z","shell.execute_reply":"2025-01-28T16:57:08.739940Z"},"jupyter":{"outputs_hidden":false}}
## install some extra libraries
!pip install --no-deps ftfy regex tqdm
!pip install omegaconf==2.0.0 pytorch-lightning==1.0.8
!pip uninstall torchtext --yes
!pip install einops

# %% [code] {"id":"Zn_gTOEw7h8x","execution":{"iopub.status.busy":"2025-01-28T16:57:08.742204Z","iopub.execute_input":"2025-01-28T16:57:08.742461Z","iopub.status.idle":"2025-01-28T16:57:08.747399Z","shell.execute_reply.started":"2025-01-28T16:57:08.742437Z","shell.execute_reply":"2025-01-28T16:57:08.746481Z"},"jupyter":{"outputs_hidden":false}}
# import libraries
import numpy as np
import torch, os, imageio, pdb, math
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import PIL
import matplotlib.pyplot as plt

import yaml
from omegaconf import OmegaConf

# from CLIP import clip

#import warnings
#warnings.filterwarnings('ignore')

# %% [code] {"id":"pZehdKop7_bK","execution":{"iopub.status.busy":"2025-01-28T16:57:13.077861Z","iopub.execute_input":"2025-01-28T16:57:13.078142Z","iopub.status.idle":"2025-01-28T16:57:13.083751Z","shell.execute_reply.started":"2025-01-28T16:57:13.078119Z","shell.execute_reply":"2025-01-28T16:57:13.082910Z"},"jupyter":{"outputs_hidden":false}}
## helper functions

def show_from_tensor(tensor):
  img = tensor.clone()
  img = img.mul(255).byte()
  img = img.cpu().numpy().transpose((1,2,0))

  plt.figure(figsize=(10,7))
  plt.axis('off')
  plt.imshow(img)
  plt.show()

def norm_data(data):
  return (data.clip(-1,1)+1)/2 ### range between 0 and 1 in the result

### Parameters
learning_rate = .5
batch_size = 1
wd = .1
noise_factor = .22

total_iter=800
im_shape = [450, 450, 3] # height, width, channel
size1, size2, channels = im_shape

# %% [code] {"id":"JfApIAoR-N55","outputId":"e0b5609b-4021-41fe-c17b-02e0eac94241","execution":{"iopub.status.busy":"2025-01-28T16:57:14.128162Z","iopub.execute_input":"2025-01-28T16:57:14.128451Z","iopub.status.idle":"2025-01-28T16:57:14.499134Z","shell.execute_reply.started":"2025-01-28T16:57:14.128429Z","shell.execute_reply":"2025-01-28T16:57:14.498335Z"},"jupyter":{"outputs_hidden":false}}
### CLIP MODEL ###
# clipmodel, _ = clip.load('ViT-B/32', jit=False)
clipmodel = model_jina
clipmodel.eval()
# print(clip.available_models())
print(dir(model_jina))
# print("Clip model visual input resolution: ", clipmodel.visual.input_resolution)

device=torch.device("cuda:0")
torch.cuda.empty_cache()

# %% [code] {"id":"ow5gAnmVhNuT","outputId":"e24dbe37-7cd1-4e2a-d545-b4026e2253e4","execution":{"iopub.status.busy":"2025-01-28T16:57:14.500418Z","iopub.execute_input":"2025-01-28T16:57:14.500674Z","iopub.status.idle":"2025-01-28T16:57:14.687342Z","shell.execute_reply.started":"2025-01-28T16:57:14.500652Z","shell.execute_reply":"2025-01-28T16:57:14.686392Z"},"jupyter":{"outputs_hidden":false}}
!ls

# %% [code] {"id":"C_MJ3iNdAW7Z","execution":{"iopub.status.busy":"2025-01-28T16:57:14.917291Z","iopub.execute_input":"2025-01-28T16:57:14.917566Z","iopub.status.idle":"2025-01-28T16:59:23.543060Z","shell.execute_reply.started":"2025-01-28T16:57:14.917540Z","shell.execute_reply":"2025-01-28T16:59:23.541931Z"},"jupyter":{"outputs_hidden":false}}
## Taming transformer instantiation

%cd taming-transformers/

!mkdir -p models/vqgan_imagenet_f16_16384/checkpoints
!mkdir -p models/vqgan_imagenet_f16_16384/configs

if len(os.listdir('models/vqgan_imagenet_f16_16384/checkpoints/')) == 0:
   !wget 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1' -O 'models/vqgan_imagenet_f16_16384/checkpoints/last.ckpt'
   !wget 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1' -O 'models/vqgan_imagenet_f16_16384/configs/model.yaml'

!curl -o utils.py https://raw.githubusercontent.com/calinjovrea-imvision/utils-jina/refs/heads/main/utils.py
!mv utils.py /taming/data/utils.py
!cat taming/data/utils.py

# %% [code] {"execution":{"iopub.status.busy":"2025-01-28T17:12:02.870737Z","iopub.execute_input":"2025-01-28T17:12:02.871132Z","iopub.status.idle":"2025-01-28T17:12:09.663749Z","shell.execute_reply.started":"2025-01-28T17:12:02.871101Z","shell.execute_reply":"2025-01-28T17:12:09.662707Z"}}
from taming.models.vqgan import VQModel

def load_config(config_path, display=False):
   config_data = OmegaConf.load(config_path)
   if display:
     print(yaml.dump(OmegaConf.to_container(config_data)))
   return config_data

def load_vqgan(config, chk_path=None):
  model = VQModel(**config.model.params)
  if chk_path is not None:
    state_dict = torch.load(chk_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
  return model.eval()

def generator(x):
  x = taming_model.post_quant_conv(x)
  x = taming_model.decoder(x)
  return x

taming_config = load_config("./models/vqgan_imagenet_f16_16384/configs/model.yaml", display=True)
taming_model = load_vqgan(taming_config, chk_path="./models/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to(device)


# %% [code] {"id":"ftDHxwo_B2F_","outputId":"8bff8943-915e-4017-f56e-f738e97e2405","jupyter":{"outputs_hidden":false}}


# %% [code] {"id":"hY6dOXE7Drgn","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-01-28T17:12:17.140345Z","iopub.execute_input":"2025-01-28T17:12:17.140663Z","iopub.status.idle":"2025-01-28T17:12:17.145661Z","shell.execute_reply.started":"2025-01-28T17:12:17.140637Z","shell.execute_reply":"2025-01-28T17:12:17.144917Z"}}
### Declare the values that we are going to optimize

class Parameters(torch.nn.Module):
  def __init__(self):
    super(Parameters, self).__init__()
    self.data = .5*torch.randn(batch_size, 256, size1//16, size2//16).cuda() # 1x256x14x15 (225/16, 400/16)
    self.data = torch.nn.Parameter(torch.sin(self.data))

  def forward(self):
    return self.data

def init_params():
  params=Parameters().cuda()
  optimizer = torch.optim.AdamW([{'params':[params.data], 'lr': learning_rate}], weight_decay=wd)
  return params, optimizer

# %% [code] {"id":"BsMOgdHPGFI8","outputId":"40b24266-8907-4ba3-e3ec-46f85389841d","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-01-28T17:12:18.744908Z","iopub.execute_input":"2025-01-28T17:12:18.745227Z","iopub.status.idle":"2025-01-28T17:12:20.726402Z","shell.execute_reply.started":"2025-01-28T17:12:18.745199Z","shell.execute_reply":"2025-01-28T17:12:20.725599Z"}}
### Encoding prompts and a few more things
normalize = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

def encodeText(text):
  t=tokenizer_jina.tokenize(text)
  t=model_jina.encode_text(t)
  return t

def createEncodings(include, exclude, extras):
  include_enc=[]
  for text in include:
    include_enc.append(encodeText(text))
  exclude_enc=encodeText(exclude) if exclude != '' else 0
  extras_enc=encodeText(extras) if extras !='' else 0

  return include_enc, exclude_enc, extras_enc

augTransform = torch.nn.Sequential(
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomAffine(30, (.2, .2), fill=0)
).cuda()

Params, optimizer = init_params()

with torch.no_grad():
  print(Params().shape)
  img= norm_data(generator(Params()).cpu()) # 1 x 3 x 224 x 400 [225 x 400]
  print("img dimensions: ",img.shape)
  show_from_tensor(img[0])

# %% [code] {"id":"XsgmO1LeI_5e","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-01-28T17:12:24.624212Z","iopub.execute_input":"2025-01-28T17:12:24.624511Z","iopub.status.idle":"2025-01-28T17:12:24.631524Z","shell.execute_reply.started":"2025-01-28T17:12:24.624484Z","shell.execute_reply":"2025-01-28T17:12:24.630510Z"}}
### create crops

def create_crops(img, num_crops=32):
  p=size1//2
  img = torch.nn.functional.pad(img, (p,p,p,p), mode='constant', value=0) # 1 x 3 x 448 x 624 (adding 112*2 on all sides to 224x400)

  img = augTransform(img) #RandomHorizontalFlip and RandomAffine

  crop_set = []
  for ch in range(num_crops):
    gap1= int(torch.normal(1.2, .3, ()).clip(.43, 1.9) * size1)
    offsetx = torch.randint(0, int(size1*2-gap1),())
    offsety = torch.randint(0, int(size1*2-gap1),())

    crop=img[:,:,offsetx:offsetx+gap1, offsety:offsety+gap1]

    crop = torch.nn.functional.interpolate(crop,(224,224), mode='bilinear', align_corners=True)
    crop_set.append(crop)

  img_crops=torch.cat(crop_set,0) ## 30 x 3 x 224 x 224

  randnormal = torch.randn_like(img_crops, requires_grad=False)
  num_rands=0
  randstotal=torch.rand((img_crops.shape[0],1,1,1)).cuda() #32

  for ns in range(num_rands):
    randstotal*=torch.rand((img_crops.shape[0],1,1,1)).cuda()

  img_crops = img_crops + noise_factor*randstotal*randnormal

  return img_crops

# %% [code] {"id":"lwESrZefM0Vt","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-01-28T17:12:27.483911Z","iopub.execute_input":"2025-01-28T17:12:27.484252Z","iopub.status.idle":"2025-01-28T17:12:27.488919Z","shell.execute_reply.started":"2025-01-28T17:12:27.484219Z","shell.execute_reply":"2025-01-28T17:12:27.488153Z"}}
### Show current state of generation

def showme(Params, show_crop):
  with torch.no_grad():
    generated = generator(Params())

    if (show_crop):
      print("Augmented cropped example")
      aug_gen = generated.float() # 1 x 3 x 224 x 400
      aug_gen = create_crops(aug_gen, num_crops=1)
      aug_gen_norm = norm_data(aug_gen[0])
      show_from_tensor(aug_gen_norm)

    print("Generation")
    latest_gen=norm_data(generated.cpu()) # 1 x 3 x 224 x 400
    show_from_tensor(latest_gen[0])

  return (latest_gen[0])

# %% [code] {"id":"6uV4VJn7N-FI","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-01-28T17:12:28.258344Z","iopub.execute_input":"2025-01-28T17:12:28.258672Z","iopub.status.idle":"2025-01-28T17:12:28.266888Z","shell.execute_reply.started":"2025-01-28T17:12:28.258639Z","shell.execute_reply":"2025-01-28T17:12:28.266077Z"}}
# Optimization process
from PIL import Image
import numpy as np

def optimize_result(Params, prompt):
  alpha=1 ## the importance of the include encodings
  beta=.5 ## the importance of the exclude encodings

  ## image encoding
  # out = generator(Params())
  # out = norm_data(out)
  # out = create_crops(out)
  # out = normalize(out) # 30 x 3 x 224 x 224
  # out = out.cpu().detach().numpy()[0][0]
  # print(out.shape)
  # out = np.clip(out, 0, 255).astype(np.uint8)
  # out = Image.fromarray(out)
  # print(out)
  # image_enc=clipmodel.encode_image(out) ## 30 x 512
  # print(image_enc)
  # print(type(image_enc))
  # print(image_enc.shape)
  # ## text encoding  w1 and w2
  # final_enc = w1*prompt + w1*extras_enc # prompt and extras_enc : 1 x 512
  # final_text_include_enc = final_enc / final_enc.norm(dim=-1, keepdim=True) # 1 x 512
  # final_text_exclude_enc = exclude_enc

  # Image encoding pipeline
  out = generator(Params())
  out = norm_data(out)
  out = create_crops(out)
  out = normalize(out)  # Tensor: shape (30, 3, 224, 224)

  print(out.shape)
  # Optionally move to CPU if required, otherwise keep on GPU
  # out = out[0, 0]  # Selecting the first crop (for simplicity)

  # # Convert the tensor to uint8 format and clip values between 0 and 255
  # out = torch.clamp(out, 0, 255).to(torch.uint8)

  # print(out.shape)

  # # Convert the tensor to a PIL Image for display or further processing
  # image_out = Image.fromarray(out.permute(1, 2, 0).cpu().numpy())

  # # Encode the image with clipmodel
  # image_enc = clipmodel.encode_image(out.unsqueeze(0))  # (1, 512)

  # out = torch.clamp(out, 0, 255).to(torch.uint8)
  # print(out.shape)
  # Convert the 2D tensor to a NumPy array (height x width)
  # out_np = out.cpu().numpy()

  # Convert to PIL Image
  # image_out = Image.fromarray(out_np)

  # Display the image (optional)
  # image_out.show()

  # Encode the image with clipmodel (if needed)
  # image_enc = clipmodel.encode_image(image_out)

  # Print out the encoding shape for verification
  # Assuming `out` has the shape (32, 3, 224, 224)
  out = torch.clamp(out, 0, 255).to(torch.uint8)  # Clamp values to valid range for images
  print(out.shape)  # Should print torch.Size([32, 3, 224, 224])

  # Move to CPU and convert to NumPy (if needed for PIL or processing)
  out_np = out.permute(0, 2, 3, 1).cpu().numpy()  # Convert to (32, 224, 224, 3) format for PIL

  # Convert each image to PIL and encode all 32 images
  image_encodings = []
  for i in range(out_np.shape[0]):  # Iterate through all 32 images
      # print(out_np[i].shape)
      # img_np = out[i].permute(1, 2, 0).cpu().numpy()  # Convert CHW to HWC
      img_pil = Image.fromarray(out_np[i], mode="RGB")
      # image_out = Image.fromarray(out_np[i])  # Convert the ith image to PIL format
      image_enc = model_jina.encode_image(img_pil)  # Encode ith image
      image_encodings.append(torch.tensor(image_enc))

  # Stack all image encodings into a single tensor
  image_encodings = torch.stack(image_encodings, dim=0)
  print("Image encoding shape:", image_encodings.shape)

  final_embedding = image_encodings.mean(dim=0)  # Shape: (1024)
  print(f"final_embedding {final_embedding.shape}")

  # Text encoding with w1 and w2
  final_enc = w1 * prompt + w1 * extras_enc  # (1, 512)

  final_enc = torch.tensor(final_enc, requires_grad=True)  # Convert NumPy array to tensor if needed

  # Normalize the final text encoding
  final_text_include_enc = final_enc / final_enc.norm(dim=-1, keepdim=True)  # (1, 512)
  final_text_exclude_enc = exclude_enc  # Assuming exclude_enc is already a tensor

  # image_enc = torch.tensor(image_enc)
  final_text_exclude_enc = torch.tensor(final_text_exclude_enc)

  ## calculate the loss
  main_loss = torch.cosine_similarity(final_text_include_enc, final_embedding, -1) # 30
  penalize_loss = torch.cosine_similarity(final_text_exclude_enc, final_embedding, -1) # 30

  final_loss = -alpha*main_loss + beta*penalize_loss

  return final_loss

def optimize(Params, optimizer, prompt):
  loss = optimize_result(Params, prompt).mean()
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss

# %% [code] {"id":"9vpVuN8iQuLb","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-01-28T17:12:35.059043Z","iopub.execute_input":"2025-01-28T17:12:35.059334Z","iopub.status.idle":"2025-01-28T17:12:35.064561Z","shell.execute_reply.started":"2025-01-28T17:12:35.059309Z","shell.execute_reply":"2025-01-28T17:12:35.063646Z"}}
### training loop

def training_loop(Params, optimizer, show_crop=False):
  res_img=[]
  res_z=[]

  for prompt in include_enc:
    iteration=0
    Params, optimizer = init_params() # 1 x 256 x 14 x 25 (225/16, 400/16)

    for it in range(total_iter):
      loss = optimize(Params, optimizer, prompt)

      if iteration>=80 and iteration%show_step == 0:
        new_img = showme(Params, show_crop)
        res_img.append(new_img)
        res_z.append(Params()) # 1 x 256 x 14 x 25
        print("loss:", loss.item(), "\niteration:",iteration)

      iteration+=1
    torch.cuda.empty_cache()
  return res_img, res_z

# %% [code] {"id":"9q4kwgudSuI5","outputId":"45e757e4-0c2a-4839-aeb7-f711cba38021","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-01-28T17:12:37.400336Z","iopub.execute_input":"2025-01-28T17:12:37.400636Z","iopub.status.idle":"2025-01-28T18:06:26.788726Z","shell.execute_reply.started":"2025-01-28T17:12:37.400609Z","shell.execute_reply":"2025-01-28T18:06:26.787545Z"}}
torch.cuda.empty_cache()
include=['sketch of a lady', 'sketch of a man on a horse']
# include=['A painting of a pineapple in a bowl']
exclude='watermark'
extras = ""
w1=1
w2=1
noise_factor= .22
total_iter=110
show_step=10 # set this to see the result every 10 interations beyond iteration 80
include_enc, exclude_enc, extras_enc = createEncodings(include, exclude, extras)
res_img, res_z=training_loop(Params, optimizer, show_crop=True)

# %% [code] {"id":"ScpytmgQUffY","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-01-28T18:54:49.060625Z","iopub.execute_input":"2025-01-28T18:54:49.060986Z","iopub.status.idle":"2025-01-28T18:54:49.132017Z","shell.execute_reply.started":"2025-01-28T18:54:49.060954Z","shell.execute_reply":"2025-01-28T18:54:49.130667Z"}}
print(len(res_img), len(res_z))
print(res_img[0].shape, res_z[0].shape)
print(res_z[0].max(), res_z[0].min())

# %% [code] {"id":"W_NnBA9AXaqg","outputId":"a6649c66-cd6c-4570-97ee-196db0e1bee2","jupyter":{"outputs_hidden":false},"execution":{"execution_failed":"2025-01-28T18:46:48.711Z"}}
torch.cuda.empty_cache()
include=['A forest with purple trees', 'an elephant at the top of a mountain, looking at the stars','one hundred people with green jackets']
exclude='watermark, cropped, confusing, incoherent, cut, blurry'
extras = ""
w1=1
w2=1
noise_factor= .22
total_iter=110
show_step=total_iter-1 # set this if you want to interpolate between only the final versions
include_enc, exclude_enc, extras_enc = createEncodings(include, exclude, extras)
res_img, res_z=training_loop(Params, optimizer, show_crop=False)

# %% [code] {"id":"c3lTWZaDYBJR","jupyter":{"outputs_hidden":false},"execution":{"execution_failed":"2025-01-28T18:46:48.711Z"}}
def interpolate(res_z_list, duration_list):
  gen_img_list=[]
  fps = 25

  for idx, (z, duration) in enumerate(zip(res_z_list, duration_list)):
    num_steps = int(duration*fps)
    z1=z
    z2=res_z_list[(idx+1)%len(res_z_list)] # 1 x 256 x 14 x 25 (225/16, 400/16)

    for step in range(num_steps):
      alpha = math.sin(1.5*step/num_steps)**6
      z_new = alpha * z2 + (1-alpha) * z1

      new_gen=norm_data(generator(z_new).cpu())[0] ## 3 x 224 x 400
      new_img=T.ToPILImage(mode='RGB')(new_gen)
      gen_img_list.append(new_img)

  return gen_img_list

durations=[5,5,5,5,5,5]
interp_result_img_list = interpolate(res_z, durations)

# %% [code] {"id":"WeSmO7WRasl_","jupyter":{"outputs_hidden":false},"execution":{"execution_failed":"2025-01-28T18:46:48.711Z"}}
## create a video
out_video_path=f"../video.mp4"
writer = imageio.get_writer(out_video_path, fps=25)
for pil_img in interp_result_img_list:
  img = np.array(pil_img, dtype=np.uint8)
  writer.append_data(img)

writer.close()

# %% [code] {"id":"VmlxcyLzbbu_","outputId":"3fcc0859-a71b-4b5f-f15c-8d2f64c64914","jupyter":{"outputs_hidden":false},"execution":{"execution_failed":"2025-01-28T18:46:48.711Z"}}
from IPython.display import HTML
from base64 import b64encode

mp4 = open('../video.mp4','rb').read()
data="data:video/mp4;base64,"+b64encode(mp4).decode()
HTML("""<video width=800 controls><source src="%s" type="video/mp4"></video>""" % data)

# %% [code] {"id":"8g77m5Urb8YH","jupyter":{"outputs_hidden":false}}

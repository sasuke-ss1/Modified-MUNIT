import DA_net.utils as utils
from argparse import ArgumentParser
import DA_net.S_reconstruction as Srecon
import DA_net.R_reconstruction as Rrecon
import DA_net.End2End as End2End
import torch

iterations = 0
def train(opt,models):
    fake = utils.Folder_data(opt.fake, opt.batch_size, train=True)
    clear = utils.Folder_data(opt.clear, opt.batch_size, train=True)
    real = utils.Folder_data(opt.real, opt.batch_size, train=True)
    while True:
        for it, (fake_img, real_img, clear_img) in enumerate(zip(fake, real, clear)):
            model = models(opt.m_config, opt.res_config, opt.d_config)
            model.optimize(fake_img, real_img, clear_img)
            iterations += 1
            if(iterations % opt.save_image):
                #utils.save_images(model.r2s_recon_img, model.s_recon_img, fake_img, real_img, clear_img)
                print(f"{iterations}  G: {model.netLoss}   D:{model.loss_D}")
                torch.save(model.state_dict(), f"./chekpoints/{model}.pth")
            if(iterations >= opt.max_iter):
                return
                


parser = ArgumentParser()
parser.add_argument("--model", type=int, help ="Model to train", required=True)
parser.add_argument('--fake', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the fake images.')
parser.add_argument('--real', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the real images.')
parser.add_argument('--clean', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the clean images.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--bacth_size', type=int, default=1, help='batch_size')
parser.add_argument('--max_iter', type=int, default=100000, help="max iterations")
parser.add_argument('--save_image', type=int, default=5000, help='save images iterations')
parser.add_argument('--m_config', type=str, default='', help='Path to config files for MUNIT.')
parser.add_argument('--res_config', type=str, default='', help='Path to config files for Resnet.')
parser.add_argument('--d_config', type=str, default='', help='Path to config files for Discriminator.')
opts = parser.parse_args()
model_dict ={
    0:Srecon,
    1:Rrecon,
    2:End2End
}
train(opts, model_dict[opts.model])


    

import DA_net.utils as utils
from argparse import ArgumentParser
import DA_net.S_reconstruction as Srecon
import DA_net.R_reconstruction as Rrecon
import DA_net.End2End as End2End
import torch
log = open("./checkpoints/S_log.txt", 'a')
iterations = 0
def train(opt):
    fake = utils.Folder_data(opt.fake, opt.batch_size, train=False)
    clear = utils.Folder_data(opt.clear, opt.batch_size, train=False)
    real = utils.Folder_data(opt.real, opt.batch_size, train=True)
    iterations = 0
    while True:
        for it, (fake_img, real_img, clear_img) in enumerate(zip(fake, real, clear)):
            model = Srecon.S_recon(opt.m_config, opt.res_config, opt.d_config).cuda()
            model.optimize(fake_img, real_img, clear_img)
            iterations += 1
            if(not iterations % opt.save_image):
                utils.save_images(model.r2s_recon_img.detach(), model.s_recon_img.detach(), fake_img.detach(), real_img.detach(), clear_img.detach(), iterations)
                log.write(f"{iterations}  G: {model.netLoss}   D:{model.loss_D}")
                
                #print(model.s_recon_img.shape, model.r2s_recon_img.shape )
                torch.save(model.state_dict(), f"./checkpoints/S_weight/Srecon_{iterations}.pth")
            if(iterations >= opt.max_iter):
                return
                


parser = ArgumentParser()
parser.add_argument("--model", type=int, help ="Model to train", required=True)
parser.add_argument('--fake', type=str, default='/home/neham/uw_datasets/UWCNN_NYU/type1_data/underwater_type_1/', help='Path to the fake images.')
parser.add_argument('--real', type=str, default='/home/neham/uw_datasets/trainA/', help='Path to the real images.')
parser.add_argument('--clear', type=str, default='/home/neham/uw_datasets/UWCNN_NYU/type1_data/gt_type_type_1/', help='Path to the clean images.')
parser.add_argument('--output_path', type=str, default='/home/intern/ss_sasuke/outputs/', help="outputs path")
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--max_iter', type=int, default=100000, help="max iterations")
parser.add_argument('--save_image', type=int, default=5000, help='save images iterations')
parser.add_argument('--m_config', type=str, default='/home/intern/ss_sasuke/Modified-MUNIT/MUNIT/config.yaml', help='Path to config files for MUNIT.')
parser.add_argument('--res_config', type=str, default='/home/intern/ss_sasuke/Modified-MUNIT/DA_net/res_config:.yml', help='Path to config files for Resnet.')
parser.add_argument('--d_config', type=str, default='/home/intern/ss_sasuke/Modified-MUNIT/DA_net/D_config.yml', help='Path to config files for Discriminator.')
parser.add_argument("--e2eConfig", type=str, default="/home/intern/ss_sasuke/Modified-MUNIT/DA_net/e2eConfig.yml", help="Path to config for End2End")
opts = parser.parse_args()

train(opts)


    

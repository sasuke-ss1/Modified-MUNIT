from py import process
import DA_net.utils as utils
from argparse import ArgumentParser
import DA_net.S_reconstruction as Srecon
import DA_net.R_reconstruction as Rrecon
import DA_net.End2End as End2End
import torch
import os
from time import process_time


iterations = 0
def train(opt, log):
    fake = utils.Folder_data(opt.fake, opt.batch_size, train=False)
    clear = utils.Folder_data(opt.clear, opt.batch_size, train=False)
    real = utils.Folder_data(opt.real, opt.batch_size, train=True)
    iterations = 0
    tmp = [("S_weights", "Srecon", "S_images"), ("R_weights", "Rrecon", "R_images"), ("e2e_weights", "e2e_recon", "e2e_images")]
    if opt.model == 0:
        model = Srecon.S_recon(opt.m_config, opt.res_config, opt.d_config).cuda()
    elif opt.model == 1:
        model = Rrecon.R_recon(opt.m_config, opt.res_config, opt.d_config).cuda()
    elif opt.model == 2:
        model = End2End.End2End(opt.m_config, opt.res_config, opt.d_config, opt.e2eConfig)
    else:
        raise NotImplementedError
    while True:
        for it, (fake_img, real_img, clear_img) in enumerate(zip(fake, real, clear)):
            start_time = process_time()
            model.optimize(fake_img, real_img, clear_img)
            iterations += 1
            print(f"{iterations} Elapsed Time: {process_time() - start_time}")
            if(not iterations % opt.save_image):
                model.save(fake_img, real_img, clear_img, f"./checkpoints/{tmp[opt.model][2]}/{iterations}.png")
                torch.save(model.state_dict(), f"./checkpoints/{tmp[opt.model][0]}/{tmp[opt.model][1]}_{iterations}.pth")
            if(not iterations % opt.print_log):
                print(f"{iterations}  G: {model.netLoss}   D:{model.loss_D}", file = log)
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
parser.add_argument('--save_image', type=int, default=5, help='save images iterations')
parser.add_argument("--print_log", type=int, default=2, help ="iteration to save log")
parser.add_argument('--m_config', type=str, default='/home/intern/ss_sasuke/Modified-MUNIT/MUNIT/config.yaml', help='Path to config files for MUNIT.')
parser.add_argument('--res_config', type=str, default='/home/intern/ss_sasuke/Modified-MUNIT/DA_net/res_config:.yml', help='Path to config files for Resnet.')
parser.add_argument('--d_config', type=str, default='/home/intern/ss_sasuke/Modified-MUNIT/DA_net/D_config.yml', help='Path to config files for Discriminator.')
parser.add_argument("--e2eConfig", type=str, default="/home/intern/ss_sasuke/Modified-MUNIT/DA_net/e2eConfig.yml", help="Path to config for End2End")
opts = parser.parse_args()

paths = [("S_images", "S_weights", "S_log"), ("R_images", "R_weights", "R_log"), ("e2e_images", "e2e_weights", "e2e_log")]

try:
    os.mkdir(f"./checkpoints/{paths[opts.model][0]}")
    os.mkdir(f"./checkpoints/{paths[opts.model][1]}")    
except FileExistsError:
    pass
log = open(f"./checkpoints/{paths[opts.model][2]}.txt", 'a',buffering = 1)
train(opts, log)


    
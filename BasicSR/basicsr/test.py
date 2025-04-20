import logging
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    from ptflops import get_model_complexity_info
    import time

    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    model.net_g.eval()
    device = next(model.net_g.parameters()).device

    # ================== FLOPs & Params ==================
    try:
        input_shape = opt['network_g'].get('input_shape', (3, 64, 64))
        flops, params = get_model_complexity_info(model.net_g, input_shape, as_strings=True, print_per_layer_stat=False)
        logger.info(f"[模型统计] FLOPs: {flops}, Params: {params}")
    except Exception as e:
        logger.warning(f"[模型统计] 计算 FLOPs 出错: {e}")

    # ================== 推理时间 ==================
    try:
        dummy_input = torch.randn(1, *input_shape).to(device)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        with torch.no_grad():
            _ = model.net_g(dummy_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed_time = time.time() - start_time
        logger.info(f"[模型统计] 单次推理时间: {elapsed_time * 1000:.3f} ms")
    except Exception as e:
        logger.warning(f"[模型统计] 推理时间测试出错: {e}")


    # 开始验证
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)

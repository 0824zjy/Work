import os

def generate_txt_from_images(image_dir, save_txt_path):
    """
    扫描 image_dir 下所有图片文件，
    生成不带扩展名的 stem 列表 txt
    """
    assert os.path.exists(image_dir), f"{image_dir} 不存在"

    exts = ('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff')
    files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(exts)
    ])

    stems = [os.path.splitext(f)[0] for f in files]

    with open(save_txt_path, 'w') as f:
        for s in stems:
            f.write(s + '\n')

    print(f"生成完成: {save_txt_path}")
    print(f"样本数: {len(stems)}")


if __name__ == "__main__":

    # # ====== 数据集1 ======
    # image_dir1 = "/data/zjy_work/ISIC2018/exp_mask2img_cn_only/images"
    # save_txt1  = "/data/zjy_work/data_txt/train_exp_cn_only.txt"

    # generate_txt_from_images(image_dir1, save_txt1)


    # ====== 数据集2 ======
    image_dir2 = "/data/zjy_work/BGDiff/Ours/ablation_sbg/results/full_sbg_real5_syn15/images"
    save_txt2  = "/data/zjy_work/data_txt/train_PH2_full_sbg_real5_syn15_seed0.txt"

    generate_txt_from_images(image_dir2, save_txt2)

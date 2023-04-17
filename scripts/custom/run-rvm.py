import torch


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--backbone",
                        choices=["mobilenetv3", "resnet50"],
                        default="mobilenetv3")
    args = parser.parse_args()

    model = torch.hub.load("PeterL1n/RobustVideoMatting", args.backbone)
    convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")

    processed_imgs = convert_video(model,
                                   input_source=f"{args.data_dir}/images",
                                   output_type="png_sequence",
                                   output_alpha=f"{args.data_dir}/masks_rvm",
                                   output_composition=f"{args.data_dir}/masked_rvm_images",
                                   device="cuda")
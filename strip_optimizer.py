import torch


# checkpoint-0.pth.tar
# checkpoint-1.pth.tar
# checkpoint-2.pth.tar
# checkpoint-3.pth.tar
# checkpoint-4.pth.tar
# checkpoint-5.pth.tar
# checkpoint-6.pth.tar
# checkpoint-7.pth.tar
# checkpoint-8.pth.tar
# checkpoint-9.pth.tar
# last.pth.tar
# model_best.pth.tar


work_dir = "output/train/20240616-174339-nextvit_large-224/"
for weight in [
    # "checkpoint-0.pth.tar",
    # "checkpoint-1.pth.tar",
    # "checkpoint-2.pth.tar",
    "checkpoint-3.pth.tar",
    "checkpoint-4.pth.tar",
    "checkpoint-5.pth.tar",
    "checkpoint-6.pth.tar",
    "checkpoint-7.pth.tar",
    "checkpoint-8.pth.tar",
    "checkpoint-9.pth.tar",
    "checkpoint-10.pth.tar",
    "checkpoint-11.pth.tar",
    "checkpoint-12.pth.tar",
    "checkpoint-13.pth.tar",
    "checkpoint-14.pth.tar",
]:
    file = f"{work_dir}/{weight}"
    x = torch.load(file, map_location=torch.device("cpu"))
    if "optimizer" in x.keys():
        x["optimizer"] = None
    torch.save(x, file)

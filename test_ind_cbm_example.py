import argparse

import torch
from tqdm import tqdm

from architecture.extended_cbm import ExtendedCBMOutput, init_from_checkpoint
from cbm_datasets import get_transform_cub
from cbm_datasets.cub import SUB

# as in CBM
MAPPING = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
    93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]

ATTRIBUTE_FILE = '/pfs/work9/workspace/scratch/ma_faroesch-master-thesis/datasets/CUB_200_2011/attributes/attributes.txt'
LABEL_PATH = '/pfs/work9/workspace/scratch/ma_faroesch-master-thesis/datasets/CUB_200_2011/class_attr_data_10/val.pkl'
IMG_DIR_PATH = '/pfs/work9/workspace/scratch/ma_faroesch-master-thesis/datasets/CUB_200_2011/images'
N_CLASSES = 200

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--test_for_compliment", action='store_true')
    args = parser.parse_args()
    return args



def load_model(args):

    model = init_from_checkpoint(run_id='17m6p70g', epoch=15, dataset='cub_112', device='cuda')
    model.eval()
    return model




if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    model = load_model(args).to(device)
    stats = []
    stats_nums = []
    total_incorrect = 0
    num_birds = 0

    transform = get_transform_cub(img_size=256, center_crop_size=None)
    test_dataset = SUB(
        root_dir="/pfs/work9/workspace/scratch/ma_faroesch-master-thesis/datasets/CUB_200_2011",
        transform=transform,
        reference_dataset_name="CUB_112",
        LABEL_PATH=LABEL_PATH
    )

    used_attributes = test_dataset.attributes.reset_index(drop=True)

    # column_names = ['file', 'changed_attr', 'bird']
    # img_paths = pd.read_csv(args.file_path, header=None)
    # img_paths.columns = column_names

    pbar = tqdm(range(len(test_dataset)))
    for i in pbar:
        sample = test_dataset[i]

        try:
            used_attr_name = sample['used_attr_name'].replace('--','::')
            bird_name = sample['bird_name']
            s_minus_idx = used_attributes[used_attributes == used_attr_name].index[0]
        except IndexError:
            continue

        num_birds += 1
        img = torch.Tensor(sample['image']).unsqueeze(0).to(device)

        outputs:ExtendedCBMOutput = model(img)

        pred = outputs.concept_module.concept_probs.cpu() >= 0.5

        if args.test_for_compliment:
            base_class_idx = test_dataset.get_bird_idx_by_name(bird_name=bird_name)
            if not base_class_idx:
                raise ValueError(f"No class id found for bird_name='{bird_name}'")
            s_minus_idx = test_dataset.get_removed_attribute_index(base_class_idx=base_class_idx, used_attr_name=used_attr_name, used_attribute_names=used_attributes.to_list())
            if s_minus_idx is None:
                num_birds -= 1
                continue
            if pred[0][s_minus_idx]:
                total_incorrect += 1
        else:
            if not pred[0][s_minus_idx]:
                total_incorrect += 1

        pbar.set_postfix({'s_minus_idx': s_minus_idx, 'total_incorrect':total_incorrect, 'num_birds':num_birds, 'score': f"{(total_incorrect/num_birds):.2f}"})

    with open(args.output_path, mode='w') as output:
        correct = num_birds - total_incorrect
        print('total correct: ' + str(correct / num_birds), file=output)
        print('total incorrect: ' + str(total_incorrect / num_birds), file=output)
        print(f'images evaluated: {num_birds}', file=output)
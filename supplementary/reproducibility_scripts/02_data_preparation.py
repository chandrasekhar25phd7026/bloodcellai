"""
BloodCellAI reproducibility scripts -- 02: Per-seed data preparation.
Builds class-balanced subsamples for the three larger classification
datasets (NIH_Malaria, AcuteLeukemia, Raabin_WBC) and resizes the
Malaria_BBoxes source images. Run once per seed before training that seed.
Run 01_setup.py first.
"""
import os, shutil, random

def create_capped_subset(src_dir, dst_dir, max_per_class, seed=42):
    if os.path.exists(dst_dir):
        return dst_dir
    os.makedirs(dst_dir, exist_ok=True)
    rng = random.Random(seed)
    for cls in os.listdir(src_dir):
        cls_src = os.path.join(src_dir, cls)
        if not os.path.isdir(cls_src):
            continue
        files = [f for f in os.listdir(cls_src) if os.path.isfile(os.path.join(cls_src, f))]
        rng.shuffle(files)
        cls_dst = os.path.join(dst_dir, cls)
        os.makedirs(cls_dst, exist_ok=True)
        for f in files[:max_per_class]:
            shutil.copy2(os.path.join(cls_src, f), os.path.join(cls_dst, f))
    return dst_dir


def prepare_all(seed):
    if not os.path.exists("/kaggle/working/Chula-RBC-12-Dataset"):
        os.system("git clone https://github.com/Chula-PIC-Lab/Chula-RBC-12-Dataset.git "
                  "/kaggle/working/Chula-RBC-12-Dataset")

    malaria_resized = "/kaggle/working/malaria_resized"
    if not os.path.exists(f"{malaria_resized}/images"):
        import cv2
        os.makedirs(f"{malaria_resized}/images", exist_ok=True)
        src_dir = "/kaggle/input/datasets/kmader/malaria-bounding-boxes/malaria/images"
        for fname in os.listdir(src_dir):
            img = cv2.imread(os.path.join(src_dir, fname))
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = 800 / max(h, w)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            cv2.imwrite(f"{malaria_resized}/images/{fname}", img)
        shutil.copy2("/kaggle/input/datasets/kmader/malaria-bounding-boxes/malaria/training.json", malaria_resized)
        shutil.copy2("/kaggle/input/datasets/kmader/malaria-bounding-boxes/malaria/test.json", malaria_resized)

    nih_clean = "/kaggle/working/nih_malaria_clean"
    if not os.path.exists(nih_clean):
        os.makedirs(nih_clean, exist_ok=True)
        src = "/kaggle/input/datasets/iarunava/cell-images-for-detecting-malaria/cell_images"
        for cls in ["Parasitized", "Uninfected"]:
            shutil.copytree(os.path.join(src, cls), os.path.join(nih_clean, cls), dirs_exist_ok=True)
    nih_capped = create_capped_subset(nih_clean, f"/kaggle/working/nih_malaria_capped_{seed}",
                                       max_per_class=1000, seed=seed)

    leukemia_capped = create_capped_subset(
        "/kaggle/input/datasets/mehradaria/leukemia/Original",
        f"/kaggle/working/leukemia_capped_{seed}", max_per_class=600, seed=seed
    )

    raabin_capped = create_capped_subset(
        "/kaggle/input/datasets/paultimothymooney/blood-cells/dataset2-master/dataset2-master/images/TRAIN",
        f"/kaggle/working/raabin_capped_{seed}", max_per_class=1000, seed=seed
    )

    return malaria_resized, nih_capped, leukemia_capped, raabin_capped


if __name__ == "__main__":
    SEED = 42  # change per session: 42, 123, 456, 789, 999
    malaria_resized, nih_capped, leukemia_capped, raabin_capped = prepare_all(SEED)
    print("Malaria resized:", malaria_resized)
    print("NIH capped:", nih_capped)
    print("Leukemia capped:", leukemia_capped)
    print("Raabin capped:", raabin_capped)

import os

# Root data paths
DATA_ROOT = r"C:\Users\RAOVI\PSGN_Project\data"
RENDER_PATH = os.path.join(DATA_ROOT, "renders", "rendering")
SHAPENET_PATH = os.path.join(DATA_ROOT, "ShapeNetCore.v2")

# Categories we are using
CLASSES = {
    "airplane": "02691156",
    "car": "02958343",
    "chair": "03001627"
}

def get_paired_data():
    pairs = []
    for cname, cid in CLASSES.items():
        render_class_path = os.path.join(RENDER_PATH, cid)
        model_class_path = os.path.join(SHAPENET_PATH, cid)
        if not os.path.exists(render_class_path):
            print(f"⚠️ Skipping {cname} — render path not found: {render_class_path}")
            continue
        for model_dir in os.listdir(render_class_path):
            render_dir = os.path.join(render_class_path, model_dir)
            model_obj = os.path.join(model_class_path, model_dir.split("_")[-1], "models", "model_normalized.obj")
            if os.path.exists(model_obj):
                img_files = [os.path.join(render_dir, f) for f in os.listdir(render_dir) if f.endswith(".png")]
                for img in img_files:
                    pairs.append((img, model_obj))
    return pairs

if __name__ == "__main__":
    data_pairs = get_paired_data()
    print(f"✅ Total paired samples found: {len(data_pairs)}")
    if len(data_pairs) > 0:
        print("Example:")
        print(data_pairs[0])
    else:
        print("❌ No valid pairs found. Check folder names or paths.")

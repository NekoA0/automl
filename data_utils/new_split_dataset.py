#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO splitter (8:1:1) with per-class strict balancing + minority-class protection.

核心優化：
1. 先統計每個類別的總目標數，按8:1:1計算每個類別在train/val/test的目標配額
2. 按類別配額優先分配，確保每個類別的目標分布嚴格貼近比例
3. 保留稀有類別保護、負樣本均衡、全域圖像/目標數校準邏輯
4. 自動生成YOLO訓練用的YAML配置檔
"""

import argparse
import csv
import random
from pathlib import Path
from collections import defaultdict, Counter
import shutil
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def read_label_file(label_path: Path):
    """讀取單張圖片的標籤檔案，回傳目標數、類別計數、是否有標籤、包含的類別"""
    cls_counter = Counter()
    classes_in_image = set()
    if not label_path.exists():
        return 0, cls_counter, False, classes_in_image
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return 0, cls_counter, True, classes_in_image
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        try:
            cid = int(float(parts[0]))
        except Exception:
            continue
        cls_counter[cid] += 1
        classes_in_image.add(cid)
    return sum(cls_counter.values()), cls_counter, True, classes_in_image

def scan_dataset(images_dir: Path, labels_dir: Path):
    """掃描資料集，回傳圖片列表、全域統計資訊"""
    items = []
    per_class_image_idxs = defaultdict(set)  # 每個類別對應的圖片索引
    per_class_obj_list = defaultdict(list)   # 每個類別對應的目標（圖片索引+目標數）
    global_cls_counter = Counter()
    negatives = 0
    missing = 0

    for img_path in sorted(images_dir.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        obj_cnt, cls_counter, has_label, classes_in_image = read_label_file(label_path)
        if not has_label:
            missing += 1
            continue
        if obj_cnt == 0:
            negatives += 1

        idx = len(items)
        items.append({
            "idx": idx,
            "img": img_path,
            "lbl": label_path,
            "obj_cnt": obj_cnt,
            "cls_counter": cls_counter,
            "classes_in_image": classes_in_image,
            "assigned": False  # 新增：標記是否已分配
        })
        global_cls_counter.update(cls_counter)
        for cid in classes_in_image:
            per_class_image_idxs[cid].add(idx)
            # 記錄每個類別在該圖片中的目標數
            per_class_obj_list[cid].append((idx, cls_counter[cid]))

    total_objects = sum(global_cls_counter.values())
    return (items, total_objects, global_cls_counter, 
            per_class_image_idxs, per_class_obj_list, 
            missing, negatives)

def write_summary_csv(out_csv: Path, items, total_objects, global_cls_counter, 
                      per_class_image_idxs, negatives, missing):
    """寫入資料集彙整CSV"""
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["num_images", len(items)])
        w.writerow(["num_objects", total_objects])
        w.writerow(["num_negatives(empty-label)", negatives])
        w.writerow(["num_missing_labels(excluded)", missing])
        w.writerow([])
        w.writerow(["class_id", "object_count", "image_count"])
        for cid in sorted(global_cls_counter.keys()):
            w.writerow([cid, global_cls_counter[cid], len(per_class_image_idxs.get(cid, set()))])

def calculate_per_class_quota(global_cls_counter, ratios=(0.8, 0.1, 0.1)):
    """計算每個類別在train/val/test中的目標配額"""
    per_class_quota = defaultdict(lambda: [0, 0, 0])  # [train, val, test]
    for cid, total_cnt in global_cls_counter.items():
        # 按比例分配目標數，確保總和等於原總數
        train_cnt = round(total_cnt * ratios[0])
        val_cnt = round(total_cnt * ratios[1])
        test_cnt = total_cnt - train_cnt - val_cnt
        
        # 校準：避免val/test為0（稀有類別保護）
        if total_cnt >= 3:
            val_cnt = max(1, val_cnt)
            test_cnt = max(1, test_cnt)
            # 重新分配差值
            diff = total_cnt - (train_cnt + val_cnt + test_cnt)
            if diff != 0:
                train_cnt += diff
        else:
            # 少於3個目標的類別全部放入train
            train_cnt = total_cnt
            val_cnt = 0
            test_cnt = 0
        
        per_class_quota[cid] = [train_cnt, val_cnt, test_cnt]
    return per_class_quota

def assign_per_class_objects(items, per_class_obj_list, per_class_quota, 
                             rare_classes, mid_classes):
    """按類別配額分配目標到各資料集"""
    # 初始化桶
    buckets = [
        {"name":"train","items":[], "n_imgs":0,"n_objs":0,"cls_counter":Counter()},
        {"name":"val",  "items":[], "n_imgs":0,"n_objs":0,"cls_counter":Counter()},
        {"name":"test", "items":[], "n_imgs":0,"n_objs":0,"cls_counter":Counter()},
    ]

    def add_to_bucket(b_idx, item):
        """將圖片添加到指定桶，並更新統計"""
        if item["assigned"]:
            return False
        buckets[b_idx]["items"].append(item)
        buckets[b_idx]["n_imgs"] += 1
        buckets[b_idx]["n_objs"] += item["obj_cnt"]
        buckets[b_idx]["cls_counter"].update(item["cls_counter"])
        item["assigned"] = True
        return True

    # 1. 優先分配稀有類別（全部到train）
    for cid in rare_classes:
        for (img_idx, obj_cnt) in per_class_obj_list[cid]:
            item = items[img_idx]
            add_to_bucket(0, item)

    # 2. 按配額分配中等/普通類別
    for cid in list(per_class_obj_list.keys()):
        if cid in rare_classes:
            continue
        
        # 取得該類別的配額和待分配目標
        train_quota, val_quota, test_quota = per_class_quota[cid]
        obj_list = per_class_obj_list[cid].copy()
        random.shuffle(obj_list)  # 隨機打亂，避免順序偏差
        
        train_assigned = 0
        val_assigned = 0
        test_assigned = 0

        for (img_idx, obj_cnt) in obj_list:
            item = items[img_idx]
            if item["assigned"]:
                continue
            
            # 按配額分配
            if train_assigned < train_quota:
                if add_to_bucket(0, item):
                    train_assigned += obj_cnt
            elif val_assigned < val_quota:
                if add_to_bucket(1, item):
                    val_assigned += obj_cnt
            elif test_assigned < test_quota:
                if add_to_bucket(2, item):
                    test_assigned += obj_cnt

    # 3. 分配負樣本（無目標圖片），按8:1:1均分
    negative_items = [item for item in items if item["obj_cnt"] == 0 and not item["assigned"]]
    random.shuffle(negative_items)
    n_neg = len(negative_items)
    neg_train = round(n_neg * 0.8)
    neg_val = round(n_neg * 0.1)
    neg_test = n_neg - neg_train - neg_val

    for i, item in enumerate(negative_items):
        if i < neg_train:
            add_to_bucket(0, item)
        elif i < neg_train + neg_val:
            add_to_bucket(1, item)
        else:
            add_to_bucket(2, item)

    # 4. 分配剩餘未分配的圖片（補充到配額不足的桶）
    unassigned = [item for item in items if not item["assigned"]]
    random.shuffle(unassigned)
    
    # 計算各桶的圖片缺口（按8:1:1全域比例）
    total_imgs = len(items)
    tgt_train = round(total_imgs * 0.8)
    tgt_val = round(total_imgs * 0.1)
    tgt_test = total_imgs - tgt_train - tgt_val

    for item in unassigned:
        # 優先補充缺口最大的桶
        train_deficit = tgt_train - buckets[0]["n_imgs"]
        val_deficit = tgt_val - buckets[1]["n_imgs"]
        test_deficit = tgt_test - buckets[2]["n_imgs"]

        if train_deficit > 0:
            add_to_bucket(0, item)
        elif val_deficit > 0:
            add_to_bucket(1, item)
        else:
            add_to_bucket(2, item)

    return buckets

def describe_bucket(b):
    """列印桶的統計資訊"""
    cls_cnt = b["cls_counter"]
    denom = sum(cls_cnt.values())
    parts = []
    for k, v in cls_cnt.most_common(8):
        p = (v / denom * 100.0) if denom > 0 else 0.0
        parts.append(f"{k}:{v}({p:.1f}%)")
    return (f"{b['name']}: 圖片數={b['n_imgs']}, 目標數={b['n_objs']}, "
            f"分布=[{', '.join(parts)}]")

def save_split_lists(buckets, out_dir: Path):
    """保存各資料集的圖片路徑列表"""
    out_dir.mkdir(parents=True, exist_ok=True)
    # 直接保存在 out_dir，不建立 splits 子目錄
    for b in buckets:
        with (out_dir / f"{b['name']}.txt").open("w", encoding="utf-8") as f:
            for it in b["items"]:
                f.write(str(it["img"].resolve()) + "\n")
    return out_dir

def optional_copy_files(buckets, out_dir: Path):
    """可選：複製圖片和標籤到對應目錄"""
    for b in buckets:
        img_dst = out_dir / b["name"] / "images"
        lbl_dst = out_dir / b["name"] / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for it in b["items"]:
            shutil.copy2(it["img"], img_dst / it["img"].name)
            shutil.copy2(it["lbl"], lbl_dst / it["lbl"].name)

def generate_yolo_yaml(out_dir: Path, global_cls_counter, class_names=None):
    """生成YOLO訓練用的YAML配置檔
    Args:
        out_dir: 輸出目錄
        global_cls_counter: 全域類別計數器
        class_names: 類別名稱列表（若為None則使用 class_0, class_1...）
    """
    # 若未提供類別名稱，自動生成
    if class_names is None:
        class_names = [f"class_{cid}" for cid in sorted(global_cls_counter.keys())]
    
    # 取得絕對路徑
    base_path = out_dir.resolve()

    # 根據使用者提供的舊有寫法：使用絕對路徑並統一為 forward slashes
    path_val = str(base_path).replace('\\', '/')

    yaml_data = {
        "path":  path_val,      # 資料集根目錄
        "train": "train.txt",   # 訓練集圖片列表
        "val":   "val.txt",     # 驗證集圖片列表
        "test":  "test.txt",    # 測試集圖片列表
        "nc":    len(global_cls_counter), # 類別數量
        "names": class_names              # 類別名稱
    }
    
    # 寫入YAML檔 (改名為 data.yaml)
    yaml_path = out_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"\n[YAML] YOLO配置檔已生成: {yaml_path}")

    # 嘗試移動 data.yaml 到 Dataset/user/task
    try:
        abs_out = out_dir.resolve()
        parts = abs_out.parts
        if 'dataset_home' in parts:
            # 找到最後一個 dataset_home 的索引
            idx = len(parts) - 1 - parts[::-1].index('dataset_home')
            # 構建 Dataset 路徑
            base_root = Path(*parts[:idx])
            rel_path = Path(*parts[idx+1:])
            
            target_dir = base_root / "Dataset" / rel_path
            target_dir.mkdir(parents=True, exist_ok=True)
            
            target_yaml = target_dir / "data.yaml"
            shutil.move(str(yaml_path), str(target_yaml))
            print(f"[MOVE] data.yaml moved to: {target_yaml}")
            return target_yaml
    except Exception as e:
        print(f"[WARNING] Could not move data.yaml: {e}")

    return yaml_path

def main():
    ap = argparse.ArgumentParser(description="YOLO 8:1:1 分割工具（含逐類別嚴格均衡 + 少數類別保護）")
    ap.add_argument("--images", type=Path, required=True, help="圖片目錄")
    ap.add_argument("--labels", type=Path, required=True, help="標籤目錄")
    ap.add_argument("--out", type=Path, required=True, help="輸出目錄")
    ap.add_argument("--seed", type=int, default=2025, help="隨機種子")
    ap.add_argument("--rare-obj-thresh", type=int, default=30, help="稀有類別目標數閾值（全部放入train）")
    ap.add_argument("--copy-files", type=str, default="false", help="是否複製檔案（true/false）")
    ap.add_argument("--summary-only", action="store_true", help="僅生成彙整，不分割")
    ap.add_argument("--class-names", type=str, default=None, help="類別名稱列表（逗號分隔，例如：car,person,bike）")
    ap.add_argument("--class-txt", type=Path, default=None, help="類別名稱檔案路徑（一行一個類別）")
    args = ap.parse_args()

    # 初始化
    random.seed(args.seed)
    copy_files = args.copy_files.lower() in {"1","true","yes","y"}
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # 處理類別名稱
    class_names = None
    if args.class_txt and args.class_txt.exists():
        content = args.class_txt.read_text(encoding="utf-8")
        class_names = [line.strip() for line in content.splitlines() if line.strip()]
    elif args.class_names:
        class_names = [name.strip() for name in args.class_names.split(",")]

    # 1. 掃描資料集
    (items, total_objects, global_cls_counter, per_class_image_idxs, 
     per_class_obj_list, missing, negatives) = scan_dataset(args.images, args.labels)

    # 驗證類別名稱數量
    if class_names and len(class_names) != len(global_cls_counter):
        raise ValueError(f"提供的類別名稱數量({len(class_names)})與資料集中的類別數量({len(global_cls_counter)})不符")

    # 2. 生成彙整
    csv_path = out_dir / "dataset_summary.csv"
    write_summary_csv(csv_path, items, total_objects, global_cls_counter, 
                      per_class_image_idxs, negatives, missing)
    print(f"[SUMMARY] 圖片數={len(items)}, 目標總數={total_objects}, 負樣本數={negatives}, 缺失標籤數={missing}")
    print(f"[SUMMARY] 彙整檔案: {csv_path}")

    if args.summary_only:
        # 即使僅生成彙整，也可選擇生成YAML
        generate_yolo_yaml(out_dir, global_cls_counter, class_names)
        return

    # 3. 定義稀有/中等/普通類別
    rare_classes = set([cid for cid, cnt in global_cls_counter.items() if cnt <= args.rare_obj_thresh])
    mid_classes = set([cid for cid, cnt in global_cls_counter.items() 
                       if args.rare_obj_thresh < cnt <= args.rare_obj_thresh * 3])
    common_classes = set(global_cls_counter.keys()) - rare_classes - mid_classes

    # 4. 計算每個類別的目標配額
    per_class_quota = calculate_per_class_quota(global_cls_counter, ratios=(0.8, 0.1, 0.1))
    print(f"\n[QUOTA] 類別配額（train/val/test）:")
    for cid in sorted(global_cls_counter.keys()):
        quota = per_class_quota[cid]
        print(f"  類別{cid}: 總目標={global_cls_counter[cid]} → 配額={quota}")

    # 5. 按配額分配圖片
    buckets = assign_per_class_objects(items, per_class_obj_list, per_class_quota, 
                                       rare_classes, mid_classes)

    # 6. 保存分割列表
    splits_dir = save_split_lists(buckets, out_dir)
    print(f"\n[SPLIT] 分割列表保存到: {splits_dir}")
    for b in buckets:
        print(f"  - {describe_bucket(b)}")

    # 7. 可選：複製檔案
    if copy_files:
        print(f"\n[COPY] 正在複製檔案到 {out_dir}...")
        optional_copy_files(buckets, out_dir)
        print("[COPY] 檔案複製完成")

    # 8. 驗證每個類別的實際分布
    print(f"\n[VERIFY] 每個類別的實際分布:")
    for cid in sorted(global_cls_counter.keys()):
        train_cnt = buckets[0]["cls_counter"].get(cid, 0)
        val_cnt = buckets[1]["cls_counter"].get(cid, 0)
        test_cnt = buckets[2]["cls_counter"].get(cid, 0)
        total = train_cnt + val_cnt + test_cnt
        print(f"  類別{cid}: train={train_cnt}({train_cnt/total*100:.1f}%), val={val_cnt}({val_cnt/total*100:.1f}%), test={test_cnt}({test_cnt/total*100:.1f}%)")

    # 9. 生成YOLO YAML配置檔
    generate_yolo_yaml(out_dir, global_cls_counter, class_names)

if __name__ == "__main__":
    main()
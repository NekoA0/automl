#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO to xanylabeling JSON Format Converter

This script converts YOLO format annotations (.txt) to xanylabeling JSON format.
"""

import os
import argparse
import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image


def load_yaml_classes(yaml_path: str) -> List[str]:
    """
    從 YAML 檔案讀取類別名稱列表
    
    Args:
        yaml_path: YAML 檔案路徑
        
    Returns:
        類別名稱列表
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    return data['names']


def yolo_to_xanylabeling_bbox(yolo_bbox: List[float], img_width: int, img_height: int) -> List[List[float]]:
    """
    將 YOLO 格式的邊界框轉換為 xanylabeling 格式
    
    YOLO 格式: [x_center, y_center, width, height] (正規化到 0-1)
    xanylabeling 格式: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]] (絕對像素座標)
    
    Args:
        yolo_bbox: YOLO 格式的邊界框 [x_center, y_center, width, height]
        img_width: 圖片寬度
        img_height: 圖片高度
        
    Returns:
        xanylabeling 格式的四個角點座標
    """
    x_center, y_center, width, height = yolo_bbox
    
    # 將正規化座標轉換為絕對座標
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height
    
    # 計算左上角和右下角座標
    x1 = x_center_abs - width_abs / 2
    y1 = y_center_abs - height_abs / 2
    x2 = x_center_abs + width_abs / 2
    y2 = y_center_abs + height_abs / 2
    
    # 返回四個角點 (左上、右上、右下、左下)
    return [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ]


def convert_yolo_to_xanylabeling(
    txt_path: str,
    image_path: str,
    class_names: List[str],
) -> Dict:
    """
    將單個 YOLO 標註檔案轉換為 xanylabeling JSON 格式
    
    Args:
        txt_path: YOLO 標註檔案路徑 (.txt)
        image_path: 對應的圖片檔案路徑
        class_names: 類別名稱列表
        
    Returns:
        xanylabeling JSON 格式的字典
    """
    # 讀取圖片獲取尺寸
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        raise ValueError(f"無法讀取圖片檔案 {image_path}: {str(e)}")
    
    # 取得圖片檔案名稱
    image_filename = os.path.basename(image_path)
    
    shapes = []
    
    # 讀取 YOLO 標註檔案
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 解析 YOLO 格式: class_id x_center y_center width height
        parts = line.split()
        class_id = int(parts[0])
        yolo_bbox = [float(x) for x in parts[1:5]]
        
        # 轉換邊界框座標
        points = yolo_to_xanylabeling_bbox(yolo_bbox, img_width, img_height)
        
        # 建立 shape 物件
        shape = {
            "label": class_names[class_id],
            "shape_type": "rectangle",
            "flags": {},
            "points": points,
            "group_id": None,
            "description": None,
            "difficult": False,
            "attributes": {}
        }
        shapes.append(shape)
    
    # 建立 xanylabeling JSON 格式
    xanylabeling_json = {
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width
    }
    
    return xanylabeling_json


def batch_convert(
    images_dir: str,
    labels_dir: str,
    yaml_path: str,
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
):
    """
    批次轉換 YOLO 標註為 xanylabeling JSON 格式
    
    Args:
        images_dir: 圖片資料夾路徑
        labels_dir: YOLO 標註檔案資料夾路徑
        yaml_path: YAML 設定檔路徑
        image_extensions: 支援的圖片副檔名
    """
    # 載入類別名稱
    class_names = load_yaml_classes(yaml_path)
    print(f"已載入 {len(class_names)} 個類別: {class_names}")
    
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    # 確認資料夾存在
    if not images_path.exists():
        raise FileNotFoundError(f"圖片資料夾不存在: {images_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(f"標註資料夾不存在: {labels_dir}")
    
    # 取得所有圖片檔案
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_path.glob(f'*{ext}')))
    
    converted_count = 0
    skipped_count = 0
    
    for image_file in image_files:
        # 找對應的 .txt 檔案
        txt_file = labels_path / f"{image_file.stem}.txt"
        
        if not txt_file.exists():
            print(f"警告: 找不到標註檔案 {txt_file.name}，跳過")
            skipped_count += 1
            continue
        
        # 轉換為 xanylabeling JSON
        try:
            xanylabeling_json = convert_yolo_to_xanylabeling(
                txt_path=str(txt_file),
                image_path=str(image_file),
                class_names=class_names
            )
            
            # 儲存 JSON 檔案到 images 資料夾
            json_file = images_path / f"{image_file.stem}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(xanylabeling_json, f, indent=2, ensure_ascii=False)
            
            converted_count += 1
            # print(f"✓ 已轉換: {image_file.name} -> {json_file.name}")
            
        except Exception as e:
            print(f"✗ 轉換失敗 {image_file.name}: {str(e)}")
            skipped_count += 1
    
    print(f"轉換完成:")
    print(f"成功: {converted_count} 個檔案")
    print(f"跳過: {skipped_count} 個檔案")


def main():
    """
    主函式 - 解析命令列參數並執行轉換
    """
    parser = argparse.ArgumentParser(description='將 YOLO 格式標註轉換為 xanylabeling JSON 格式')
    
    parser.add_argument('--images', type=str, required=True, help='圖片資料夾路徑')
    parser.add_argument('--labels', type=str, required=True, help='YOLO 標註檔案資料夾路徑')
    parser.add_argument('--yaml', type=str, required=True, help='包含類別名稱的 YAML 設定檔路徑')
    
    args = parser.parse_args()
    
    # 執行批次轉換
    batch_convert(
        images_dir=args.images,
        labels_dir=args.labels,
        yaml_path=args.yaml
    )


if __name__ == "__main__":
    main()
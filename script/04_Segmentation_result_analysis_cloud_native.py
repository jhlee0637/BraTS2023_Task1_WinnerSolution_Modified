#!/usr/bin/env python3
"""
클라우드 네이티브 Segmentation 결과 분석 스크립트 (Dice Score, Hausdorff Distance 등)
Usage: python 04_Segmentation_result_analysis_cloud_native.py <s3_bucket> <job_id>
"""
import os
import sys
import numpy as np
import nibabel as nib
from scipy.spatial.distance import cdist
import pandas as pd
import gc
import boto3
import yaml
from datetime import datetime

def download_ground_truth_from_s3(s3_bucket, job_id, local_gt_dir):
    """S3에서 Ground Truth 데이터 다운로드"""
    import subprocess
    
    os.makedirs(local_gt_dir, exist_ok=True)
    
    s3_data_path = f"s3://{s3_bucket}/userdata/{job_id}/uploads/data/"
    
    print(f"S3에서 Ground Truth 다운로드: {s3_data_path} -> {local_gt_dir}")
    
    cmd = ["aws", "s3", "sync", s3_data_path, local_gt_dir, "--quiet"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Ground Truth 다운로드 성공")
        return True
    else:
        print(f"❌ Ground Truth 다운로드 실패: {result.stderr}")
        return False

def find_ground_truth_file(case_id, gt_folder):
    """케이스 ID에 해당하는 Ground Truth 파일 찾기"""
    # BraTS-GLI-00066-000.nii.gz -> BraTS-GLI-00066-000 추출
    case_name = case_id.replace('.nii.gz', '')
    
    # Ground Truth 파일 경로들 시도
    possible_paths = [
        os.path.join(gt_folder, case_name, f"{case_name}-seg.nii.gz"),
        os.path.join(gt_folder, f"{case_name}-seg.nii.gz"),
        os.path.join(gt_folder, case_name, "seg.nii.gz")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None
    """Dice Score 계산 (메모리 효율적)"""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    
    del pred_mask, gt_mask
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2.0 * intersection / union

def dice_score(pred, gt, label):
    """Dice Score 계산"""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2.0 * intersection / union

def hausdorff_distance(pred, gt, label, spacing=(1.0, 1.0, 1.0)):
    """메모리 효율적인 Hausdorff Distance 계산 (g5.2xlarge 최적화)"""
    try:
        pred_mask = (pred == label).astype(np.uint8)
        gt_mask = (gt == label).astype(np.uint8)
        
        # 빈 마스크 처리
        if not np.any(pred_mask) and not np.any(gt_mask):
            return 0.0
        if not np.any(pred_mask) or not np.any(gt_mask):
            return 373.13  # 최대값 반환
        
        # 경계 픽셀만 추출 (메모리 절약)
        from scipy import ndimage
        pred_boundary = pred_mask - ndimage.binary_erosion(pred_mask)
        gt_boundary = gt_mask - ndimage.binary_erosion(gt_mask)
        
        pred_coords = np.column_stack(np.where(pred_boundary))
        gt_coords = np.column_stack(np.where(gt_boundary))
        
        # 좌표가 너무 많으면 샘플링 (메모리 제한)
        max_points = 3000  # g5.2xlarge에 맞춤
        if len(pred_coords) > max_points:
            idx = np.random.choice(len(pred_coords), max_points, replace=False)
            pred_coords = pred_coords[idx]
        if len(gt_coords) > max_points:
            idx = np.random.choice(len(gt_coords), max_points, replace=False)
            gt_coords = gt_coords[idx]
        
        # 물리적 좌표로 변환
        pred_coords = pred_coords * np.array(spacing)
        gt_coords = gt_coords * np.array(spacing)
        
        # 청크 단위로 거리 계산 (메모리 절약)
        chunk_size = 500  # 더 작은 청크
        max_dist_pred_to_gt = 0
        max_dist_gt_to_pred = 0
        
        # pred에서 gt까지의 최대 거리
        for i in range(0, len(pred_coords), chunk_size):
            chunk = pred_coords[i:i+chunk_size]
            distances = np.sqrt(np.sum((chunk[:, np.newaxis] - gt_coords[np.newaxis, :]) ** 2, axis=2))
            min_distances = np.min(distances, axis=1)
            max_dist_pred_to_gt = max(max_dist_pred_to_gt, np.max(min_distances))
            del distances, min_distances  # 메모리 해제
        
        # gt에서 pred까지의 최대 거리
        for i in range(0, len(gt_coords), chunk_size):
            chunk = gt_coords[i:i+chunk_size]
            distances = np.sqrt(np.sum((chunk[:, np.newaxis] - pred_coords[np.newaxis, :]) ** 2, axis=2))
            min_distances = np.min(distances, axis=1)
            max_dist_gt_to_pred = max(max_dist_gt_to_pred, np.max(min_distances))
            del distances, min_distances  # 메모리 해제
        
        return max(max_dist_pred_to_gt, max_dist_gt_to_pred)
        
    except Exception as e:
        print(f"Hausdorff distance calculation failed: {e}")
        return 999.0  # 오류 시 기본값

def load_config_from_s3(s3_bucket, s3_key):
    """S3에서 config 파일 로드"""
    s3_client = boto3.client('s3')
    try:
        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        config_content = response['Body'].read().decode('utf-8')
        return yaml.safe_load(config_content)
    except Exception as e:
        print(f"Config 로드 실패: {e}")
        return {}

def analyze_segmentation_results(prediction_folder, ground_truth_folder, output_file):
    """Segmentation 결과 분석"""
    results = []
    labels = [1, 2, 3]  # BraTS labels: ET, TC, WT
    label_names = ['ET', 'TC', 'WT']
    
    pred_files = [f for f in os.listdir(prediction_folder) if f.endswith('.nii.gz')]
    
    for pred_file in pred_files:
        print(f"Processing: {pred_file}")
        
        # 파일 경로
        pred_path = os.path.join(prediction_folder, pred_file)
        
        # Ground truth 파일 찾기 (개선된 로직)
        case_id = pred_file.replace('.nii.gz', '')
        gt_path = find_ground_truth_file(pred_file, ground_truth_folder)
        
        if gt_path is None:
            print(f"Ground truth not found for {case_id}")
            continue
        
        print(f"Processing: {pred_file} vs {os.path.basename(gt_path)}")
        
        try:
            # 이미지 로드
            pred_img = nib.load(pred_path)
            gt_img = nib.load(gt_path)
            
            pred_data = pred_img.get_fdata().astype(np.uint8)
            gt_data = gt_img.get_fdata().astype(np.uint8)
            
            # 각 라벨별 메트릭 계산
            case_results = {'case_id': case_id}
            
            for label, label_name in zip(labels, label_names):
                dice = dice_score(pred_data, gt_data, label)
                hd = hausdorff_distance(pred_data, gt_data, label, spacing=pred_img.header.get_zooms()[:3])
                
                case_results[f'dice_{label_name}'] = dice
                case_results[f'hd_{label_name}'] = hd if hd != float('inf') else np.nan
            
            results.append(case_results)
            
            # 메모리 정리
            del pred_data, gt_data
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
            continue
    
    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(results)
    
    # 통계 계산
    summary_stats = {}
    for label_name in label_names:
        dice_col = f'dice_{label_name}'
        hd_col = f'hd_{label_name}'
        
        if dice_col in df.columns:
            summary_stats[f'{dice_col}_mean'] = df[dice_col].mean()
            summary_stats[f'{dice_col}_std'] = df[dice_col].std()
        
        if hd_col in df.columns:
            summary_stats[f'{hd_col}_mean'] = df[hd_col].mean()
            summary_stats[f'{hd_col}_std'] = df[hd_col].std()
    
    # 결과 저장
    df.to_csv(output_file, index=False)
    
    # 요약 통계 출력
    print("\n=== Analysis Summary ===")
    for key, value in summary_stats.items():
        print(f"{key}: {value:.4f}")
    
    return df, summary_stats

def main():
    if len(sys.argv) != 3:
        print("Usage: python 04_Segmentation_result_analysis_cloud_native.py <s3_bucket> <job_id>")
        sys.exit(1)
    
    s3_bucket = sys.argv[1]
    job_id = sys.argv[2]
    
    # 경로 설정
    running_dir = f"/home/ec2-user/project/final/result/{job_id}"
    log_dir = os.path.join(running_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"=== Segmentation Result Analysis Started ===")
    print(f"Job ID: {job_id}")
    print(f"S3 Bucket: {s3_bucket}")
    
    try:
        # 로그 설정
        running_dir = f"/home/ec2-user/project/final/result/{job_id}"
        log_dir = f'{running_dir}/log'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, '04_Segmentation_result_analysis_cloud_native.log')
        
        # 로그 파일과 콘솔 동시 출력을 위한 클래스
        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
        
        # 실시간 S3 업로드 함수
        def periodic_s3_uploader(local_path, s3_bucket, s3_key, interval, stop_event):
            import threading
            import time
            s3_client = boto3.client('s3')
            while not stop_event.wait(interval):
                try:
                    if os.path.exists(local_path):
                        s3_client.upload_file(local_path, s3_bucket, s3_key)
                except Exception as e:
                    print(f"[S3 Upload Error] {e}")
            try:
                if os.path.exists(local_path):
                    s3_client.upload_file(local_path, s3_bucket, s3_key)
            except:
                pass
        
        # S3 실시간 업로드 시작
        import threading
        stop_event = threading.Event()
        s3_log_key = f"userdata/{job_id}/result/log/04_Segmentation_result_analysis_cloud_native.log"
        uploader_thread = threading.Thread(
            target=periodic_s3_uploader,
            args=(log_file, s3_bucket, s3_log_key, 3, stop_event)
        )
        uploader_thread.daemon = True
        uploader_thread.start()
        
        # 표준 출력을 로그 파일과 콘솔에 동시 출력
        log_f = open(log_file, 'w', encoding='utf-8')
        sys.stdout = Tee(sys.stdout, log_f)
        
        # 1. Config 로드
        config_s3_key = f"userdata/{job_id}/uploads/config.yaml"
        config = load_config_from_s3(s3_bucket, config_s3_key)
        
        # 2. 경로 설정
        print("\n--- Step 2: 경로 설정 ---")
        running_dir = f"/home/ec2-user/project/final/result/{job_id}"
        prediction_folder = f'{running_dir}/BraTS2023_inference/output'
        ground_truth_folder = f'/home/ec2-user/s3-data/userdata/{job_id}/uploads/data'
        output_file = f'{running_dir}/segmentation_analysis_results.csv'
        
        # 3. 분석 실행
        print("\n--- Step 3: Analyzing Segmentation Results ---")
        df, summary_stats = analyze_segmentation_results(prediction_folder, ground_truth_folder, output_file)
        
        print(f"\n=== Analysis Completed Successfully ===")
        print(f"Results saved to: {output_file}")
        print(f"Processed {len(df)} cases")
        
        # S3 업로드 종료
        stop_event.set()
        uploader_thread.join()
        
    except Exception as e:
        print(f"Error: {e}")
        # S3 업로드 종료
        if 'stop_event' in locals():
            stop_event.set()
        if 'uploader_thread' in locals():
            uploader_thread.join()
        sys.exit(1)

if __name__ == "__main__":
    main()

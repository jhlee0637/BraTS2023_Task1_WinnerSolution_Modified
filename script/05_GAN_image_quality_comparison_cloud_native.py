#!/usr/bin/env python3
"""
클라우드 네이티브 Segmentation 모델 비교 분석 스크립트
Usage: python 05_Segmentation_model_comparison_cloud_native.py <s3_bucket> <job_id>
"""
import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import gc
import boto3
import yaml
from datetime import datetime

def dice_score(pred, gt, label):
    """Dice Score 계산 (메모리 효율적)"""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    
    del pred_mask, gt_mask
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2.0 * intersection / union

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

def analyze_gan_quality(synthetic_folder, real_folder, output_dir):
    """GAN 생성 데이터 품질 분석"""
    results = []
    
    # 합성 데이터 폴더들 찾기
    synthetic_cases = [d for d in os.listdir(synthetic_folder) if os.path.isdir(os.path.join(synthetic_folder, d))]
    
    for case_dir in synthetic_cases:
        case_path = os.path.join(synthetic_folder, case_dir)
        
        # 합성 데이터 파일들 찾기
        synthetic_files = {}
        for file in os.listdir(case_path):
            if file.endswith('-t1c.nii.gz'):
                synthetic_files['t1c'] = os.path.join(case_path, file)
            elif file.endswith('-t1n.nii.gz'):
                synthetic_files['t1n'] = os.path.join(case_path, file)
            elif file.endswith('-t2w.nii.gz'):
                synthetic_files['t2w'] = os.path.join(case_path, file)
            elif file.endswith('-t2f.nii.gz'):
                synthetic_files['t2f'] = os.path.join(case_path, file)
            elif file.endswith('-seg.nii.gz'):
                synthetic_files['seg'] = os.path.join(case_path, file)
        
        # 대응하는 실제 데이터 찾기
        base_case_id = case_dir.split('_')[0]  # BraTS-GLI-xxxxx-000 부분 추출
        real_case_path = None
        
        for real_dir in os.listdir(real_folder):
            if base_case_id in real_dir:
                real_case_path = os.path.join(real_folder, real_dir)
                break
        
        if not real_case_path or not os.path.exists(real_case_path):
            continue
        
        # 실제 데이터 파일들 찾기
        real_files = {}
        for file in os.listdir(real_case_path):
            if file.endswith('-t1c.nii.gz'):
                real_files['t1c'] = os.path.join(real_case_path, file)
            elif file.endswith('-t1n.nii.gz'):
                real_files['t1n'] = os.path.join(real_case_path, file)
            elif file.endswith('-t2w.nii.gz'):
                real_files['t2w'] = os.path.join(real_case_path, file)
            elif file.endswith('-t2f.nii.gz'):
                real_files['t2f'] = os.path.join(real_case_path, file)
            elif file.endswith('-seg.nii.gz'):
                real_files['seg'] = os.path.join(real_case_path, file)
        
        # 각 모달리티별 비교
        case_results = {'case_id': case_dir}
        
        for modality in ['t1c', 't1n', 't2w', 't2f']:
            if modality in synthetic_files and modality in real_files:
                try:
                    # 이미지 로드
                    synthetic_img = nib.load(synthetic_files[modality])
                    real_img = nib.load(real_files[modality])
                    
                    synthetic_data = synthetic_img.get_fdata()
                    real_data = real_img.get_fdata()
                    
                    # 통계적 비교
                    mse = np.mean((synthetic_data - real_data) ** 2)
                    psnr = 20 * np.log10(np.max(real_data) / np.sqrt(mse)) if mse > 0 else float('inf')
                    
                    # 구조적 유사성 (간단한 버전)
                    correlation = np.corrcoef(synthetic_data.flatten(), real_data.flatten())[0, 1]
                    
                    case_results[f'{modality}_mse'] = mse
                    case_results[f'{modality}_psnr'] = psnr
                    case_results[f'{modality}_correlation'] = correlation
                    
                    del synthetic_data, real_data
                    
                except Exception as e:
                    print(f"Error processing {case_dir} {modality}: {e}")
                    continue
        
        results.append(case_results)
        gc.collect()
    
    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(results)
    
    # 시각화 생성
    create_comparison_plots(df, output_dir)
    
    # 결과 저장
    output_file = os.path.join(output_dir, 'gan_comparison_results.csv')
    df.to_csv(output_file, index=False)
    
    return df

def create_comparison_plots(df, output_dir):
    """비교 결과 시각화"""
    os.makedirs(output_dir, exist_ok=True)
    
    modalities = ['t1c', 't1n', 't2w', 't2f']
    metrics = ['mse', 'psnr', 'correlation']
    
    # 각 메트릭별 박스플롯
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        metric_data = []
        labels = []
        
        for modality in modalities:
            col_name = f'{modality}_{metric}'
            if col_name in df.columns:
                data = df[col_name].dropna()
                if len(data) > 0:
                    metric_data.append(data)
                    labels.append(modality.upper())
        
        if metric_data:
            plt.boxplot(metric_data, labels=labels)
            plt.title(f'{metric.upper()} Comparison Across Modalities')
            plt.ylabel(metric.upper())
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 상관관계 히트맵
    correlation_cols = [col for col in df.columns if 'correlation' in col]
    if correlation_cols:
        plt.figure(figsize=(10, 8))
        correlation_data = df[correlation_cols].corr()
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Between Modalities')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    if len(sys.argv) != 3:
        print("Usage: python 05_GAN_model_comparison_cloud_native.py <s3_bucket> <job_id>")
        sys.exit(1)
    
    s3_bucket = sys.argv[1]
    job_id = sys.argv[2]
    
    # 경로 설정
    running_dir = f"/home/ec2-user/project/final/result/{job_id}"
    log_dir = os.path.join(running_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"=== GAN Model Comparison Started ===")
    print(f"Job ID: {job_id}")
    print(f"S3 Bucket: {s3_bucket}")
    
    try:
        # 1. Config 로드
        config_s3_key = f"userdata/{job_id}/uploads/config.yaml"
        config = load_config_from_s3(s3_bucket, config_s3_key)
        
        # 2. 경로 설정
        synthetic_folder = f'{running_dir}/GliGAN/Checkpoint/brats2023/Synthetic_dataset_random_labels'
        real_folder = f'/home/ec2-user/s3-data/userdata/{job_id}/uploads/data'
        output_dir = f'{running_dir}/GAN_comparison_results'
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 3. GAN 품질 분석
        print("\n--- Step 1: Analyzing GAN Quality ---")
        df = analyze_gan_quality(synthetic_folder, real_folder, output_dir)
        
        print(f"\n=== GAN Comparison Completed Successfully ===")
        print(f"Results saved to: {output_dir}")
        print(f"Analyzed {len(df)} synthetic cases")
        
        # 요약 통계 출력
        print("\n=== Summary Statistics ===")
        for col in df.columns:
            if col != 'case_id' and df[col].dtype in ['float64', 'int64']:
                print(f"{col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

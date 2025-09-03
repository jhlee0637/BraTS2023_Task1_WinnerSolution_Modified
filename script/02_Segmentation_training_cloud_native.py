#!/usr/bin/env python3
"""
클라우드 네이티브 Enhanced Segmentation 모델 훈련 스크립트 (원본 + 합성 데이터)
Usage: python 02_Segmentation_training_cloud_native_enhanced.py <s3_bucket> <job_id>
"""
import json
import shutil
import os 
import sys
import SimpleITK as sitk
import numpy as np
import boto3
import yaml
from datetime import datetime

def upload_results_to_s3(s3_bucket, job_id, local_results_dir):
    """결과를 S3에 업로드"""
    s3_client = boto3.client('s3')
    
    try:
        import subprocess
        s3_prefix = f"userdata/{job_id}/result_enhanced"
        
        # AWS CLI sync 사용하여 업로드
        cmd = [
            "aws", "s3", "sync", 
            local_results_dir,
            f"s3://{s3_bucket}/{s3_prefix}/",
            "--exclude", "*.tmp",
            "--exclude", "temp_*"
        ]
        
        print(f"S3 업로드 시작: {local_results_dir} -> s3://{s3_bucket}/{s3_prefix}/")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ S3 업로드 성공!")
            return True
        else:
            print(f"❌ S3 업로드 실패: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ S3 업로드 오류: {e}")
        return False

def setup_cloud_paths(job_id):
    """클라우드 환경 경로 설정"""
    running_dir = f"/home/ec2-user/project/final/result/{job_id}"
    
    paths = {
        'source_brats': f'/home/ec2-user/s3-data/userdata/{job_id}/uploads/data',
        'source_gligan': f'{running_dir}/GliGAN/Checkpoint/brats2023/Synthetic_dataset_random_labels',
        'destination_folder': f'{running_dir}/nnUNet/nnUNet_raw/Dataset233_BraTS_2023_Enhanced',
        'nnUNet_preprocessed': f'{running_dir}/nnUNet/nnUNet_preprocessed',
        'nnUNet_results': f'{running_dir}/nnUNet/nnUNet_results',
        'nnUNet_raw': f'{running_dir}/nnUNet/nnUNet_raw',
        'nnUNetv2_plan_and_preprocess': f"/home/ec2-user/project/final/BraTS_2023_2024_solutions/Segmentation_Tasks/nnUNet_install/nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py",
        'nnUNetv2_train': f"/home/ec2-user/project/final/BraTS_2023_2024_solutions/Segmentation_Tasks/nnUNet_install/nnunetv2/run/run_training.py",
        'nnUNetTrainer': f'/home/ec2-user/project/final/BraTS_2023_2024_solutions/Segmentation_Tasks/nnUNet_install/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py',
        'plan_json_original': f"/home/ec2-user/project/final/BraTS_2023_2024_solutions/Segmentation_Tasks/example/nnUNetPlans_2023_glioma.json"
    }
    
    # 환경변수 설정
    for key in ['nnUNet_preprocessed', 'nnUNet_results', 'nnUNet_raw']:
        os.environ[key] = paths[key]
        print(f"{key}: {paths[key]}")
    
    return paths

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

def run_and_log_realtime(cmd, log_file_path):
    """명령어 실행 및 실시간 로깅"""
    import subprocess
    
    print(f"\n{'='*50}")
    print(f"Executing: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print(f"{'='*50}")
    
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"\n{'='*50}\n")
        log_file.write(f"Executing: {' '.join(cmd) if isinstance(cmd, list) else cmd}\n")
        log_file.write(f"{'='*50}\n")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ''):
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            output_line = f"{timestamp} [stdout] {line.rstrip()}"
            print(output_line)
            log_file.write(output_line + '\n')
            log_file.flush()
        
        process.wait()
        
        log_file.write(f"\n--- Command finished with exit code {process.returncode} ---\n")
        print(f"\n--- Command finished with exit code {process.returncode} ---")
        
        return process.returncode

def create_dataset_json(destination_folder):
    """dataset.json 파일 생성"""
    json_content = {
        "channel_names": {
            "0": "t1c",
            "1": "t1n", 
            "2": "t2f",
            "3": "t2"
        },
        "labels": {
            "background": 0,
            "whole tumor": [1, 2, 3],
            "tumor core": [2, 3], 
            "enhancing tumor": 3,
        },
        "numTraining": len(os.listdir(os.path.join(destination_folder, "labelsTr"))),
        "file_ending": ".nii.gz",
        "regions_class_order": [1, 2, 3]
    }
    
    with open(f'{destination_folder}/dataset.json', 'w') as json_file:
        json.dump(json_content, json_file, indent=4)

def convert_brats_data_enhanced(source_brats, source_gligan, destination_folder):
    """원본 + 합성 데이터 변환"""
    os.makedirs(os.path.join(destination_folder, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(destination_folder, "labelsTr"), exist_ok=True)
    
    # 1. 원본 BraTS 데이터 처리
    print("Processing original BraTS data...")
    for patient_dir in os.listdir(source_brats):
        patient_path = os.path.join(source_brats, patient_dir)
        if not os.path.isdir(patient_path):
            continue
            
        # 파일 복사
        for i, suffix in enumerate(['t1c', 't1n', 't2f', 't2w']):
            src_file = os.path.join(patient_path, f"{patient_dir}-{suffix}.nii.gz")
            if os.path.exists(src_file):
                dst_file = os.path.join(destination_folder, "imagesTr", f"{patient_dir}_{i:04d}.nii.gz")
                shutil.copy(src_file, dst_file)
        
        # 라벨 파일
        seg_file = os.path.join(patient_path, f"{patient_dir}-seg.nii.gz")
        if os.path.exists(seg_file):
            dst_seg = os.path.join(destination_folder, "labelsTr", f"{patient_dir}.nii.gz")
            shutil.copy(seg_file, dst_seg)
    
    # 2. 합성 데이터 처리
    print("Processing synthetic GliGAN data...")
    for synthetic_dir in os.listdir(source_gligan):
        synthetic_path = os.path.join(source_gligan, synthetic_dir)
        if not os.path.isdir(synthetic_path):
            continue
            
        # 합성 데이터 파일 매핑
        modality_map = {
            'scan_t1ce': 0,
            'scan_t1': 1, 
            'scan_flair': 2,
            'scan_t2': 3
        }
        
        for filename in os.listdir(synthetic_path):
            if filename.endswith('.nii.gz'):
                src_file = os.path.join(synthetic_path, filename)
                
                if filename.endswith('-seg.nii.gz'):
                    # 라벨 파일
                    dst_seg = os.path.join(destination_folder, "labelsTr", f"{synthetic_dir}.nii.gz")
                    shutil.copy(src_file, dst_seg)
                else:
                    # 이미지 파일
                    for modality, idx in modality_map.items():
                        if modality in filename:
                            dst_file = os.path.join(destination_folder, "imagesTr", f"{synthetic_dir}_{idx:04d}.nii.gz")
                            shutil.copy(src_file, dst_file)
                            break

def main():
    if len(sys.argv) != 3:
        print("Usage: python 02_Segmentation_training_cloud_native_enhanced.py <s3_bucket> <job_id>")
        sys.exit(1)
    
    s3_bucket = sys.argv[1]
    job_id = sys.argv[2]
    
    # 로그 설정
    running_dir = f"/home/ec2-user/project/final/result/{job_id}"
    log_dir = os.path.join(running_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, '02_Segmentation_training_cloud_native_enhanced.log')
    
    print(f"=== Enhanced Segmentation Training Started ===")
    print(f"Job ID: {job_id}")
    print(f"S3 Bucket: {s3_bucket}")
    print(f"Log file: {log_file_path}")
    
    try:
        # 1. 경로 설정
        paths = setup_cloud_paths(job_id)
        
        # 2. Config 로드 및 설정 적용
        config_s3_key = f"userdata/{job_id}/uploads/config.yaml"
        config = load_config_from_s3(s3_bucket, config_s3_key)
        
        # Segmentation 설정 가져오기
        seg_config = config.get('segmentation_training', {})
        num_workers = seg_config.get('num_workers', 2)
        batch_size = seg_config.get('batch_size', 2)
        dataset_id = 232  # Dataset ID 232 사용
        
        print(f"Enhanced Segmentation training config:")
        print(f"  num_workers: {num_workers}")
        print(f"  batch_size: {batch_size}")
        print(f"  dataset_id: {dataset_id}")
        
        # 3. 데이터 준비 및 변환 (원본 + 합성)
        print("\n--- Step 1: Enhanced Data Preparation ---")
        
        dataset_folder_name = f'Dataset{dataset_id}_BraTS_2023_Enhanced'
        dataset_folder = os.path.join(paths['nnUNet_raw'], dataset_folder_name)
        
        # 데이터 존재 확인
        print(f"Original BraTS data exists: {os.path.exists(paths['source_brats'])}")
        print(f"Synthetic GliGAN data exists: {os.path.exists(paths['source_gligan'])}")
        
        # 원본 + 합성 데이터 변환
        print("Converting original + synthetic data...")
        convert_brats_data_enhanced(paths['source_brats'], paths['source_gligan'], dataset_folder)
        
        # dataset.json 생성
        print("Creating dataset.json...")
        create_dataset_json(dataset_folder)
        
        print(f"Enhanced data preparation completed for dataset {dataset_id}")
        
        # 4. nnUNet 전처리
        print("\n--- Step 2: nnUNet Preprocessing ---")
        cmd_preprocess = [
            "python", paths['nnUNetv2_plan_and_preprocess'],
            "-d", str(dataset_id),
            "-c", "3d_fullres",
            "--verify_dataset_integrity"
        ]
        exit_code = run_and_log_realtime(cmd_preprocess, log_file_path)
        if exit_code != 0:
            raise Exception(f"Preprocessing failed with exit code {exit_code}")
        
        # 5. nnUNet 훈련
        print("\n--- Step 3: nnUNet Training ---")
        
        # 환경변수로 워커 수 제한
        os.environ['nnUNet_n_proc_DA'] = str(num_workers)
        
        cmd_train = [
            "python", paths['nnUNetv2_train'],
            str(dataset_id), "3d_fullres", "0",
            "--npz"
        ]
        exit_code = run_and_log_realtime(cmd_train, log_file_path)
        if exit_code != 0:
            raise Exception(f"Training failed with exit code {exit_code}")
        
        print(f"\n=== Enhanced Segmentation Training Completed Successfully ===")
        
        # S3 업로드
        print("\n--- Step 6: S3 업로드 ---")
        upload_results_to_s3(s3_bucket, job_id, running_dir)
        
    except Exception as e:
        print(f"❌ Enhanced training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
클라우드 네이티브 이미지 품질 분석 스크립트
Usage: python 06_Image_quality_analysis_cloud_native.py <s3_bucket> <job_id>
"""
import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops
import yaml
import boto3
from datetime import datetime

def load_config_from_s3(s3_bucket, s3_key):
    """S3에서 config 파일 로드"""
    try:
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        config_content = response['Body'].read().decode('utf-8')
        return yaml.safe_load(config_content)
    except Exception as e:
        print(f"Config 로드 실패: {e}")
        return {}

def calculate_ssim(img1, img2):
    """구조적 유사성 지수 계산"""
    # 정규화
    img1_norm = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8)
    img2_norm = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8)
    
    # 2D 슬라이스별 SSIM 계산 후 평균
    ssim_scores = []
    for i in range(min(img1_norm.shape[2], img2_norm.shape[2])):
        score = ssim(img1_norm[:,:,i], img2_norm[:,:,i], data_range=1.0)
        ssim_scores.append(score)
    
    return np.mean(ssim_scores)

def calculate_texture_features(image):
    """텍스처 특성 계산"""
    # 이미지를 8비트로 변환
    img_norm = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    
    # 중간 슬라이스 선택
    mid_slice = img_norm[:,:,img_norm.shape[2]//2]
    
    # GLCM 계산
    glcm = graycomatrix(mid_slice, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # 텍스처 특성 추출
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    
    return {
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'energy': energy
    }

def calculate_intensity_stats(image):
    """픽셀 강도 통계 계산"""
    return {
        'mean': np.mean(image),
        'std': np.std(image),
        'min': np.min(image),
        'max': np.max(image),
        'median': np.median(image),
        'q25': np.percentile(image, 25),
        'q75': np.percentile(image, 75)
    }

def calculate_snr_cnr(image):
    """SNR/CNR 계산"""
    # 간단한 SNR 계산 (신호 대 노이즈 비)
    signal = np.mean(image[image > np.percentile(image, 75)])
    noise = np.std(image[image < np.percentile(image, 25)])
    snr = signal / (noise + 1e-8)
    
    # 간단한 CNR 계산 (대조도 대 노이즈 비)
    high_intensity = np.mean(image[image > np.percentile(image, 90)])
    low_intensity = np.mean(image[image < np.percentile(image, 10)])
    cnr = (high_intensity - low_intensity) / (noise + 1e-8)
    
    return {'snr': snr, 'cnr': cnr}

def analyze_image_quality(synthetic_folder, real_folder, output_dir):
    """이미지 품질 분석 (fake_label_X, real_label_X 폴더 구조 지원)"""
    results = []
    
    # 합성 데이터 케이스 찾기 (fake_label_X 폴더들)
    synthetic_cases = [d for d in os.listdir(synthetic_folder) 
                      if os.path.isdir(os.path.join(synthetic_folder, d)) and 'fake_label' in d]
    
    print(f"발견된 합성 데이터 케이스: {len(synthetic_cases)}")
    
    for case_dir in synthetic_cases:
        case_path = os.path.join(synthetic_folder, case_dir)
        
        # 케이스 ID 추출 (예: BraTS-GLI-00066-000_fake_label_1 -> BraTS-GLI-00066-000)
        base_case_id = case_dir.split('_fake_label')[0]
        
        # 합성 데이터 파일들 찾기
        synthetic_files = {}
        for file in os.listdir(case_path):
            if file.endswith('-scan_t1ce.nii.gz'):
                synthetic_files['t1c'] = os.path.join(case_path, file)
            elif file.endswith('-scan_t1.nii.gz'):
                synthetic_files['t1n'] = os.path.join(case_path, file)
            elif file.endswith('-scan_t2.nii.gz'):
                synthetic_files['t2w'] = os.path.join(case_path, file)
            elif file.endswith('-scan_flair.nii.gz'):
                synthetic_files['t2f'] = os.path.join(case_path, file)
        
        # 대응하는 실제 데이터 찾기
        real_case_path = None
        for real_dir in os.listdir(real_folder):
            if real_dir.startswith(base_case_id) and os.path.isdir(os.path.join(real_folder, real_dir)):
                real_case_path = os.path.join(real_folder, real_dir)
                break
        
        if not real_case_path:
            print(f"⚠️  실제 데이터를 찾을 수 없음: {base_case_id}")
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
        
        # 각 모달리티별 품질 분석
        for modality in ['t1c', 't1n', 't2w', 't2f']:
            if modality in synthetic_files and modality in real_files:
                try:
                    print(f"분석 중: {base_case_id} - {modality}")
                    
                    # 이미지 로드
                    synthetic_img = nib.load(synthetic_files[modality])
                    real_img = nib.load(real_files[modality])
                    
                    synthetic_data = synthetic_img.get_fdata()
                    real_data = real_img.get_fdata()
                    
                    # 품질 지표 계산
                    ssim_val = calculate_ssim(synthetic_data, real_data)
                    snr_cnr_synthetic = calculate_snr_cnr(synthetic_data)
                    snr_cnr_real = calculate_snr_cnr(real_data)
                    
                    results.append({
                        'case_id': base_case_id,
                        'modality': modality,
                        'ssim': ssim_val,
                        'synthetic_snr': snr_cnr_synthetic['snr'],
                        'synthetic_cnr': snr_cnr_synthetic['cnr'],
                        'real_snr': snr_cnr_real['snr'],
                        'real_cnr': snr_cnr_real['cnr'],
                        'synthetic_mean': np.mean(synthetic_data),
                        'real_mean': np.mean(real_data),
                        'synthetic_std': np.std(synthetic_data),
                        'real_std': np.std(real_data)
                    })
                    
                except Exception as e:
                    print(f"❌ 오류 발생 {base_case_id}-{modality}: {e}")
                    continue
    
    return pd.DataFrame(results)

def create_visualizations(df, output_dir):
    """시각화 생성"""
    plt.style.use('default')
    
    # 1. SSIM 분포
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='modality', y='ssim')
    plt.title('Structural Similarity (SSIM) by Modality')
    plt.ylabel('SSIM Score')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ssim_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 강도 통계 비교
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Mean 비교
    axes[0,0].scatter(df['real_mean'], df['synthetic_mean'], alpha=0.6)
    axes[0,0].plot([df['real_mean'].min(), df['real_mean'].max()], 
                   [df['real_mean'].min(), df['real_mean'].max()], 'r--')
    axes[0,0].set_xlabel('Real Mean Intensity')
    axes[0,0].set_ylabel('Synthetic Mean Intensity')
    axes[0,0].set_title('Mean Intensity Comparison')
    
    # STD 비교
    axes[0,1].scatter(df['real_std'], df['synthetic_std'], alpha=0.6)
    axes[0,1].plot([df['real_std'].min(), df['real_std'].max()], 
                   [df['real_std'].min(), df['real_std'].max()], 'r--')
    axes[0,1].set_xlabel('Real Std Intensity')
    axes[0,1].set_ylabel('Synthetic Std Intensity')
    axes[0,1].set_title('Standard Deviation Comparison')
    
    # SNR 비교
    axes[1,0].scatter(df['real_snr'], df['synthetic_snr'], alpha=0.6)
    axes[1,0].plot([df['real_snr'].min(), df['real_snr'].max()], 
                   [df['real_snr'].min(), df['real_snr'].max()], 'r--')
    axes[1,0].set_xlabel('Real SNR')
    axes[1,0].set_ylabel('Synthetic SNR')
    axes[1,0].set_title('SNR Comparison')
    
    # CNR 비교
    axes[1,1].scatter(df['real_cnr'], df['synthetic_cnr'], alpha=0.6)
    axes[1,1].plot([df['real_cnr'].min(), df['real_cnr'].max()], 
                   [df['real_cnr'].min(), df['real_cnr'].max()], 'r--')
    axes[1,1].set_xlabel('Real CNR')
    axes[1,1].set_ylabel('Synthetic CNR')
    axes[1,1].set_title('CNR Comparison')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'intensity_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def upload_results_to_s3(s3_bucket, job_id, local_result_dir):
    """결과를 S3에 업로드"""
    try:
        import subprocess
        s3_result_path = f"s3://{s3_bucket}/userdata/{job_id}/result/"
        
        cmd = ["aws", "s3", "sync", local_result_dir, s3_result_path, "--quiet"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ S3 업로드 성공!")
            return True
        else:
            print(f"❌ S3 업로드 실패: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ S3 업로드 중 오류: {e}")
        return False

def main():
    """메인 실행 함수"""
    if len(sys.argv) != 3:
        print("Usage: python 06_Image_quality_analysis_cloud_native.py <s3_bucket> <job_id>")
        sys.exit(1)
    
    s3_bucket = sys.argv[1]
    job_id = sys.argv[2]
    
    print("="*80)
    print("이미지 품질 분석 (Cloud Native)")
    print("="*80)
    print(f"S3 Bucket: {s3_bucket}")
    print(f"Job ID: {job_id}")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 로그 설정
        running_dir = f"/home/ec2-user/project/final/result/{job_id}"
        log_dir = f'{running_dir}/log'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, '06_Image_quality_analysis_cloud_native.log')
        
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
                except: pass
            try:
                if os.path.exists(local_path): s3_client.upload_file(local_path, s3_bucket, s3_key)
            except: pass
        
        # S3 실시간 업로드 시작
        import threading
        stop_event = threading.Event()
        s3_log_key = f"userdata/{job_id}/result/log/06_Image_quality_analysis_cloud_native.log"
        uploader_thread = threading.Thread(target=periodic_s3_uploader, args=(log_file, s3_bucket, s3_log_key, 3, stop_event))
        uploader_thread.daemon = True
        uploader_thread.start()

        # 표준 출력을 로그 파일과 콘솔에 동시 출력
        log_f = open(log_file, 'w', encoding='utf-8')
        sys.stdout = Tee(sys.stdout, log_f)
        
        # 1. Config 로드
        config_s3_key = f"userdata/{job_id}/uploads/config.yaml"
        config = load_config_from_s3(s3_bucket, config_s3_key)
        
        # 2. 경로 설정
        running_dir = f"/home/ec2-user/project/final/result/{job_id}"
        synthetic_folder = f'{running_dir}/GliGAN/Checkpoint/brats2023/Synthetic_dataset_random_labels'
        real_folder = f'/home/ec2-user/s3-data/userdata/{job_id}/uploads/data'
        output_dir = f'{running_dir}/analysis/06_image_quality'
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 3. 이미지 품질 분석
        print("\n--- Step 1: 이미지 품질 분석 ---")
        df = analyze_image_quality(synthetic_folder, real_folder, output_dir)
        
        if len(df) == 0:
            print("❌ 분석할 데이터가 없습니다!")
            return
        
        # 4. 결과 저장
        print("\n--- Step 2: 결과 저장 ---")
        df.to_csv(os.path.join(output_dir, 'image_quality_results.csv'), index=False)
        
        # 5. 시각화 생성
        print("\n--- Step 3: 시각화 생성 ---")
        create_visualizations(df, output_dir)
        
        # 6. 요약 통계
        print("\n--- Step 4: 요약 통계 ---")
        summary_stats = df.groupby('modality')[['ssim', 'synthetic_snr', 'real_snr', 'synthetic_cnr', 'real_cnr']].mean()
        summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
        
        # 7. S3 업로드
        print("\n--- Step 5: S3 업로드 ---")
        upload_results_to_s3(s3_bucket, job_id, running_dir)
        
        print(f"\n=== 이미지 품질 분석 완료! ===")
        print(f"결과 위치: {output_dir}")
        print(f"분석된 케이스 수: {len(df)}")
        print(f"평균 SSIM: {df['ssim'].mean():.3f}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)
    
    print(f"\n완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

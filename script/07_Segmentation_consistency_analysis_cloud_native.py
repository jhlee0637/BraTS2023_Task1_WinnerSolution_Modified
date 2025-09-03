#!/usr/bin/env python3
"""
클라우드 네이티브 Segmentation 일관성 분석 스크립트
Usage: python 07_Segmentation_consistency_analysis_cloud_native.py <s3_bucket> <job_id>
"""
import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import yaml
import boto3
from datetime import datetime
import gc

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

def dice_score(pred, gt, label):
    """Dice Score 계산"""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    
    del pred_mask, gt_mask
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return 2.0 * intersection / union

def sensitivity(pred, gt, label):
    """Sensitivity 계산"""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    tp = np.sum(pred_mask & gt_mask)
    fn = np.sum(~pred_mask & gt_mask)
    
    del pred_mask, gt_mask
    
    if tp + fn == 0:
        return 1.0
    return tp / (tp + fn)

def specificity(pred, gt, label):
    """Specificity 계산"""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    tn = np.sum(~pred_mask & ~gt_mask)
    fp = np.sum(pred_mask & ~gt_mask)
    
    del pred_mask, gt_mask
    
    if tn + fp == 0:
        return 1.0
    return tn / (tn + fp)

def evaluate_case(pred_file, gt_file, model_name, case_name):
    """단일 케이스 평가"""
    pred_nii = nib.load(pred_file)
    gt_nii = nib.load(gt_file)
    
    pred_data = pred_nii.get_fdata().astype(np.uint8)
    gt_data = gt_nii.get_fdata().astype(np.uint8)
    
    del pred_nii, gt_nii
    
    results = {'Model': model_name, 'Case': case_name}
    
    # 개별 라벨별 평가
    for label in [1, 2, 3]:
        label_name = {1: 'NCR_NET', 2: 'ED', 3: 'ET'}[label]
        results[f'Dice_{label_name}'] = dice_score(pred_data, gt_data, label)
        results[f'Sens_{label_name}'] = sensitivity(pred_data, gt_data, label)
        results[f'Spec_{label_name}'] = specificity(pred_data, gt_data, label)
    
    # 복합 영역 평가
    pred_wt = (pred_data > 0).astype(np.uint8)
    gt_wt = (gt_data > 0).astype(np.uint8)
    results['Dice_WT'] = dice_score(pred_wt, gt_wt, 1)
    results['Sens_WT'] = sensitivity(pred_wt, gt_wt, 1)
    results['Spec_WT'] = specificity(pred_wt, gt_wt, 1)
    
    pred_tc = ((pred_data == 1) | (pred_data == 3)).astype(np.uint8)
    gt_tc = ((gt_data == 1) | (gt_data == 3)).astype(np.uint8)
    results['Dice_TC'] = dice_score(pred_tc, gt_tc, 1)
    results['Sens_TC'] = sensitivity(pred_tc, gt_tc, 1)
    results['Spec_TC'] = specificity(pred_tc, gt_tc, 1)
    
    del pred_data, gt_data, pred_wt, gt_wt, pred_tc, gt_tc
    gc.collect()
    
    return results

def analyze_model(pred_folder, gt_folder, model_name):
    """모델별 전체 분석"""
    results_list = []
    pred_files = [f for f in os.listdir(pred_folder) if f.endswith('.nii.gz')]
    
    print(f"\n🔍 {model_name} 모델 분석 중...")
    print(f"   발견된 파일 수: {len(pred_files)}")
    
    if len(pred_files) == 0:
        print(f"   ❌ {pred_folder}에서 .nii.gz 파일을 찾을 수 없습니다!")
        return []
    
    for pred_file in sorted(pred_files):
        case_name = pred_file.replace('.nii.gz', '')
        pred_path = os.path.join(pred_folder, pred_file)
        
        # Ground truth 파일 찾기
        base_case = case_name
        gt_file = f"{base_case}-seg.nii.gz"
        gt_path = os.path.join(gt_folder, base_case, gt_file)
        
        if not os.path.exists(gt_path):
            print(f"   ⚠️  {case_name}의 정답 파일을 찾을 수 없습니다: {gt_path}")
            continue
        
        try:
            results = evaluate_case(pred_path, gt_path, model_name, case_name)
            results_list.append(results)
            print(f"   ✅ {case_name}: Dice WT={results['Dice_WT']:.3f}")
        except Exception as e:
            print(f"   ❌ {case_name} 분석 실패: {e}")
    
    print(f"   📊 {model_name}: 총 {len(results_list)}개 케이스 분석 완료")
    return results_list

def create_visualizations(df, output_dir):
    """시각화 생성"""
    plt.style.use('default')
    
    # 1. Dice Score 비교 박스플롯
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    dice_metrics = ['Dice_WT', 'Dice_TC', 'Dice_ET', 'Dice_NCR_NET']
    titles = ['Whole Tumor (WT)', 'Tumor Core (TC)', 'Enhancing Tumor (ET)', 'NCR/NET']
    
    for i, (metric, title) in enumerate(zip(dice_metrics, titles)):
        ax = axes[i//2, i%2]
        sns.boxplot(data=df, x='Model', y=metric, ax=ax)
        ax.set_title(f'{title} - Dice Score Comparison')
        ax.set_ylabel('Dice Score')
        ax.set_ylim(0, 1)
        
        # 평균값 표시
        for j, model in enumerate(df['Model'].unique()):
            mean_val = df[df['Model'] == model][metric].mean()
            ax.text(j, mean_val + 0.02, f'{mean_val:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_score_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 전체 성능 레이더 차트
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    metrics = ['Dice_WT', 'Dice_TC', 'Dice_ET', 'Sens_WT', 'Sens_TC', 'Sens_ET']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model][metrics].mean().tolist()
        model_data += model_data[:1]
        
        ax.plot(angles, model_data, 'o-', linewidth=2, label=model)
        ax.fill(angles, model_data, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance Comparison\n(Radar Chart)', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.savefig(os.path.join(output_dir, 'performance_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 케이스별 성능 히트맵
    pivot_data = df.pivot(index='Case', columns='Model', values='Dice_WT')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                center=0.8, vmin=0, vmax=1)
    plt.title('Case-wise Dice WT Performance Heatmap')
    plt.ylabel('Cases')
    plt.xlabel('Models')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'case_performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def statistical_comparison(df, output_dir):
    """통계적 비교 분석"""
    models = df['Model'].unique()
    if len(models) != 2:
        print("⚠️  통계 비교는 2개 모델에서만 가능합니다")
        return None
    
    model1, model2 = models
    metrics = ['Dice_WT', 'Dice_TC', 'Dice_ET', 'Sens_WT', 'Sens_TC', 'Sens_ET']
    
    stats_results = []
    
    for metric in metrics:
        data1 = df[df['Model'] == model1][metric]
        data2 = df[df['Model'] == model2][metric]
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(data1, data2)
        
        # Wilcoxon signed-rank test
        w_stat, w_p_value = stats.wilcoxon(data1, data2)
        
        stats_results.append({
            'Metric': metric,
            f'{model1}_Mean': data1.mean(),
            f'{model1}_Std': data1.std(),
            f'{model2}_Mean': data2.mean(),
            f'{model2}_Std': data2.std(),
            'T_Statistic': t_stat,
            'T_P_Value': p_value,
            'Wilcoxon_Statistic': w_stat,
            'Wilcoxon_P_Value': w_p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
    
    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv(os.path.join(output_dir, 'statistical_comparison.csv'), index=False)
    
    return stats_df

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
        print("Usage: python 07_Segmentation_consistency_analysis_cloud_native.py <s3_bucket> <job_id>")
        sys.exit(1)
    
    s3_bucket = sys.argv[1]
    job_id = sys.argv[2]
    
    print("="*80)
    print("Segmentation 일관성 분석 (Cloud Native)")
    print("="*80)
    print(f"S3 Bucket: {s3_bucket}")
    print(f"Job ID: {job_id}")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 로그 설정
        running_dir = f"/home/ec2-user/project/final/result/{job_id}"
        log_dir = f'{running_dir}/log'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, '07_Segmentation_consistency_analysis_cloud_native.log')
        
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
        s3_log_key = f"userdata/{job_id}/result/log/07_Segmentation_consistency_analysis_cloud_native.log"
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
        
        # 우승팀 모델 결과 (Real+GAN 데이터로 훈련됨)
        enhanced_model_results = f'{running_dir}/BraTS2023_inference/output'
        
        # 실제 데이터만으로 훈련된 모델 결과 (가정: 별도 훈련 필요)
        real_only_model_results = f'{running_dir}/real_only_segmentation/output'
        
        # Ground Truth
        ground_truth_folder = f'/home/ec2-user/s3-data/userdata/{job_id}/uploads/data'
        
        # 출력 디렉토리
        output_dir = f'{running_dir}/analysis/07_segmentation_consistency'
        os.makedirs(output_dir, exist_ok=True)
        
        # 3. 모델 분석
        print("\n--- Step 1: 모델별 성능 분석 ---")
        
        # Enhanced Model (우승팀 모델) 분석
        enhanced_results = analyze_model(enhanced_model_results, ground_truth_folder, "Enhanced_Model_RealGAN")
        
        # Real Only Model 분석 (존재하는 경우)
        real_only_results = []
        if os.path.exists(real_only_model_results):
            real_only_results = analyze_model(real_only_model_results, ground_truth_folder, "Real_Only_Model")
        else:
            print("⚠️  Real Only Model 결과를 찾을 수 없습니다. Enhanced Model만 분석합니다.")
        
        if not enhanced_results and not real_only_results:
            print("❌ 분석할 데이터가 없습니다!")
            return
        
        # 4. 결과 통합
        print("\n--- Step 2: 결과 통합 ---")
        all_results = enhanced_results + real_only_results
        df = pd.DataFrame(all_results)
        
        # 5. 결과 저장
        df.to_csv(os.path.join(output_dir, 'detailed_comparison.csv'), index=False)
        
        # 6. 요약 통계
        summary = df.groupby('Model')[['Dice_WT', 'Dice_TC', 'Dice_ET', 'Sens_WT', 'Sens_TC', 'Sens_ET']].agg(['mean', 'std'])
        summary.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
        
        # 7. 시각화 생성
        print("\n--- Step 3: 시각화 생성 ---")
        create_visualizations(df, output_dir)
        
        # 8. 통계적 비교
        print("\n--- Step 4: 통계적 비교 ---")
        stats_df = statistical_comparison(df, output_dir)
        
        # 9. S3 업로드
        print("\n--- Step 5: S3 업로드 ---")
        upload_results_to_s3(s3_bucket, job_id, running_dir)
        
        # 10. 결과 출력
        print("\n" + "="*80)
        print("📋 Segmentation 일관성 분석 결과")
        print("="*80)
        
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            print(f"\n🎯 {model} 성능:")
            print(f"   Dice WT: {model_data['Dice_WT'].mean():.4f} ± {model_data['Dice_WT'].std():.4f}")
            print(f"   Dice TC: {model_data['Dice_TC'].mean():.4f} ± {model_data['Dice_TC'].std():.4f}")
            print(f"   Dice ET: {model_data['Dice_ET'].mean():.4f} ± {model_data['Dice_ET'].std():.4f}")
        
        if stats_df is not None:
            print(f"\n📊 통계적 유의성:")
            for _, row in stats_df.iterrows():
                significance = "🔴 유의함" if row['Significant'] == 'Yes' else "🟢 유의하지 않음"
                print(f"   {row['Metric']}: p={row['T_P_Value']:.4f} ({significance})")
        
        print(f"\n💾 결과 저장 위치: {output_dir}")
        print(f"분석된 케이스 수: {len(df)}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)
    
    print(f"\n완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

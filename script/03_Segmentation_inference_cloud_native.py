#!/usr/bin/env python3
"""
클라우드 네이티브 BraTS2023 Segmentation Inference 스크립트
Usage: python 03_Segmentation_inference_cloud_native.py <s3_bucket> <job_id>
"""
import os
import sys
import shutil
import boto3
import yaml
from datetime import datetime

def upload_results_to_s3(s3_bucket, job_id, local_results_dir):
    """결과를 S3에 업로드"""
    s3_client = boto3.client('s3')
    
    try:
        import subprocess
        s3_prefix = f"userdata/{job_id}/result"
        
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
        'input_folder': f'/home/ec2-user/s3-data/userdata/{job_id}/uploads/data',
        'brats_inference_dir': f'{running_dir}/BraTS2023_inference',
        'nnUNet_results': f'{running_dir}/nnUNet/nnUNet_results',
        'source_main': '/home/ec2-user/project/final/BraTS_2023_2024_solutions/Segmentation_Tasks/BraTS2023_inference/main.py',
        'source_infer': '/home/ec2-user/project/final/BraTS_2023_2024_solutions/Segmentation_Tasks/BraTS2023_inference/infer_low_disk.py'
    }
    
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

def setup_brats_inference_environment(paths):
    """BraTS2023_inference 환경 설정"""
    brats_dir = paths['brats_inference_dir']
    
    # BraTS2023_inference 디렉토리 생성
    os.makedirs(brats_dir, exist_ok=True)
    
    # 필요한 파일들 복사
    shutil.copy2(paths['source_main'], os.path.join(brats_dir, 'main.py'))
    shutil.copy2(paths['source_infer'], os.path.join(brats_dir, 'infer_low_disk.py'))
    
    print(f"BraTS2023_inference 환경 설정 완료: {brats_dir}")
    return brats_dir

def run_brats_inference(paths, config):
    """BraTS 표준 inference 실행"""
    brats_dir = paths['brats_inference_dir']
    input_folder = paths['input_folder']
    nnunet_results = paths['nnUNet_results']
    output_folder = os.path.join(brats_dir, 'output')
    
    # 작업 디렉토리를 BraTS2023_inference로 변경
    original_cwd = os.getcwd()
    os.chdir(brats_dir)
    
    try:
        # BraTS inference 실행
        cmd = [
            "python", "main.py",
            "--data_path", input_folder,
            "--output_path", output_folder,
            "--nnUNet_results", nnunet_results
        ]
        
        print(f"실행 명령어: {' '.join(cmd)}")
        print(f"작업 디렉토리: {brats_dir}")
        
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ BraTS inference 성공!")
            print(f"결과 저장 위치: {output_folder}")
            return True
        else:
            print(f"❌ BraTS inference 실패: {result.stderr}")
            return False
            
    finally:
        # 원래 작업 디렉토리로 복원
        os.chdir(original_cwd)

def run_and_log_realtime(cmd, log_file_path, s3_bucket=None, s3_key=None):
    """실시간 로그와 함께 명령어 실행"""
    import subprocess
    import threading
    import boto3
    
    def periodic_s3_uploader(local_path, s3_bucket, s3_key, interval, stop_event):
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
    
    print(f"실행 명령어: {' '.join(cmd)}")
    
    uploader_thread = None
    stop_event = threading.Event()
    if s3_bucket and s3_key:
        uploader_thread = threading.Thread(
            target=periodic_s3_uploader,
            args=(log_file_path, s3_bucket, s3_key, 3, stop_event)
        )
        uploader_thread.daemon = True
        uploader_thread.start()
    
    with open(log_file_path, 'w') as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip())
            log_file.write(line)
            log_file.flush()
        
        process.wait()
    
    if uploader_thread:
        stop_event.set()
        uploader_thread.join()
    
    return process.returncode

def main():
    if len(sys.argv) != 3:
        print("Usage: python 03_Segmentation_inference_cloud_native.py <s3_bucket> <job_id>")
        sys.exit(1)
    
    s3_bucket = sys.argv[1]
    job_id = sys.argv[2]
    
    print("="*80)
    print("BraTS2023 Segmentation Inference (Cloud Native)")
    print("="*80)
    print(f"S3 Bucket: {s3_bucket}")
    print(f"Job ID: {job_id}")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 경로 설정
        print("\n--- Step 1: 경로 설정 ---")
        paths = setup_cloud_paths(job_id)
        
        # 2. Config 로드
        print("\n--- Step 2: Config 로드 ---")
        config_s3_key = f"userdata/{job_id}/uploads/config.yaml"
        config = load_config_from_s3(s3_bucket, config_s3_key)
        
        # 3. BraTS2023_inference 환경 설정
        print("\n--- Step 3: BraTS2023_inference 환경 설정 ---")
        brats_dir = setup_brats_inference_environment(paths)
        
        # 4. 입력 데이터 확인
        print("\n--- Step 4: 입력 데이터 확인 ---")
        input_folder = paths['input_folder']
        if not os.path.exists(input_folder):
            raise Exception(f"입력 폴더가 존재하지 않습니다: {input_folder}")
        
        input_files = []
        for root, dirs, files in os.walk(input_folder):
            input_files.extend([f for f in files if f.endswith('.nii.gz')])
        print(f"입력 파일 수: {len(input_files)}")
        
        # 5. nnUNet 결과 확인
        print("\n--- Step 5: nnUNet 모델 확인 ---")
        nnunet_results = paths['nnUNet_results']
        if not os.path.exists(nnunet_results):
            raise Exception(f"nnUNet 결과 폴더가 존재하지 않습니다: {nnunet_results}")
        
        # 6. BraTS inference 실행
        print("\n--- Step 6: BraTS Inference 실행 ---")
        success = run_brats_inference(paths, config)
        
        if success:
            print(f"\n✅ BraTS2023 Segmentation Inference 완료!")
            print(f"결과 위치: {os.path.join(brats_dir, 'output')}")
            
            # S3 업로드
            print("\n--- Step 7: S3 업로드 ---")
            running_dir = f"/home/ec2-user/project/final/result/{job_id}"
            upload_results_to_s3(s3_bucket, job_id, running_dir)
        else:
            raise Exception("BraTS inference 실패")
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)
    
    print(f"\n완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

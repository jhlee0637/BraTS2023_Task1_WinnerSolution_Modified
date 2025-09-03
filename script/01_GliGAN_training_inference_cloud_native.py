#!/usr/bin/env python3
"""
AWS 기반 GLiGAN 모델 훈련 및 추론 스크립트 (Manifest 및 DB 연동 버전)
- S3의 Manifest 파일을 읽어 필요한 데이터만 동적으로 다운로드합니다.
- AWS Secrets Manager에서 DB 접속 정보를 안전하게 가져옵니다.
- 작업 완료 또는 실패 시, RDS(Aurora) DB에 작업 상태를 업데이트합니다.
"""
# 표준 라이브러리
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
import json
import psycopg2 # DB 연결을 위해 추가
import pytz     # 시간대 설정을 위해 추가

# 서드파티 라이브러리
import boto3
import yaml
from botocore.exceptions import ClientError, NoCredentialsError

# --- 새로운 헬퍼 함수들 ---

def get_db_secrets():
    """AWS Secrets Manager에서 DB 접속 정보를 가져옵니다."""
    secret_name = "rds-aurora-credentials"
    region_name = "ap-northeast-2"
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)
    try:
        print("Retrieving database credentials from Secrets Manager...")
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret = get_secret_value_response['SecretString']
        print("Successfully retrieved credentials.")
        return json.loads(secret)
    except ClientError as e:
        print(f"ERROR: Could not retrieve secret '{secret_name}'.")
        raise e

def update_job_status_in_db(job_id, status, db_creds, is_starting=False, is_ending=False):
    """작업 상태를 DB에 업데이트 (start_time, end_time 포함)"""
    conn = None
    try:
        print(f"Connecting to database to update job '{job_id}' to status '{status}'...")
        conn = psycopg2.connect(
            host=db_creds['host'], port=db_creds.get('port', 5432),
            dbname=db_creds['dbname'], user=db_creds['username'], password=db_creds['password']
        )
        with conn.cursor() as cur:
            now_utc = datetime.now(pytz.utc)
            if is_starting:
                sql = "UPDATE jobs SET status = %s, start_time = %s WHERE job_id = %s"
                cur.execute(sql, (status, now_utc, job_id))
            elif is_ending:
                sql = "UPDATE jobs SET status = %s, end_time = %s WHERE job_id = %s"
                cur.execute(sql, (status, now_utc, job_id))
            else:
                sql = "UPDATE jobs SET status = %s WHERE job_id = %s"
                cur.execute(sql, (status, job_id))
            conn.commit()
        print(f"Database update successful.")
    except Exception as e:
        print(f"WARNING: Database status update failed: {e}")
    finally:
        if conn:
            conn.close()

def wait_for_s3_mount_files(csv_path, max_wait=300):
    """CSV에 있는 파일들이 S3 마운트에서 접근 가능할 때까지 대기"""
    import pandas as pd
    import time
    
    print(f"Checking S3 mounted files accessibility...")
    df = pd.read_csv(csv_path)
    
    start_time = time.time()
    for idx, row in df.iterrows():
        files_to_check = [row['scan_t1ce'], row['scan_t2'], row['scan_flair'], row['scan_t1'], row['label']]
        
        for file_path in files_to_check:
            wait_count = 0
            while not os.path.exists(file_path):
                if time.time() - start_time > max_wait:
                    raise TimeoutError(f"File not accessible after {max_wait}s: {file_path}")
                
                if wait_count % 10 == 0:  # 10초마다 로그
                    print(f"Waiting for file: {os.path.basename(file_path)}")
                
                time.sleep(1)
                wait_count += 1
    
    print(f"All files accessible in S3 mount. Total wait: {time.time() - start_time:.1f}s")


def localize_data_from_manifest(s3_bucket, job_id, local_data_dir):
    """manifest.csv를 읽어 S3 마운트된 경로에서 데이터에 접근합니다."""
    s3_mount_dir = "/home/ec2-user/s3-data"
    manifest_path = f"{s3_mount_dir}/userdata/{job_id}/uploads/manifest.csv"
    
    print(f"Reading manifest from mounted S3: {manifest_path}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    print("Reading manifest and using S3 mounted paths...")
    with open(manifest_path, 'r') as f:
        s3_uris = [line.strip() for line in f if line.strip().startswith('s3://')]

    # S3 URI를 마운트된 로컬 경로로 변환
    mounted_paths = []
    for s3_uri in s3_uris:
        # s3://say1-2team-bucket/data/... -> /home/ec2-user/s3-data/data/...
        s3_path = s3_uri.replace(f"s3://{s3_bucket}/", "")
        mounted_path = os.path.join(s3_mount_dir, s3_path)
        mounted_paths.append(mounted_path)
        
    print(f"Using {len(mounted_paths)} files from S3 mount (no download needed)")
    return mounted_paths


def check_s3_connection(bucket_to_test: str) -> bool:
    """
    S3 연결 및 필요한 권한(읽기, 쓰기, 삭제)을 종합적으로 확인합니다.

    EC2 인스턴스에 할당된 IAM 역할을 사용하여 권한을 테스트합니다.

    Args:
        bucket_to_test (str): 테스트할 S3 버킷 이름.

    Returns:
        bool: 연결 및 권한이 모두 정상이면 True, 그렇지 않으면 False.
    """
    print("=" * 50)
    print("AWS S3 Connection and Permissions Check")

    # 1. Boto3 S3 클라이언트 생성 확인
    try:
        s3_client = boto3.client('s3')
        print("✅ Step 1: S3 클라이언트 생성 성공 (Boto3 Handoff OK)")
    except Exception as e:
        print(f"❌ Step 1: S3 클라이언트 생성 실패: {e}")
        print("   ㄴ Boto3 라이브러리가 설치되어 있는지 확인하세요.")
        return False

    # 2. AWS 자격 증명(IAM 역할) 확인
    try:
        sts_client = boto3.client('sts')
        identity = sts_client.get_caller_identity()
        role_arn = identity['Arn']
        print(f"✅ Step 2: IAM 역할 자격 증명 확인 성공 (Assumed Role: {role_arn})")
    except (NoCredentialsError, ClientError) as e:
        print(f"❌ Step 2: IAM 역할 자격 증명 확인 실패: {e}")
        print("   ㄴ EC2 인스턴스에 IAM 역할이 제대로 연결되었는지 확인하세요.")
        return False

    # 3. 특정 버킷 접근 가능 여부 확인 (ListObjectsV2 권한)
    try:
        s3_client.head_bucket(Bucket=bucket_to_test)
        print(f"✅ Step 3: '{bucket_to_test}' 버킷 접근 성공 (s3:ListBucket OK)")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"❌ Step 3: '{bucket_to_test}' 버킷을 찾을 수 없음 (404 Not Found)")
            print("   ㄴ 버킷 이름이 정확한지 확인하세요.")
        elif error_code == '403':
            print(f"❌ Step 3: '{bucket_to_test}' 버킷 접근 권한 없음 (403 Forbidden)")
            print("   ㄴ IAM 역할에 's3:ListBucket' 권한이 있는지 확인하세요.")
        else:
            print(f"❌ Step 3: 버킷 접근 중 에러 발생: {e}")
        return False

    # 4. 파일 업로드/다운로드/삭제 권한 종합 테스트
    test_key = 's3-connection-check.tmp'
    test_body = b'S3 connection check file.'
    try:
        # 4-1. 업로드 테스트 (s3:PutObject)
        s3_client.put_object(Bucket=bucket_to_test, Key=test_key, Body=test_body)
        print("     ✅ Upload successful (s3:PutObject OK)")

        # 4-2. 다운로드 (s3:GetObject)
        response = s3_client.get_object(Bucket=bucket_to_test, Key=test_key)
        downloaded_body = response['Body'].read()
        assert test_body == downloaded_body, "콘텐츠 불일치!"
        print("     ✅ Download successful and content verified (s3:GetObject OK)")
    except ClientError as e:
        print(f"❌ Step 4: 파일 처리 중 권한 에러 발생: {e}")
        print("   ㄴ IAM 역할에 's3:PutObject' 또는 's3:GetObject' 권한을 확인하세요.")
        return False
    except Exception as e:
        print(f"❌ Step 4: 파일 처리 중 예상치 못한 에러 발생: {e}")
        return False
    finally:
        # 4-3. 테스트 파일 삭제 (s3:DeleteObject)
        try:
            s3_client.delete_object(Bucket=bucket_to_test, Key=test_key)
            print("     ✅ Cleanup successful (s3:DeleteObject OK)")
        except ClientError as e:
            print(f"❌ Step 4: 테스트 파일 삭제 실패: {e}")
            print("   ㄴ IAM 역할에 's3:DeleteObject' 권한을 확인하세요.")

    print("=" * 50)
    return True


def load_config_from_s3(s3_bucket: str, s3_key: str):
    """S3에서 YAML 설정 파일을 다운로드하고 로드합니다."""
    s3_client = boto3.client('s3')
    
    print(f"[Config] Loading configuration from s3://{s3_bucket}/{s3_key}")
    
    try:
        # S3에서 config 파일 다운로드
        response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
        config_content = response['Body'].read().decode('utf-8')
        
        # YAML 파싱
        config = yaml.safe_load(config_content)
        print("✅ [Config] Successfully loaded configuration from S3")
        return config
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            print(f"❌ [Config] Configuration file not found: s3://{s3_bucket}/{s3_key}")
        elif error_code == 'NoSuchBucket':
            print(f"❌ [Config] Bucket not found: {s3_bucket}")
        else:
            print(f"❌ [Config] Error accessing S3: {e}")
        raise
    except yaml.YAMLError as e:
        print(f"❌ [Config] Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        print(f"❌ [Config] Unexpected error loading config from S3: {e}")
        raise


def sync_s3_data_to_local(s3_bucket: str, s3_data_prefix: str, local_data_dir: str) -> bool:
    """
    S3에서 훈련 데이터를 로컬로 동기화합니다.
    
    Args:
        s3_bucket (str): S3 버킷 이름
        s3_data_prefix (str): S3 데이터 prefix
        local_data_dir (str): 로컬 데이터 디렉토리
        
    Returns:
        bool: 성공 시 True, 실패 시 False
    """
    if not s3_data_prefix:
        print("S3 data prefix not specified. Skipping data download.")
        return True
        
    s3_uri = f"s3://{s3_bucket}/{s3_data_prefix}"
    
    print(f"\n[S3 Data Sync] Syncing training data from {s3_uri} to {local_data_dir}")
    
    try:
        # 로컬 데이터 디렉토리 생성
        os.makedirs(local_data_dir, exist_ok=True)
        
        # AWS CLI s3 sync 명령 실행
        cmd = ["aws", "s3", "sync", s3_uri, local_data_dir, "--quiet"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ [S3 Data Sync] Successfully synced training data from {s3_uri}")
            return True
        else:
            print(f"❌ [S3 Data Sync] Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ [S3 Data Sync] Unexpected error: {e}")
        return False


def periodic_s3_uploader(local_path, s3_bucket, s3_key, interval, stop_event):
    """지정된 로컬 파일을 주기적으로 S3에 업로드합니다."""
    s3_client = boto3.client('s3')
    print(f"[S3 Uploader] Starting periodic uploads of '{local_path}' to "
          f"'s3://{s3_bucket}/{s3_key}' every {interval} seconds.")

    while not stop_event.wait(interval):
        try:
            if os.path.exists(local_path):
                s3_client.upload_file(local_path, s3_bucket, s3_key)
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[S3 Uploader] Log synced to S3 at {now}")
        except Exception as e:
            print(f"[S3 Uploader] Error uploading to S3: {e}")

    print("[S3 Uploader] Stop signal received. Performing final upload...")
    try:
        if os.path.exists(local_path):
            s3_client.upload_file(local_path, s3_bucket, s3_key)
            print("[S3 Uploader] Final log sync complete. Thread finished.")
    except Exception as e:
        print(f"[S3 Uploader] Error during final upload: {e}")


def upload_results_to_s3(local_brats_dir, s3_bucket, s3_prefix):
    """로컬 디렉토리의 모든 파일을 재귀적으로 S3에 업로드합니다."""
    s3_client = boto3.client('s3')

    print("\n[Final Upload] Starting upload of results to S3...")
    print(f"Local path: {local_brats_dir}")
    print(f"S3 destination: s3://{s3_bucket}/{s3_prefix}")

    if not os.path.exists(local_brats_dir):
        print(f"❌ Local directory not found: {local_brats_dir}")
        return False

    upload_count = 0
    error_count = 0

    try:
        for root, _, files in os.walk(local_brats_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_brats_dir)
                s3_key = os.path.join(s3_prefix, relative_path).replace('\\', '/')

                try:
                    s3_client.upload_file(local_file_path, s3_bucket, s3_key)
                    upload_count += 1
                    print(f"✅ Uploaded: {relative_path}")
                except Exception as e:
                    error_count += 1
                    print(f"❌ Failed to upload {relative_path}: {e}")

        print("\n[Final Upload] Upload completed!")
        print(f"✅ Successfully uploaded: {upload_count} files")
        if error_count > 0:
            print(f"❌ Failed uploads: {error_count} files")

        return error_count == 0

    except Exception as e:
        print(f"❌ [Final Upload] Unexpected error during upload: {e}")
        return False


def setup_working_dir(result_dir: str, job_id: str) -> str:
    """job_id를 기반으로 작업 디렉토리를 생성하고 경로를 반환합니다."""
    if not all(c.isalnum() or c in '_-' for c in job_id):
        raise ValueError(f"Invalid characters in job_id: {job_id}")

    running_dir = os.path.join(result_dir, job_id)
    log_dir = os.path.join(running_dir, "log")

    if os.path.exists(running_dir):
        print(f"Warning: Working directory '{running_dir}' already exists.")

    os.makedirs(log_dir, exist_ok=True)
    return running_dir


def get_cmd(config: dict, python_file: str, config_name: str, data_path: str = None) -> list:
    """설정 파일과 파이썬 파일명을 기반으로 실행할 명령어 리스트를 생성합니다."""
    cmd = ["python", "-u", python_file]
    
    config_params = config.get(config_name, {})
    # config 파일에 datadir이 있다면 제거 (동적 경로로 덮어쓰기 위함)
    config_params.pop('datadir', None)

    for parameter, value in config_params.items():
        cmd.append(f"--{parameter}")
        cmd.append(str(value))
    
    # csv_creator 단계일 때만 --datadir 인자를 동적 경로로 추가
    if config_name == "csv_creator" and data_path:
        cmd.extend(['--datadir', data_path])
    
    return cmd


def stream_reader(stream, log_file_handle, stream_name):
    """프로세스의 stdout/stderr 스트림을 읽어 실시간으로 로깅합니다."""
    try:
        for line in iter(stream.readline, ''):
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_line = f"[{timestamp}] [{stream_name}] {line}"
            print(log_line, end='')
            log_file_handle.write(log_line)
            log_file_handle.flush()
    finally:
        stream.close()


def run_and_log_realtime(cmd, log_file_path, s3_bucket=None, s3_key=None):
    """
    명령어를 실행하고 표준 출력/에러를 파일에 실시간으로 로깅하며,
    주기적으로 S3에 로그를 업로드합니다.
    """
    cmd_str = ' '.join(cmd)
    print("\n" + "=" * 50)
    print(f"Executing: {cmd_str}")
    print(f"Logging to: {log_file_path}")
    print("=" * 50)

    uploader_thread = None
    stop_event = threading.Event()
    if s3_bucket and s3_key:
        uploader_thread = threading.Thread(
            target=periodic_s3_uploader,
            args=(log_file_path, s3_bucket, s3_key, 3, stop_event)
        )
        uploader_thread.daemon = True
        uploader_thread.start()

    return_code = -1
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 50 + "\n")
            f.write(f"Executing: {cmd_str}\n")
            f.write("=" * 50 + "\n")
            f.flush()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )

            stdout_thread = threading.Thread(
                target=stream_reader, args=(process.stdout, f, "stdout"))
            stderr_thread = threading.Thread(
                target=stream_reader, args=(process.stderr, f, "stderr"))
            stdout_thread.start()
            stderr_thread.start()

            return_code = process.wait()

            stdout_thread.join()
            stderr_thread.join()

            f.write(f"\n--- Command finished with exit code {return_code} ---\n")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        return_code = -1
    finally:
        if uploader_thread:
            stop_event.set()
            uploader_thread.join()

    return return_code


def main():
    """스크립트의 메인 실행 함수입니다."""
    
    # 1. 초기 설정 및 인자 확인
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <s3_bucket> <job_id>")
        sys.exit(1)
    
    s3_bucket = sys.argv[1]
    job_id = sys.argv[2]
    db_creds = None
    
    try:
        # 2. 필수 자격 증명 로드 (가장 먼저 수행)
        print("--- Step 1: Loading Credentials ---")
        db_creds = get_db_secrets()
        
        # 3. 작업 시작을 DB에 알림
        update_job_status_in_db(job_id, 'RUNNING', db_creds, is_starting=True)
        
        # 4. 로컬 작업 환경 설정
        print("\n--- Step 2: Setting up Local Environment ---")
        BASE_PROJECT_DIR = "/home/ec2-user/project/final"
        RESULT_DIR = os.path.join(BASE_PROJECT_DIR, "result")
        WINNER_DIR = os.path.join(BASE_PROJECT_DIR, "BraTS_2023_2024_solutions")
        running_dir = setup_working_dir(RESULT_DIR, job_id)
        localized_data_dir = os.path.join(running_dir, 'data')
        
        # 5. S3에서 원본 manifest.csv와 config.yaml 다운로드
        print("\n--- Step 3: Downloading Config and Manifest Files ---")
        
        manifest_s3_key = f"userdata/{job_id}/uploads/manifest.csv"
        config_s3_key = f"userdata/{job_id}/uploads/config.yaml"
        
        local_manifest_path = os.path.join(running_dir, 'GliGAN', 'Checkpoint', 'brats2023', 'manifest.csv')
        local_config_path = os.path.join(running_dir, 'config.yaml')
        
        os.makedirs(os.path.dirname(local_manifest_path), exist_ok=True)
        os.makedirs(os.path.dirname(local_config_path), exist_ok=True)
        
        s3_client = boto3.client('s3')
        s3_client.download_file(s3_bucket, manifest_s3_key, local_manifest_path)
        s3_client.download_file(s3_bucket, config_s3_key, local_config_path)
        
        print(f"Downloaded manifest from s3://{s3_bucket}/{manifest_s3_key}")
        print(f"Downloaded config from s3://{s3_bucket}/{config_s3_key}")
        
        # Config 파일 로드
        with open(local_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # CSV 경로를 brats2023.csv로 수정
        csv_path = '../../Checkpoint/brats2023/brats2023.csv'
        config['gligan_training_first']['csv_path'] = csv_path
        config['gligan_training_second']['csv_path'] = csv_path
        config['label_generator']['csv_path'] = csv_path
        config['gligan_inference']['csv_path'] = csv_path
        
        # 7. 소스코드 복사 및 로그 파일 경로 설정
        gligan_source_dir = os.path.join(WINNER_DIR, "Segmentation_Tasks", "GliGAN", "src")
        gligan_dest_dir = os.path.join(running_dir, 'GliGAN')
        gligan_src_dir = os.path.join(gligan_dest_dir, 'src')
        shutil.copytree(gligan_source_dir, gligan_src_dir, dirs_exist_ok=True)
        log_file_path = os.path.join(running_dir, "log", "01_GliGAN_training_inference_cloud_native.log")

        # 8. ML 파이프라인 순차 실행
        print("\n--- Step 5: Executing ML Pipeline ---")
        os.chdir(os.path.join(gligan_src_dir, 'train'))
        
        print("\n[Pipeline Stage 5.1] CSV 생성 (S3 마운트 경로 사용)")
        # job별 S3 마운트된 로컬 경로 사용
        s3_mount_data_path = f"/home/ec2-user/s3-data/userdata/{job_id}/uploads/data/"
        cmd_csv = get_cmd(config, "csv_creator_cloud.py", "csv_creator", s3_mount_data_path)
        exit_code = run_and_log_realtime(cmd_csv, log_file_path, s3_bucket, f"userdata/{job_id}/result/log/01_GliGAN_training_inference_cloud_native.log")
        if exit_code != 0:
            raise Exception(f"CSV creation failed with exit code {exit_code}")
        
        # S3 마운트 파일 접근성 확인
        csv_file_path = os.path.join(gligan_dest_dir, 'Checkpoint', 'brats2023', 'brats2023.csv')
        wait_for_s3_mount_files(csv_file_path)

        print("\n[Pipeline Stage 5.2] GliGAN Training")
        modalities = ['t1ce', 't1', 't2', 'flair']
        training_stages = ["gligan_training_first", "gligan_training_second"]
        for modality in modalities:
            for stage in training_stages:
                cmd_train = get_cmd(config, "tumour_main.py", stage)
                cmd_train.extend(['--modality', modality])
                exit_code = run_and_log_realtime(cmd_train, log_file_path, s3_bucket, f"userdata/{job_id}/result/log/01_GliGAN_training_inference_cloud_native.log")
                if exit_code != 0:
                    raise Exception(f"Training failed for {modality} {stage} with exit code {exit_code}")

        print("\n[Pipeline Stage 5.3] Label 생성")
        cmd_label = get_cmd(config, "label_main.py", "label_generator")
        exit_code = run_and_log_realtime(cmd_label, log_file_path, s3_bucket, f"userdata/{job_id}/result/log/01_GliGAN_training_inference_cloud_native.log")
        if exit_code != 0:
            raise Exception(f"Label generation failed with exit code {exit_code}")

        print("\n[Pipeline Stage 5.4] GliGAN Inference")
        os.chdir(os.path.join(gligan_src_dir, 'infer'))
        cmd_infer = get_cmd(config, "main_random_label_random_dataset_generator_multiprocess.py", "gligan_inference")
        exit_code = run_and_log_realtime(cmd_infer, log_file_path, s3_bucket, f"userdata/{job_id}/result/log/01_GliGAN_training_inference_cloud_native.log")
        if exit_code != 0:
            raise Exception(f"Inference failed with exit code {exit_code}")

        # 9. 최종 결과물 S3 업로드
        print("\n--- Step 6: Uploading Results to S3 ---")
        brats_dir = os.path.join(gligan_dest_dir, "Checkpoint", "brats2023")
        s3_result_prefix = f"userdata/{job_id}/result/"
        upload_results_to_s3(brats_dir, s3_bucket, s3_result_prefix)

        # 10. 작업 완료 상태 파일 생성
        status_file = os.path.join(running_dir, 'COMPLETED')
        with open(status_file, 'w') as f:
            f.write(f"Job {job_id} completed successfully at {datetime.now().isoformat()}\n")
        print(f"✅ Status file created: {status_file}")
        
        print("\n--- Pipeline Finished Successfully ---")

    except Exception as e:
        # 11. 에러 발생 시 에러 상태 파일 생성
        print(f"\n--- FATAL ERROR occurred during the pipeline: {e} ---")
        
        error_file = os.path.join(running_dir, 'FAILED')
        with open(error_file, 'w') as f:
            f.write(f"Job {job_id} failed at {datetime.now().isoformat()}\n")
            f.write(f"Error: {str(e)}\n")
        print(f"❌ Error status file created: {error_file}")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
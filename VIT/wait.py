import subprocess
import time


def get_gpu_utilization(gpu_index=0):
    """获取指定GPU的计算利用率百分比"""
    cmd = [
        'nvidia-smi',
        '--query-gpu=utilization.gpu',
        '--format=csv,noheader,nounits',
        '-i', str(gpu_index)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi命令执行失败: {result.stderr}")
    return int(result.stdout.strip())


def wait_for_gpu(threshold=20, gpu_index=0, check_interval=10):
    """
    等待直到指定GPU的使用率低于或等于阈值

    参数：
        threshold: 使用率阈值（百分比）
        gpu_index: 要监控的GPU索引
        check_interval: 检查间隔时间（秒）
    """
    while True:
        try:
            util = get_gpu_utilization(gpu_index)
            if util <= threshold:
                print(f"GPU {gpu_index} 使用率 {util}% ≤ {threshold}%，继续执行")
                break
            print(f"GPU {gpu_index} 使用率 {util}% > {threshold}%，等待{check_interval}秒...")
            time.sleep(check_interval)
        except Exception as e:
            print(f"GPU监控出错: {e}")
            time.sleep(check_interval)  # 防止错误时高频重试


# 使用示例
if __name__ == "__main__":
    print("等待GPU资源释放...")
    wait_for_gpu(
        threshold=60,  # 使用率阈值设为50%
        gpu_index=0,  # 监控第一个GPU
        check_interval=10  # 每5秒检查一次
    )
    print("开始执行GPU计算任务")

    # 这里添加你的PyTorch代码
    # import torch
    # model = torch.nn.Linear(10, 10).cuda()
    # ...
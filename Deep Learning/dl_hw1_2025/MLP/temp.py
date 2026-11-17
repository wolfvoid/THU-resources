import os
import shutil
import glob

def reorganize_tensorboard_files(runs_dir):
    """将平铺的events文件重新组织到子文件夹"""

    # 检查目录是否存在
    if not os.path.exists(runs_dir):
        print(f"错误：目录 {runs_dir} 不存在")
        return

    # 查找所有events文件（包括可能的变体）
    patterns = patterns = [
        "events.out.tfevents.*",      # 标准模式
        "events.*",                    # 简化模式
        "*tfevents*",                 # 包含tfevents
        "*events*",                   # 包含events
        "*.tfevents.*",               # 点号分隔
    ]

    all_events_files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(runs_dir, pattern))
        all_events_files.extend(files)

    # 去重
    all_events_files = list(set(all_events_files))

    print(f"找到 {len(all_events_files)} 个events文件:")
    for f in all_events_files:
        print(f"  - {os.path.basename(f)}")

    if len(all_events_files) == 0:
        print("没有找到events文件，无法重组")
        return

    # 重组文件
    for i, event_file in enumerate(all_events_files, 1):
        # 创建实验目录
        exp_dir = os.path.join(runs_dir, f"experiment_{i:03d}")
        os.makedirs(exp_dir, exist_ok=True)

        # 移动文件
        new_path = os.path.join(exp_dir, os.path.basename(event_file))

        try:
            shutil.move(event_file, new_path)
            print(f"移动成功: {os.path.basename(event_file)} -> experiment_{i:03d}/")
        except Exception as e:
            print(f"移动失败 {event_file}: {e}")

    print("重组完成！")

# 执行重组
reorganize_tensorboard_files('./runs')

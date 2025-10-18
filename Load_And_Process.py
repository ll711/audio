#load audio

# python
from pathlib import Path
from typing import List
import numpy as np
import librosa

def load_speakers_wav_to_matrix(root_dir: Path,
                                first: int = 1,
                                last: int = 20,
                                per_speaker: int = 20,
                                sr: int = 16000,
                                mono: bool = True) -> np.ndarray:
    """
    Load .wav files from speaker{first..last} (also supports speak{i}).
    Return a 2D numpy array with shape (num_valid_speakers, per_speaker) and dtype=object.
    Each cell is a 1D waveform ndarray; pad with None if a speaker has fewer than per_speaker files.
    加载 speaker{first..last}（兼容 speak{i}）中的 .wav。
    返回 shape=(有效说话人数量, per_speaker)、dtype=object 的二维 numpy 数组。
    每个单元格为 1D 波形 ndarray；若不足 per_speaker，则用 None 补齐。
    """
    # Normalize root directory to Path
    # 将根目录标准化为 Path
    root_dir = Path(root_dir)

    # Accumulators for data rows and missing speakers
    # 用于存放数据行与缺失说话人索引的容器
    rows: List[List[object]] = []
    missing = []

    for i in range(first, last + 1):
        # Prefer 'speaker{i}', fall back to 'speak{i}'
        # 优先使用 'speaker{i}'，否则回退到 'speak{i}'
        d1 = root_dir / f"speaker{i}"
        d2 = root_dir / f"speak{i}"
        speaker_dir = d1 if d1.exists() else (d2 if d2.exists() else None)
        if not speaker_dir:
            # Record missing speaker index and continue
            # 记录缺失的说话人索引并继续
            missing.append(i)
            continue

        # Collect .wav files and sort; limit to per_speaker
        # 收集并排序 .wav 文件；截取至 per_speaker 个
        wav_files = sorted(speaker_dir.glob("*.wav"))
        if len(wav_files) < per_speaker:
            print(f"警告: `{speaker_dir}` 仅找到 {len(wav_files)} 个 wav，将用 None 补齐到 {per_speaker}。")

        wav_files = wav_files[:per_speaker]

        # Load each audio file into a row
        # 加载每个音频文件并填充到当前行
        row: List[object] = []
        for p in wav_files:
            try:
                # Load audio with target sample rate and mono option
                # 按目标采样率及单声道选项加载音频
                y, _ = librosa.load(str(p), sr=sr, mono=mono)
                row.append(y)
            except Exception as e:
                # On failure, keep a placeholder to preserve alignment
                # 失败时用占位符保持列对齐
                print(f"跳过文件 `{p}`，原因: {e}")
                row.append(None)

        # Pad the row to a fixed number of columns
        # 将当前行补齐到固定列数
        if len(row) < per_speaker:
            row.extend([None] * (per_speaker - len(row)))

        rows.append(row)

    # Build the 2D numpy object matrix
    # 构建二维 numpy 对象矩阵
    mat = np.array(rows, dtype=object)
    print(f"矩阵形状: {mat.shape}" + (f"；缺失的 speaker 索引: {missing}" if missing else ""))
    return mat


if __name__ == "__main__":
    # Root directory points to 'sound-simple' under the current file
    # 根目录指向当前文件下的 'sound-simple'
    root = Path(__file__).resolve().parent / "sound-simple"  # or an absolute path
    # 或使用绝对路径

    matrix = load_speakers_wav_to_matrix(root, first=1, last=20, per_speaker=20, sr=16000, mono=True)

    # Example: access the first speaker's first audio (1D ndarray)
    # 示例：访问第 1 位说话人的第 1 条音频（一维 ndarray）
    if matrix.size and matrix[0, 0] is not None:
        print(f"第一位说话人第1条音频样本数: {len(matrix[0, 0])}")

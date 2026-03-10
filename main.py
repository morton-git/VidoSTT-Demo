# ==================== VidoSTT Colab 版 v2.1 主程式 ====================
# 作者：ya xian lin
# 目的：讓人下載後本地運行語音轉文字（含說話者分離）
# 使用：python main.py --input_file "your_audio.mp3" [其他參數]
# 注意：需先 pip install -r requirements.txt
#       如果用 GPU，需有 CUDA 支持
#       這是從 Colab Cell 2 + Cell 3 合併的版本，已調整為本地運行（無 files.upload/download）

import os
import time
import subprocess
import gc
import warnings
from datetime import datetime
from faster_whisper import WhisperModel
import opencc
import torch
import soundfile as sf
import argparse  # 用來處理命令列輸入

warnings.filterwarnings('ignore')

# ── 全局變數 ──
INPUT_LOCAL = './input'  # 輸入資料夾
OUTPUT_LOCAL = './output'  # 輸出資料夾
os.makedirs(INPUT_LOCAL, exist_ok=True)
os.makedirs(OUTPUT_LOCAL, exist_ok=True)

# ── Cell 2 部分：說話者分離設定 ──
# 這個函數處理 HuggingFace Token 和模型載入
def setup_diarization(hf_token):
    global USE_DIARIZATION, diarization_pipeline
    USE_DIARIZATION = False
    diarization_pipeline = None

    if not hf_token.strip():
        print('ℹ️ 未填入 Token — 將使用基本 A/B 模式（無需 Token）')
        print('   → 直接繼續轉錄')
    else:
        try:
            import torch
            from huggingface_hub import login
            from pyannote.audio import Pipeline

            print('🔄 登入 HuggingFace...')
            login(token=hf_token.strip(), add_to_git_credential=False)
            print('✅ 登入成功')

            print('🔄 載入說話者分離模型（pyannote/speaker-diarization-3.1）...')
            print('   首次使用需下載模型，約需 1-2 分鐘...')
            diarization_pipeline = Pipeline.from_pretrained(
                'pyannote/speaker-diarization-3.1',
                use_auth_token=hf_token.strip()
            )

            if torch.cuda.is_available():
                diarization_pipeline = diarization_pipeline.to(torch.device('cuda'))
                print(f'✅ 說話者分離就緒（GPU：{torch.cuda.get_device_name(0)}）')
            else:
                print('✅ 說話者分離就緒（CPU 模式，速度較慢）')

            USE_DIARIZATION = True
            print('→ 現在可使用真實說話者分離')

        except Exception as e:
            err = str(e)
            print(f'\n❌ 載入失敗')
            if '401' in err or 'Unauthorized' in err:
                print('   原因：Token 無效')
                print('   解決：請重新建立 Read 權限的 Token')
                print('   連結：https://huggingface.co/settings/tokens')
            elif '403' in err or 'gated' in err.lower() or 'terms' in err.lower() or 'access' in err.lower():
                print('   原因：尚未接受模型使用條款')
                print('   解決：請點選以下連結，登入後按「Agree」:')
                print('   ① https://huggingface.co/pyannote/segmentation-3.0')
                print('   ② https://huggingface.co/pyannote/speaker-diarization-3.1')
            else:
                print(f'   錯誤訊息：{err[:200]}')
                print('   建議：確認網路正常，或改用基本模式')
            USE_DIARIZATION = False
            diarization_pipeline = None

# ── Cell 3 部分：轉錄主邏輯 ──
# 已調整為本地檔案輸入/輸出
def run_transcription(args):
    # SPEAKER_LABELS 等變數定義
    SPEAKER_LABELS = {'SPEAKER_00': args.speaker_0_name, 'SPEAKER_01': args.speaker_1_name}

    # 確認說話者分離是否已在 setup_diarization 初始化
    # (已全局變數處理)

    print('━'*50)
    mode_label = '✅ 說話者分離（真實聲紋）' if USE_DIARIZATION else '⬜ 基本模式（A/B 交替，速度較快）'
    print(f'📊 目前模式：{mode_label}')
    print('━'*50)

    # ── 取得音檔 ──
    audio_file = args.input_file
    if not audio_file or not os.path.exists(audio_file):
        print('❌ 未提供有效輸入檔案，請使用 --input_file 指定本地音檔（支援 mp4/mp3/wav/m4a）')
        return

    # ── Step 1：提取 16kHz 單聲道 WAV ──
    print('\n🔧 Step 1：轉換音訊格式為 16kHz WAV...')
    tmp_wav = './tmp_16k.wav'
    r = subprocess.run(
        ['ffmpeg', '-y', '-i', audio_file, '-ar', '16000', '-ac', '1', tmp_wav],
        capture_output=True, text=True
    )
    if not os.path.exists(tmp_wav):
        print('❌ 音訊轉換失敗！請確認檔案格式（支援 mp4/mp3/wav/m4a）')
        print(r.stderr[-300:] if r.stderr else '')
        return
    else:
        print('✅ 音訊轉換完成')

    # ── Step 2：Whisper 轉錄 ──
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compute = 'float16' if device == 'cuda' else 'int8'
    print(f'\n🧠 Step 2：載入 Whisper [{args.model_quality}]（{device.upper()}）...')
    model = WhisperModel(args.model_quality, device=device, compute_type=compute)
    t0 = time.time()
    print('🎙️ 轉錄中...')
    segs_gen, info = model.transcribe(
        tmp_wav, beam_size=5, language='zh', vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500), chunk_length=16
    )
    whisper_segs = list(segs_gen)
    print(f'✅ 轉錄完成（{time.time()-t0:.1f}s，語言信心：{info.language_probability:.1%}，共 {len(whisper_segs)} 段）')

    # ── Step 3：說話者分離（可選）──
    dia_segs = []
    if USE_DIARIZATION and diarization_pipeline:
        print('\n🔊 Step 3：說話者分離中...')
        t1 = time.time()
        audio_data, sr = sf.read(tmp_wav, dtype='float32')
        if audio_data.ndim == 1:
            audio_data = audio_data[None, :]
        waveform = torch.tensor(audio_data)
        output = diarization_pipeline(
            {'waveform': waveform, 'sample_rate': sr},
            num_speakers=args.num_speakers
        )
        for turn, _, speaker in output.itertracks(yield_label=True):
            label = SPEAKER_LABELS.get(speaker, speaker)
            dia_segs.append((turn.start, turn.end, label))
        print(f'✅ 說話者分離完成（{time.time()-t1:.1f}s，共 {len(dia_segs)} 段）')
    else:
        print('\nℹ️ Step 3：跳過說話者分離（使用 A/B 基本模式）')

    # ── Step 4：合併生成 SRT / TXT ──
    print('\n🇹🇼 Step 4：生成繁體字幕...')

    def fmt_time(s):
        h = int(s // 3600); m = int((s % 3600) // 60)
        sec = int(s % 60); ms = int((s - int(s)) * 1000)
        return f'{h:02d}:{m:02d}:{sec:02d},{ms:03d}'

    def find_speaker(start, end):
        best_spk, best_overlap = None, 0
        for ds, de, spk in dia_segs:
            overlap = min(end, de) - max(start, ds)
            if overlap > best_overlap:
                best_overlap, best_spk = overlap, spk
        return best_spk

    converter = opencc.OpenCC('s2twp')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f'VidoSTT_{args.model_quality}_{timestamp}'

    srt_lines, txt_lines = [], []
    speaker_cycle = [args.speaker_0_name, args.speaker_1_name]
    srt_idx = 0

    for i, seg in enumerate(whisper_segs, 1):
        text = converter.convert(seg.text.strip())
        if not text:
            continue
        srt_idx += 1
        if dia_segs:
            speaker = find_speaker(seg.start, seg.end) or speaker_cycle[i % 2]
        else:
            speaker = speaker_cycle[(i-1) % 2]
        srt_lines += [str(srt_idx),
                      f'{fmt_time(seg.start)} --> {fmt_time(seg.end)}',
                      f'[{speaker}] {text}', '']
        txt_lines.append(f'[{fmt_time(seg.start)}]【{speaker}】 {text}')

    # ── 儲存檔案（本地版，取代 files.download） ──
    srt_path = f'{OUTPUT_LOCAL}/{base_name}.srt'
    txt_path = f'{OUTPUT_LOCAL}/{base_name}.txt'
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(srt_lines))
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_lines))

    total = time.time() - t0
    print('━'*50)
    print(f'🎉 完成！總耗時：{total:.1f} 秒')
    print(f'📄 SRT：{srt_path}  （共 {srt_idx} 段）')
    print(f'📜 TXT：{txt_path}')
    print('━'*50)
    print('✅ 檔案已儲存到 output/ 資料夾，可重新執行處理下一個檔案')

    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VidoSTT 語音轉文字工具")
    parser.add_argument('--hf_token', type=str, default='', help='HuggingFace Token（用於說話者分離）')
    parser.add_argument('--input_file', type=str, required=True, help='輸入音檔路徑（本地檔案，支援 mp4/mp3/wav/m4a）')
    parser.add_argument('--model_quality', type=str, default='medium', choices=['tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3'], help='Whisper 模型品質')
    parser.add_argument('--num_speakers', type=int, default=2, help='說話者數量')
    parser.add_argument('--speaker_0_name', type=str, default='主持人', help='說話者0名稱')
    parser.add_argument('--speaker_1_name', type=str, default='來賓', help='說話者1名稱')

    args = parser.parse_args()
    setup_diarization(args.hf_token)
    run_transcription(args)

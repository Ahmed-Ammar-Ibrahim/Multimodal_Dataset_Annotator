import os

# تفعيل محرك FFmpeg المدمج في Qt6 لضمان تشغيل الفيديو والصوت على Windows بدون شاشة سوداء
os.environ.setdefault("QT_MEDIA_BACKEND", "ffmpeg")

import csv
import re
import sys
import traceback
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import av
from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

EMOTIONS = ["Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise", "Normal"]
CSV_HEADERS = [
    "segment_id",
    "source_video",
    "start_time_str",
    "end_time_str",
    "start_sec",
    "end_sec",
    "duration_sec",
    "visual_emotion",
    "audio_emotion",
    "text_emotion",
    "intensity",
    "confidence",
    "annotator_id",
    "transcript",
    "notes",
    "video_path",
    "audio_path",
    "text_path",
]


def format_time_m_ss_cc(ms: int) -> str:
    """تحويل المللي ثانية إلى صيغة دقائق.ثواني.أجزاء من الثانية (مثال: 0.45.02 أو 1.20.25)"""
    if ms < 0:
        ms = 0
    total_seconds = ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    centiseconds = (ms % 1000) // 10
    return f"{minutes}.{seconds:02d}.{centiseconds:02d}"


def parse_time_m_ss_cc(time_str: str) -> int:
    """
    تحويل النص بصيغة (دقائق.ثواني.أجزاء_من_الثانية) مثل 0.45.02 أو 1.20.25 إلى مللي ثانية.
    يرفع ValueError إذا كانت الصيغة غير صحيحة.
    """
    cleaned = time_str.strip()
    match = re.fullmatch(r"(\d+)\.(\d{1,2})\.(\d{1,2})", cleaned)
    if not match:
        raise ValueError(
            f"صيغة الوقت غير صحيحة: '{time_str}'.\n"
            "الصيغة المطلوبة: دقائق.ثواني.أجزاء من الثانية (مثال: 0.45.02 أو 1.20.25)"
        )
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    centiseconds_str = match.group(3)
    centiseconds = int(centiseconds_str) * 10 if len(centiseconds_str) == 1 else int(centiseconds_str)

    if seconds >= 60:
        raise ValueError(f"عدد الثواني يجب أن يكون أقل من 60 (القيمة المدخلة: {seconds}).")

    return (minutes * 60 * 1000) + (seconds * 1000) + (centiseconds * 10)


def _rescale_packet_ts(pkt: av.Packet, src_tb: Fraction, dst_tb: Optional[Fraction]) -> None:
    """تحويل الطوابع الزمنية للحزمة من time_base المشفّر إلى time_base الحاوية بدقة."""
    if dst_tb is None:
        pkt.time_base = src_tb
        return
    if src_tb != dst_tb:
        ratio = src_tb / dst_tb
        if pkt.pts is not None:
            pkt.pts = int(round(pkt.pts * ratio))
        if pkt.dts is not None:
            pkt.dts = int(round(pkt.dts * ratio))
        if pkt.duration is not None and pkt.duration > 0:
            pkt.duration = max(1, int(round(pkt.duration * ratio)))
    pkt.time_base = dst_tb


class MediaSegmentExtractor:
    """مسؤول عن قص وتصدير الفيديو MP4 والصوت WAV باستخدام PyAV بدون الاعتماد على ffmpeg.exe الخارجي."""

    @staticmethod
    def extract_segment(
        source_path: str,
        start_sec: float,
        end_sec: float,
        out_video_path: str,
        out_audio_path: str,
    ) -> None:
        if end_sec <= start_sec:
            raise ValueError("وقت النهاية يجب أن يكون أكبر من وقت البداية.")

        # 1. استخراج الفيديو إلى MP4 (H.264 + AAC إن وجد صوت)
        MediaSegmentExtractor._export_video_mp4(source_path, start_sec, end_sec, out_video_path)

        # 2. استخراج الصوت المستقل إلى WAV (PCM 16-bit, 16kHz Mono مناسب لنماذج المشاعر)
        MediaSegmentExtractor._export_audio_wav(source_path, start_sec, end_sec, out_audio_path)

    @staticmethod
    def _export_video_mp4(
        source_path: str, start_sec: float, end_sec: float, out_video_path: str
    ) -> None:
        input_container = None
        output_container = None
        try:
            input_container = av.open(source_path)
            in_video_streams = [s for s in input_container.streams if s.type == "video"]
            if not in_video_streams:
                raise RuntimeError("ملف الفيديو المصدر لا يحتوي على مسار فيديو (Video Stream).")

            in_v_stream = in_video_streams[0]
            in_audio_streams = [s for s in input_container.streams if s.type == "audio"]
            in_a_stream = in_audio_streams[0] if in_audio_streams else None

            output_container = av.open(out_video_path, mode="w")

            # تحديد معدل الإطارات (FPS)
            fps = 30
            if in_v_stream.average_rate is not None and float(in_v_stream.average_rate) > 0:
                fps = max(1, round(float(in_v_stream.average_rate)))
            elif in_v_stream.guessed_rate is not None and float(in_v_stream.guessed_rate) > 0:
                fps = max(1, round(float(in_v_stream.guessed_rate)))

            width = in_v_stream.codec_context.width
            height = in_v_stream.codec_context.height
            if not width or not height:
                raise RuntimeError("تعذر تحديد أبعاد الفيديو (Width/Height).")

            # التأكد من أن الأبعاد زوجية لمتطلبات libx264
            width = width - (width % 2)
            height = height - (height % 2)

            v_codec_tb = Fraction(1, fps)
            out_v_stream = output_container.add_stream(
                "libx264",
                rate=fps,
                options={"bf": "0", "preset": "fast", "crf": "20"},
            )
            out_v_stream.width = width
            out_v_stream.height = height
            out_v_stream.pix_fmt = "yuv420p"
            out_v_stream.codec_context.time_base = v_codec_tb

            # إعداد ترميز الصوت AAC داخل ملف MP4 إذا كان الفيديو الأصلي يحتوي على صوت
            out_a_stream = None
            resampler_mp4 = None
            a_codec_tb = None
            if in_a_stream is not None:
                sample_rate = in_a_stream.codec_context.sample_rate or 44100
                a_codec_tb = Fraction(1, sample_rate)
                out_a_stream = output_container.add_stream("aac", rate=sample_rate)
                out_a_stream.layout = "stereo"
                out_a_stream.codec_context.time_base = a_codec_tb
                resampler_mp4 = av.AudioResampler(
                    format="fltp",
                    layout="stereo",
                    rate=sample_rate,
                )

            seek_sec = max(0.0, start_sec - 2.0)
            input_container.seek(int(seek_sec * av.time_base))

            streams_to_demux = [in_v_stream] + ([in_a_stream] if in_a_stream else [])
            video_frame_count = 0
            audio_samples_written = 0
            last_v_dts: Optional[int] = None
            last_a_dts: Optional[int] = None
            video_done = False
            audio_done = in_a_stream is None

            def _mux_video_packet(pkt: av.Packet) -> None:
                nonlocal last_v_dts
                _rescale_packet_ts(pkt, v_codec_tb, out_v_stream.time_base)
                if pkt.dts is not None:
                    if last_v_dts is not None and pkt.dts <= last_v_dts:
                        pkt.dts = last_v_dts + 1
                    if pkt.pts is not None and pkt.pts < pkt.dts:
                        pkt.pts = pkt.dts
                    last_v_dts = pkt.dts
                output_container.mux(pkt)

            def _mux_audio_packet(pkt: av.Packet) -> None:
                nonlocal last_a_dts
                if out_a_stream is not None and a_codec_tb is not None:
                    _rescale_packet_ts(pkt, a_codec_tb, out_a_stream.time_base)
                if pkt.dts is not None:
                    if last_a_dts is not None and pkt.dts <= last_a_dts:
                        pkt.dts = last_a_dts + 1
                    if pkt.pts is not None and pkt.pts < pkt.dts:
                        pkt.pts = pkt.dts
                    last_a_dts = pkt.dts
                output_container.mux(pkt)

            for packet in input_container.demux(streams_to_demux):
                if video_done and audio_done:
                    break
                if packet.dts is None:
                    continue

                for frame in packet.decode():
                    if packet.stream.type == "video":
                        if video_done:
                            continue
                        if frame.pts is not None and in_v_stream.time_base is not None:
                            frame_time = float(frame.pts * in_v_stream.time_base)
                        else:
                            frame_time = video_frame_count / float(fps)

                        if frame_time < start_sec:
                            continue
                        if frame_time > end_sec:
                            video_done = True
                            if audio_done:
                                break
                            continue

                        rgb_frame = frame.reformat(width=width, height=height, format="yuv420p")
                        rgb_frame.pts = video_frame_count
                        rgb_frame.time_base = v_codec_tb
                        rgb_frame.pict_type = 0
                        video_frame_count += 1

                        for out_pkt in out_v_stream.encode(rgb_frame):
                            _mux_video_packet(out_pkt)

                    elif packet.stream.type == "audio" and out_a_stream is not None and resampler_mp4 is not None:
                        if audio_done:
                            continue
                        if frame.pts is not None and in_a_stream.time_base is not None:
                            a_time = float(frame.pts * in_a_stream.time_base)
                        else:
                            sr = in_a_stream.codec_context.sample_rate or 44100
                            a_time = audio_samples_written / float(sr)

                        if a_time < start_sec:
                            continue
                        if a_time > end_sec:
                            audio_done = True
                            if video_done:
                                break
                            continue

                        resampled_frames = resampler_mp4.resample(frame)
                        for r_frame in resampled_frames:
                            r_frame.pts = audio_samples_written
                            r_frame.time_base = a_codec_tb
                            audio_samples_written += r_frame.samples
                            for out_pkt in out_a_stream.encode(r_frame):
                                _mux_audio_packet(out_pkt)

                if video_done and audio_done:
                    break

            if video_frame_count == 0:
                raise RuntimeError(
                    f"لم يتم العثور على أي إطارات فيديو في النطاق الزمني المحدد ({start_sec:.2f}s - {end_sec:.2f}s)."
                )

            # تفريغ الـ Buffers المتبقية
            for out_pkt in out_v_stream.encode():
                _mux_video_packet(out_pkt)

            if out_a_stream is not None:
                for out_pkt in out_a_stream.encode():
                    _mux_audio_packet(out_pkt)

        finally:
            if output_container is not None:
                output_container.close()
            if input_container is not None:
                input_container.close()

    @staticmethod
    def _export_audio_wav(
        source_path: str, start_sec: float, end_sec: float, out_audio_path: str
    ) -> None:
        input_container = None
        output_container = None
        try:
            input_container = av.open(source_path)
            in_audio_streams = [s for s in input_container.streams if s.type == "audio"]
            if not in_audio_streams:
                raise RuntimeError("الفيديو لا يحتوي على مسار صوتي (Audio Stream) لاستخراج ملف WAV.")

            in_a_stream = in_audio_streams[0]
            target_rate = 16000  # قياسي لتدريب نماذج تحليل المشاعر الصوتية (SER)
            wav_tb = Fraction(1, target_rate)

            output_container = av.open(out_audio_path, mode="w")
            out_a_stream = output_container.add_stream("pcm_s16le", rate=target_rate)
            out_a_stream.layout = "mono"
            out_a_stream.codec_context.time_base = wav_tb

            resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=target_rate,
            )

            seek_sec = max(0.0, start_sec - 2.0)
            input_container.seek(int(seek_sec * av.time_base))

            samples_written = 0
            in_sr = in_a_stream.codec_context.sample_rate or 44100
            decoded_samples_seen = 0
            last_wav_dts: Optional[int] = None
            audio_done = False

            def _mux_wav_packet(pkt: av.Packet) -> None:
                nonlocal last_wav_dts
                _rescale_packet_ts(pkt, wav_tb, out_a_stream.time_base)
                if pkt.dts is not None:
                    if last_wav_dts is not None and pkt.dts <= last_wav_dts:
                        pkt.dts = last_wav_dts + 1
                    if pkt.pts is not None and pkt.pts < pkt.dts:
                        pkt.pts = pkt.dts
                    last_wav_dts = pkt.dts
                output_container.mux(pkt)

            for packet in input_container.demux(in_a_stream):
                if audio_done:
                    break
                if packet.dts is None:
                    continue

                for frame in packet.decode():
                    if frame.pts is not None and in_a_stream.time_base is not None:
                        frame_time = float(frame.pts * in_a_stream.time_base)
                    else:
                        frame_time = seek_sec + (decoded_samples_seen / float(in_sr))

                    decoded_samples_seen += frame.samples

                    if frame_time < start_sec:
                        continue
                    if frame_time > end_sec:
                        audio_done = True
                        break

                    resampled_frames = resampler.resample(frame)
                    for r_frame in resampled_frames:
                        r_frame.pts = samples_written
                        r_frame.time_base = wav_tb
                        samples_written += r_frame.samples
                        for out_pkt in out_a_stream.encode(r_frame):
                            _mux_wav_packet(out_pkt)

                if audio_done:
                    break

            if samples_written == 0:
                raise RuntimeError(
                    f"لم يتم استخراج أي عينات صوتية في النطاق الزمني المحدد ({start_sec:.2f}s - {end_sec:.2f}s)."
                )

            for out_pkt in out_a_stream.encode():
                _mux_wav_packet(out_pkt)

        finally:
            if output_container is not None:
                output_container.close()
            if input_container is not None:
                input_container.close()


class ExtractionWorker(QThread):
    """تشغيل عملية قص الفيديو والصوت في الخلفية لمنع تهنيج الواجهة."""

    finished_ok = Signal(dict)
    finished_err = Signal(dict, str)

    def __init__(self, job: Dict[str, Any]) -> None:
        super().__init__()
        self.job = job

    def run(self) -> None:
        try:
            MediaSegmentExtractor.extract_segment(
                source_path=self.job["source_path"],
                start_sec=self.job["start_sec"],
                end_sec=self.job["end_sec"],
                out_video_path=str(self.job["out_video_path"]),
                out_audio_path=str(self.job["out_audio_path"]),
            )
            self.finished_ok.emit(self.job)
        except Exception as exc:
            detailed_error = f"{type(exc).__name__}: {exc}\n\nالتفاصيل التقنية:\n{traceback.format_exc()}"
            self.finished_err.emit(self.job, detailed_error)


class EmotionDatasetBuilderApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Multimodal Emotion Dataset Builder (Video / Audio / Text)")
        self.resize(1220, 840)

        self.video_path: Optional[str] = None
        self.duration_ms: int = 0
        self.start_ms: int = 0
        self.end_ms: int = 0
        self.is_playing_range: bool = False
        self.is_slider_pressed: bool = False
        self.worker: Optional[ExtractionWorker] = None

        # مجلد الحفظ الافتراضي بجانب السكربت
        self.dataset_dir = Path.cwd() / "dataset"

        # إعداد مشغل الوسائط الأصلي (QMediaPlayer + QAudioOutput)
        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(1.0)
        self.media_player.setAudioOutput(self.audio_output)

        self._setup_ui()
        self._connect_signals()
        self._refresh_next_segment_label()

    def _setup_ui(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("ملف (File)")

        open_action = QAction("فتح فيديو (Open Video)...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_video_dialog)
        file_menu.addAction(open_action)

        folder_action = QAction("تغيير مجلد الحفظ (Dataset Folder)...", self)
        folder_action.triggered.connect(self.choose_dataset_folder)
        file_menu.addAction(folder_action)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # ================= القسم الأيسر: الفيديو والـ Timeline =================
        left_panel = QVBoxLayout()
        left_panel.setSpacing(8)

        top_bar = QHBoxLayout()
        self.btn_open_video = QPushButton("📂 فتح فيديو")
        self.btn_open_video.setMinimumHeight(34)
        self.lbl_video_name = QLabel("لم يتم تحميل أي فيديو بعد")
        self.lbl_video_name.setStyleSheet("font-weight: bold; color: #333;")
        self.btn_dataset_dir = QPushButton("📁 مجلد الحفظ")
        self.btn_dataset_dir.setMinimumHeight(34)
        top_bar.addWidget(self.btn_open_video)
        top_bar.addWidget(self.lbl_video_name, 1)
        top_bar.addWidget(self.btn_dataset_dir)
        left_panel.addLayout(top_bar)

        # شاشة عرض الفيديو
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(640, 380)
        self.video_widget.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.video_widget.setStyleSheet("background-color: #000000; border-radius: 6px;")
        self.media_player.setVideoOutput(self.video_widget)
        left_panel.addWidget(self.video_widget, 1)

        # شريط Timeline كامل
        timeline_group = QGroupBox("Timeline & Playback")
        timeline_layout = QVBoxLayout(timeline_group)

        slider_row = QHBoxLayout()
        self.lbl_current_time = QLabel("0.00.00")
        self.lbl_current_time.setStyleSheet("font-family: Consolas, monospace; font-size: 13px; font-weight: bold;")
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 0)
        self.lbl_total_time = QLabel("0.00.00")
        self.lbl_total_time.setStyleSheet("font-family: Consolas, monospace; font-size: 13px;")
        slider_row.addWidget(self.lbl_current_time)
        slider_row.addWidget(self.timeline_slider, 1)
        slider_row.addWidget(self.lbl_total_time)
        timeline_layout.addLayout(slider_row)

        controls_row = QHBoxLayout()
        self.btn_play_pause = QPushButton("▶ تشغيل / إيقاف مؤقت")
        self.btn_play_pause.setMinimumHeight(34)

        self.btn_step_back = QPushButton("⏪ -0.5s")
        self.btn_step_forward = QPushButton("⏩ +0.5s")

        self.lbl_volume = QLabel("الصوت:")
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setMaximumWidth(100)

        controls_row.addWidget(self.btn_play_pause)
        controls_row.addWidget(self.btn_step_back)
        controls_row.addWidget(self.btn_step_forward)
        controls_row.addStretch()
        controls_row.addWidget(self.lbl_volume)
        controls_row.addWidget(self.volume_slider)
        timeline_layout.addLayout(controls_row)

        left_panel.addWidget(timeline_group)

        # صندوق تحديد الـ Segment
        segment_group = QGroupBox("تحديد المقطع الزمني (Segment Selection - دقائق.ثواني.أجزاء)")
        seg_grid = QGridLayout(segment_group)

        self.btn_mark_start = QPushButton("📍 Mark Start (الآن)")
        self.btn_mark_start.setMinimumHeight(32)
        self.txt_start_time = QLineEdit("0.00.00")
        self.txt_start_time.setPlaceholderText("مثال: 0.45.02")
        self.txt_start_time.setStyleSheet("font-family: Consolas, monospace; font-size: 13px;")

        self.btn_mark_end = QPushButton("🏁 Mark End (الآن)")
        self.btn_mark_end.setMinimumHeight(32)
        self.txt_end_time = QLineEdit("0.00.00")
        self.txt_end_time.setPlaceholderText("مثال: 1.20.25")
        self.txt_end_time.setStyleSheet("font-family: Consolas, monospace; font-size: 13px;")

        self.btn_play_range = QPushButton("🔁 Play Selected Range (مراجعة الجزء)")
        self.btn_play_range.setMinimumHeight(36)
        self.btn_play_range.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold;")

        self.lbl_seg_duration = QLabel("مدة المقطع: 0.00 ثانية")
        self.lbl_seg_duration.setStyleSheet("font-weight: bold; color: #2E7D32;")

        seg_grid.addWidget(self.btn_mark_start, 0, 0)
        seg_grid.addWidget(QLabel("وقت البداية:"), 0, 1)
        seg_grid.addWidget(self.txt_start_time, 0, 2)

        seg_grid.addWidget(self.btn_mark_end, 1, 0)
        seg_grid.addWidget(QLabel("وقت النهاية:"), 1, 1)
        seg_grid.addWidget(self.txt_end_time, 1, 2)

        seg_grid.addWidget(self.btn_play_range, 2, 0, 1, 2)
        seg_grid.addWidget(self.lbl_seg_duration, 2, 2)

        left_panel.addWidget(segment_group)
        main_layout.addLayout(left_panel, 3)

        # ================= القسم الأيمن: تصنيف المشاعر والبيانات الوصفية =================
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        header_box = QGroupBox("حالة الحفظ")
        header_layout = QVBoxLayout(header_box)
        self.lbl_next_segment = QLabel("Segment القادم: segment_0001")
        self.lbl_next_segment.setStyleSheet("font-size: 14px; font-weight: bold; color: #D84315;")
        self.lbl_dataset_path = QLabel(f"المسار: {self.dataset_dir}")
        self.lbl_dataset_path.setWordWrap(True)
        self.lbl_dataset_path.setStyleSheet("font-size: 11px; color: #555;")
        header_layout.addWidget(self.lbl_next_segment)
        header_layout.addWidget(self.lbl_dataset_path)
        right_panel.addWidget(header_box)

        emotions_group = QGroupBox("تصنيف المشاعر المستقل (Multimodal Emotions)")
        emo_form = QFormLayout(emotions_group)

        self.cmb_visual_emotion = QComboBox()
        self.cmb_visual_emotion.addItems(EMOTIONS)

        self.cmb_audio_emotion = QComboBox()
        self.cmb_audio_emotion.addItems(EMOTIONS)

        self.cmb_text_emotion = QComboBox()
        self.cmb_text_emotion.addItems(EMOTIONS)

        emo_form.addRow("Visual Emotion (البصري):", self.cmb_visual_emotion)
        emo_form.addRow("Audio Emotion (الصوتي):", self.cmb_audio_emotion)
        emo_form.addRow("Text Emotion (النصي):", self.cmb_text_emotion)
        right_panel.addWidget(emotions_group)

        meta_group = QGroupBox("التقييم وبيانات المُعلّق (Metadata)")
        meta_form = QFormLayout(meta_group)

        self.spin_intensity = QSpinBox()
        self.spin_intensity.setRange(1, 5)
        self.spin_intensity.setValue(3)

        self.spin_confidence = QSpinBox()
        self.spin_confidence.setRange(1, 5)
        self.spin_confidence.setValue(5)

        self.txt_annotator_id = QLineEdit("annotator_01")
        self.txt_annotator_id.setPlaceholderText("أدخل معرف الشخص (مثال: ann_01)")

        meta_form.addRow("Intensity (1 - 5):", self.spin_intensity)
        meta_form.addRow("Confidence (1 - 5):", self.spin_confidence)
        meta_form.addRow("Annotator ID:", self.txt_annotator_id)
        right_panel.addWidget(meta_group)

        text_group = QGroupBox("النص والملاحظات")
        text_layout = QVBoxLayout(text_group)

        text_layout.addWidget(QLabel("Transcript (النص المنطوق يدويًا):"))
        self.txt_transcript = QTextEdit()
        self.txt_transcript.setPlaceholderText("اكتب النص المنطوق في هذا المقطع هنا...")
        self.txt_transcript.setMinimumHeight(90)
        text_layout.addWidget(self.txt_transcript)

        text_layout.addWidget(QLabel("Notes (ملاحظات إضافية):"))
        self.txt_notes = QLineEdit()
        self.txt_notes.setPlaceholderText("ملاحظات حول نبرة الصوت، تعبير الوجه، إلخ...")
        text_layout.addWidget(self.txt_notes)

        right_panel.addWidget(text_group, 1)

        self.btn_save_segment = QPushButton("💾 Save Segment (MP4 + WAV + TXT + CSV)")
        self.btn_save_segment.setMinimumHeight(46)
        self.btn_save_segment.setStyleSheet(
            "background-color: #2E7D32; color: white; font-size: 14px; font-weight: bold; border-radius: 6px;"
        )
        right_panel.addWidget(self.btn_save_segment)

        main_layout.addLayout(right_panel, 2)

        self.range_timer = QTimer(self)
        self.range_timer.setInterval(25)
        self.range_timer.timeout.connect(self._check_range_playback)

    def _connect_signals(self) -> None:
        self.btn_open_video.clicked.connect(self.open_video_dialog)
        self.btn_dataset_dir.clicked.connect(self.choose_dataset_folder)
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.btn_step_back.clicked.connect(lambda: self.seek_relative(-500))
        self.btn_step_forward.clicked.connect(lambda: self.seek_relative(500))

        self.volume_slider.valueChanged.connect(lambda v: self.audio_output.setVolume(v / 100.0))

        self.media_player.positionChanged.connect(self._on_position_changed)
        self.media_player.durationChanged.connect(self._on_duration_changed)
        self.media_player.errorOccurred.connect(self._on_player_error)

        self.timeline_slider.sliderPressed.connect(self._on_slider_pressed)
        self.timeline_slider.sliderReleased.connect(self._on_slider_released)
        self.timeline_slider.sliderMoved.connect(self._on_slider_moved)

        self.btn_mark_start.clicked.connect(self.mark_start_now)
        self.btn_mark_end.clicked.connect(self.mark_end_now)
        self.txt_start_time.editingFinished.connect(self._sync_times_from_inputs)
        self.txt_end_time.editingFinished.connect(self._sync_times_from_inputs)

        self.btn_play_range.clicked.connect(self.play_selected_range)
        self.btn_save_segment.clicked.connect(self.save_segment)

    def open_video_dialog(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "اختر ملف فيديو",
            "",
            "Video Files (*.mp4 *.mkv *.avi *.mov *.webm *.flv);;All Files (*.*)",
        )
        if file_path:
            self.load_video(file_path)

    def load_video(self, file_path: str) -> None:
        if not os.path.isfile(file_path):
            QMessageBox.critical(self, "خطأ", f"الملف غير موجود:\n{file_path}")
            return

        self.video_path = file_path
        self.lbl_video_name.setText(os.path.basename(file_path))
        self.is_playing_range = False
        self.range_timer.stop()

        self.media_player.stop()
        self.media_player.setSource(QUrl.fromLocalFile(file_path))
        self.audio_output.setVolume(self.volume_slider.value() / 100.0)
        self.media_player.play()

    def choose_dataset_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "اختر مجلد حفظ الـ Dataset", str(self.dataset_dir)
        )
        if folder:
            self.dataset_dir = Path(folder)
            self.lbl_dataset_path.setText(f"المسار: {self.dataset_dir}")
            self._refresh_next_segment_label()

    def _get_next_segment_index(self) -> int:
        video_dir = self.dataset_dir / "video"
        audio_dir = self.dataset_dir / "audio"
        text_dir = self.dataset_dir / "text"

        max_idx = 0
        pattern = re.compile(r"segment_(\d+)\.")

        for folder in (video_dir, audio_dir, text_dir):
            if folder.exists():
                for item in folder.iterdir():
                    m = pattern.match(item.name)
                    if m:
                        max_idx = max(max_idx, int(m.group(1)))

        csv_file = self.dataset_dir / "annotations.csv"
        if csv_file.exists():
            try:
                with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        seg_id = row.get("segment_id", "")
                        m = re.match(r"segment_(\d+)", seg_id)
                        if m:
                            max_idx = max(max_idx, int(m.group(1)))
            except Exception:
                pass

        return max_idx + 1

    def _refresh_next_segment_label(self) -> None:
        next_idx = self._get_next_segment_index()
        self.lbl_next_segment.setText(f"Segment القادم: segment_{next_idx:04d}")

    def toggle_play_pause(self) -> None:
        self.is_playing_range = False
        self.range_timer.stop()
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def seek_relative(self, delta_ms: int) -> None:
        if self.duration_ms <= 0:
            return
        new_pos = max(0, min(self.duration_ms, self.media_player.position() + delta_ms))
        self.media_player.setPosition(new_pos)

    def _on_duration_changed(self, duration: int) -> None:
        self.duration_ms = duration
        self.timeline_slider.setRange(0, duration)
        self.lbl_total_time.setText(format_time_m_ss_cc(duration))

    def _on_position_changed(self, position: int) -> None:
        self.lbl_current_time.setText(format_time_m_ss_cc(position))
        if not self.is_slider_pressed:
            self.timeline_slider.setValue(position)

    def _on_slider_pressed(self) -> None:
        self.is_slider_pressed = True

    def _on_slider_released(self) -> None:
        self.is_slider_pressed = False
        self.media_player.setPosition(self.timeline_slider.value())

    def _on_slider_moved(self, position: int) -> None:
        self.lbl_current_time.setText(format_time_m_ss_cc(position))

    def _on_player_error(self, error, error_string: str) -> None:
        QMessageBox.critical(
            self,
            "خطأ في تشغيل الفيديو",
            f"تعذر تشغيل الفيديو داخل المشغل:\n{error_string}",
        )

    def mark_start_now(self) -> None:
        pos = self.media_player.position()
        self.start_ms = pos
        self.txt_start_time.setText(format_time_m_ss_cc(pos))
        if self.end_ms < self.start_ms:
            self.end_ms = self.start_ms
            self.txt_end_time.setText(format_time_m_ss_cc(self.end_ms))
        self._update_duration_label()

    def mark_end_now(self) -> None:
        pos = self.media_player.position()
        self.end_ms = pos
        self.txt_end_time.setText(format_time_m_ss_cc(pos))
        self._update_duration_label()

    def _sync_times_from_inputs(self) -> Tuple[int, int]:
        start_ms = parse_time_m_ss_cc(self.txt_start_time.text())
        end_ms = parse_time_m_ss_cc(self.txt_end_time.text())
        if self.duration_ms > 0:
            if start_ms > self.duration_ms:
                raise ValueError("وقت البداية يتجاوز مدة الفيديو الكلية.")
            if end_ms > self.duration_ms:
                end_ms = self.duration_ms
                self.txt_end_time.setText(format_time_m_ss_cc(end_ms))
        self.start_ms = start_ms
        self.end_ms = end_ms
        self._update_duration_label()
        return start_ms, end_ms

    def _update_duration_label(self) -> None:
        diff_sec = max(0.0, (self.end_ms - self.start_ms) / 1000.0)
        self.lbl_seg_duration.setText(f"مدة المقطع: {diff_sec:.2f} ثانية")

    def play_selected_range(self) -> None:
        if not self.video_path:
            QMessageBox.warning(self, "تنبيه", "يرجى فتح ملف فيديو أولاً.")
            return
        try:
            start_ms, end_ms = self._sync_times_from_inputs()
            if end_ms <= start_ms:
                raise ValueError("وقت النهاية يجب أن يكون أكبر من وقت البداية.")
        except ValueError as e:
            QMessageBox.warning(self, "خطأ في التوقيت", str(e))
            return

        self.media_player.setPosition(start_ms)
        self.is_playing_range = True
        self.range_timer.start()
        self.media_player.play()

    def _check_range_playback(self) -> None:
        if not self.is_playing_range:
            self.range_timer.stop()
            return
        if self.media_player.position() >= self.end_ms:
            self.media_player.pause()
            self.is_playing_range = False
            self.range_timer.stop()

    def save_segment(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        if not self.video_path:
            QMessageBox.warning(self, "تنبيه", "يرجى فتح ملف فيديو أولاً قبل الحفظ.")
            return

        try:
            start_ms, end_ms = self._sync_times_from_inputs()
            if end_ms <= start_ms:
                raise ValueError("وقت النهاية يجب أن يكون أكبر من وقت البداية.")
        except ValueError as e:
            QMessageBox.warning(self, "خطأ في التوقيت", str(e))
            return

        transcript = self.txt_transcript.toPlainText().strip()
        if not transcript:
            reply = QMessageBox.question(
                self,
                "حقل النص فارغ",
                "لم تقم بكتابة أي نص في حقل Transcript.\nهل تريد المتابعة وحفظ ملف TXT فارغ؟",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self.txt_transcript.setFocus()
                return

        annotator_id = self.txt_annotator_id.text().strip()
        if not annotator_id:
            QMessageBox.warning(self, "بيانات ناقصة", "يرجى إدخال Annotator ID.")
            return

        video_dir = self.dataset_dir / "video"
        audio_dir = self.dataset_dir / "audio"
        text_dir = self.dataset_dir / "text"
        video_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)
        text_dir.mkdir(parents=True, exist_ok=True)

        seg_idx = self._get_next_segment_index()
        seg_name = f"segment_{seg_idx:04d}"

        out_video_path = video_dir / f"{seg_name}.mp4"
        out_audio_path = audio_dir / f"{seg_name}.wav"
        out_text_path = text_dir / f"{seg_name}.txt"
        csv_path = self.dataset_dir / "annotations.csv"

        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0
        duration_sec = round(end_sec - start_sec, 3)

        job: Dict[str, Any] = {
            "source_path": self.video_path,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "out_video_path": out_video_path,
            "out_audio_path": out_audio_path,
            "out_text_path": out_text_path,
            "csv_path": csv_path,
            "seg_name": seg_name,
            "transcript": transcript,
            "csv_row": {
                "segment_id": seg_name,
                "source_video": os.path.basename(self.video_path),
                "start_time_str": format_time_m_ss_cc(start_ms),
                "end_time_str": format_time_m_ss_cc(end_ms),
                "start_sec": f"{start_sec:.3f}",
                "end_sec": f"{end_sec:.3f}",
                "duration_sec": f"{duration_sec:.3f}",
                "visual_emotion": self.cmb_visual_emotion.currentText(),
                "audio_emotion": self.cmb_audio_emotion.currentText(),
                "text_emotion": self.cmb_text_emotion.currentText(),
                "intensity": self.spin_intensity.value(),
                "confidence": self.spin_confidence.value(),
                "annotator_id": annotator_id,
                "transcript": transcript,
                "notes": self.txt_notes.text().strip(),
                "video_path": str(out_video_path.relative_to(self.dataset_dir)),
                "audio_path": str(out_audio_path.relative_to(self.dataset_dir)),
                "text_path": str(out_text_path.relative_to(self.dataset_dir)),
            },
        }

        self.btn_save_segment.setEnabled(False)
        self.btn_save_segment.setText("⏳ جاري استخراج الفيديو والصوت...")

        self.worker = ExtractionWorker(job)
        self.worker.finished_ok.connect(self._on_extraction_success)
        self.worker.finished_err.connect(self._on_extraction_error)
        self.worker.start()

    def _on_extraction_success(self, job: Dict[str, Any]) -> None:
        out_video_path: Path = job["out_video_path"]
        out_audio_path: Path = job["out_audio_path"]
        out_text_path: Path = job["out_text_path"]
        csv_path: Path = job["csv_path"]
        seg_name: str = job["seg_name"]

        try:
            if not out_video_path.exists() or out_video_path.stat().st_size == 0:
                raise RuntimeError(f"فشل إنشاء ملف الفيديو أو حجمه صفر: {out_video_path}")
            if not out_audio_path.exists() or out_audio_path.stat().st_size <= 44:
                raise RuntimeError(f"فشل إنشاء ملف الصوت WAV أو حجمه فارغ: {out_audio_path}")

            out_text_path.write_text(job["transcript"], encoding="utf-8")

            file_exists = csv_path.exists() and csv_path.stat().st_size > 0
            with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(job["csv_row"])

            v_size_kb = out_video_path.stat().st_size / 1024.0
            a_size_kb = out_audio_path.stat().st_size / 1024.0

            QMessageBox.information(
                self,
                "تم الحفظ بنجاح",
                f"تم حفظ المقطع {seg_name} بنجاح:\n\n"
                f"• الفيديو: {out_video_path.name} ({v_size_kb:.1f} KB)\n"
                f"• الصوت: {out_audio_path.name} ({a_size_kb:.1f} KB)\n"
                f"• النص: {out_text_path.name}\n"
                f"• السجل: annotations.csv",
            )

            self.txt_transcript.clear()
            self.txt_notes.clear()
            self._refresh_next_segment_label()
        except Exception as exc:
            detailed_error = f"{type(exc).__name__}: {exc}\n\nالتفاصيل التقنية:\n{traceback.format_exc()}"
            self._on_extraction_error(job, detailed_error)
            return
        finally:
            self.btn_save_segment.setEnabled(True)
            self.btn_save_segment.setText("💾 Save Segment (MP4 + WAV + TXT + CSV)")

    def _on_extraction_error(self, job: Dict[str, Any], detailed_error: str) -> None:
        for key in ("out_video_path", "out_audio_path", "out_text_path"):
            partial_file: Path = job[key]
            if partial_file.exists():
                try:
                    partial_file.unlink()
                except OSError:
                    pass

        self.btn_save_segment.setEnabled(True)
        self.btn_save_segment.setText("💾 Save Segment (MP4 + WAV + TXT + CSV)")
        QMessageBox.critical(
            self,
            "فشل استخراج وحفظ المقطع",
            detailed_error,
        )


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = EmotionDatasetBuilderApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
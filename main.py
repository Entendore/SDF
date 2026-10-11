import sys
import subprocess
import shutil
import os
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QComboBox, QFileDialog, 
                              QProgressBar, QLabel, QSpinBox, QSlider, QGroupBox,
                              QMessageBox, QFormLayout, QCheckBox, QColorDialog)
from PySide6.QtOpenGL import QOpenGLWidget, QOpenGLShaderProgram, QOpenGLShader, QOpenGLFramebufferObject
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QSurfaceFormat, QPainter, QColor, QPen, QKeyEvent

# --- Shaders ---
VERTEX_SHADER = """
attribute vec3 position;
attribute vec2 texCoord;
varying vec2 vTexCoord;

void main() {
    gl_Position = vec4(position, 1.0);
    vTexCoord = texCoord;
}
"""

FRAGMENT_SHADER = """
precision highp float;
varying vec2 vTexCoord;
uniform float time;
uniform float aspect;
uniform float u_scale;
uniform float u_intensity;
uniform float u_audio;
uniform float u_glow;
uniform float u_zoom;
uniform float u_orbit_speed;
uniform int u_scene;
uniform int u_palette;
uniform bool u_vignette;
uniform vec3 u_bg_col;

// SDF Functions
float sdSphere(vec3 p, float r) { return length(p) - r; }
float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}
float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}
float sdPlane(vec3 p, vec2 h) {
    vec2 w = vec2(h.x, h.y);
    vec2 q = vec2(p.z, p.y);
    vec2 d = abs(q) - w;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

float smin(float a, float b, float k) {
    float res = exp(-k * a) + exp(-k * b);
    return -log(res) / k;
}

// Scene definition
float map(vec3 p) {
    float d = 1000.0;
    float audio_pulse = 1.0 + u_audio * 0.5;
    
    if(u_scene == 0) {
        vec3 p1 = p + vec3(sin(time)*0.5, cos(time*0.8)*0.5, 0.0);
        vec3 p2 = p + vec3(cos(time*1.2)*0.5, sin(time*0.7)*0.5, sin(time)*0.5);
        d = smin(sdSphere(p1, 1.0 * u_scale * audio_pulse), sdSphere(p2, 0.8 * u_scale * audio_pulse), 2.0);
    } 
    else if(u_scene == 1) {
        vec3 q = p;
        q.xy *= rot(time * 0.3 + u_audio * 2.0);
        q.yz *= rot(time * 0.2);
        vec3 c = vec2(2.0, 2.0).xxy;
        vec3 q1 = mod(q, c) - 0.5 * c;
        d = sdBox(q1, vec3(0.5 * u_scale));
    } 
    else if(u_scene == 2) {
        vec3 q = p;
        q.xy *= rot(time * 0.5 + length(q) + u_audio * 3.0);
        q.yz *= rot(time * 0.3);
        d = sdTorus(q, vec2(1.0 * u_scale, 0.3 * u_scale * audio_pulse));
    }
    else if(u_scene == 3) {
        vec3 q = p;
        float s = 1.0 + u_audio * 0.4;
        for(int i = 0; i < 4; i++) {
            q = abs(q) - 0.8 * s * u_scale;
            q.xy *= rot(0.5 + u_audio * 0.3);
            q.yz *= rot(0.3 + time * 0.1);
        }
        d = sdBox(q, vec3(0.5)) / pow(s, 3.0);
    }
    else if(u_scene == 4) {
        float h = sin(p.x * 2.0 + time) * 0.2 * u_audio + cos(p.z * 1.5) * 0.2 * u_audio;
        d = sdPlane(p + vec3(0.0, h, 0.0), vec2(10.0, 10.0)) - 0.1;
        vec3 q = p - vec3(0.0, 1.0 + u_audio * 2.0, 0.0);
        d = smin(d, sdSphere(q, 0.5 * u_scale), 1.0);
    }
    return d;
}

vec3 getNormal(vec3 p) {
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
    ));
}

void main() {
    vec2 uv = (vTexCoord - 0.5);
    uv.x *= aspect;

    // Camera Controls
    vec3 ro = vec3(0.0, 0.0, -u_zoom);
    vec3 rd = normalize(vec3(uv, 1.0));
    
    // Auto-Orbit
    rd.xy *= rot(time * u_orbit_speed);
    ro.xy *= rot(time * u_orbit_speed);

    float t = 0.0;
    float d = 0.0;
    for(int i = 0; i < 100; i++) {
        vec3 p = ro + rd * t;
        d = map(p);
        t += d;
        if(d < 0.001 || t > 20.0) break;
    }

    vec3 col = u_bg_col;
    if(t < 20.0) {
        vec3 p = ro + rd * t;
        vec3 n = getNormal(p);
        vec3 light = normalize(vec3(1.0, 1.0, -1.0));
        
        float diff = max(dot(n, light), 0.0);
        
        // Color Palettes
        vec3 baseColor = vec3(1.0);
        if(u_palette == 0) // Rainbow
            baseColor = 0.5 + 0.5 * cos(time + p.xyz * 2.0 + vec3(0.0, 2.0, 4.0));
        else if(u_palette == 1) // Cyberpunk (Pink/Cyan)
            baseColor = 0.5 + 0.5 * cos(3.14 + p.xyz * 1.5 + vec3(0.0, 0.5, 1.0));
        else if(u_palette == 2) // Sunset (Orange/Purple)
            baseColor = 0.5 + 0.5 * cos(1.5 + p.yzx * 1.0 + vec3(2.0, 4.0, 6.0));
        else if(u_palette == 3) // Monochrome
            baseColor = vec3(diff);
            
        col = baseColor * diff * u_intensity;
        col += baseColor * u_glow * (0.2 + u_audio * 0.8); // Audio Reactive Glow
    }

    // Post-processing: Vignette
    if(u_vignette) {
        float vig = smoothstep(1.2, 0.3, length(uv));
        col *= vig;
    }

    gl_FragColor = vec4(col, 1.0);
}
"""

class AudioDecoderThread(QThread):
    finished = Signal(np.ndarray, int)
    error = Signal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            cmd = [
                'ffmpeg', '-i', self.file_path,
                '-f', 'f32le', '-acodec', 'pcm_f32le',
                '-ac', '1', '-ar', '44100', '-'
            ]
            result = subprocess.run(cmd, capture_output=True, check=True)
            audio_data = np.frombuffer(result.stdout, dtype=np.float32)
            self.finished.emit(audio_data, 44100)
        except Exception as e:
            self.error.emit(str(e))


class SDFRenderer(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.time = 0.0
        self.is_playing = True
        self.shader_program = None
        
        # Uniforms
        self.scene_id = 0
        self.scale = 1.0
        self.intensity = 1.0
        self.audio_level = 0.0
        self.glow = 1.0
        self.zoom = 3.5
        self.orbit_speed = 0.0
        self.palette = 0
        self.vignette = True
        self.bg_color = (0.0, 0.0, 0.0)
        
        self.show_safe_area = False

    def initializeGL(self):
        self.gl().glClearColor(0.0, 0.0, 0.0, 1.0)
        self.shader_program = QOpenGLShaderProgram()
        self.shader_program.addShaderFromSourceCode(QOpenGLShader.Vertex, VERTEX_SHADER)
        self.shader_program.addShaderFromSourceCode(QOpenGLShader.Fragment, FRAGMENT_SHADER)
        self.shader_program.link()
        self.pos_attr = self.shader_program.attributeLocation("position")
        self.tex_attr = self.shader_program.attributeLocation("texCoord")

    def paintGL(self):
        self.gl().glClear(self.gl().GL_COLOR_BUFFER_BIT)
        self.shader_program.bind()

        aspect = self.width() / max(1, self.height())
        
        self.shader_program.setUniformValue("time", self.time)
        self.shader_program.setUniformValue("aspect", aspect)
        self.shader_program.setUniformValue("u_scene", self.scene_id)
        self.shader_program.setUniformValue("u_scale", self.scale)
        self.shader_program.setUniformValue("u_intensity", self.intensity)
        self.shader_program.setUniformValue("u_audio", self.audio_level)
        self.shader_program.setUniformValue("u_glow", self.glow)
        self.shader_program.setUniformValue("u_zoom", self.zoom)
        self.shader_program.setUniformValue("u_orbit_speed", self.orbit_speed)
        self.shader_program.setUniformValue("u_palette", self.palette)
        self.shader_program.setUniformValue("u_vignette", self.vignette)
        self.shader_program.setUniformValue("u_bg_col", *self.bg_color)

        vertices = np.array([-1.0, -1.0, 0.0,  1.0, -1.0, 0.0, -1.0, 1.0, 0.0,  1.0, 1.0, 0.0], dtype=np.float32)
        tex_coords = np.array([0.0, 0.0,  1.0, 0.0,  0.0, 1.0,  1.0, 1.0], dtype=np.float32)

        self.shader_program.setAttributeArray(self.pos_attr, vertices)
        self.shader_program.setAttributeArray(self.tex_attr, tex_coords)
        self.shader_program.enableAttributeArray(self.pos_attr)
        self.shader_program.enableAttributeArray(self.tex_attr)
        self.gl().glDrawArrays(self.gl().GL_TRIANGLE_STRIP, 0, 4)

        self.shader_program.disableAttributeArray(self.pos_attr)
        self.shader_program.disableAttributeArray(self.tex_attr)
        self.shader_program.release()
        
        if self.show_safe_area:
            painter = QPainter(self)
            pen = QPen(QColor(255, 0, 0, 200), 2)
            painter.setPen(pen)
            w = self.width()
            h = self.height()
            target_w = h * (9.0 / 16.0)
            x1 = (w - target_w) / 2
            x2 = x1 + target_w
            painter.fillRect(0, 0, int(x1), h, QColor(0, 0, 0, 120))
            painter.fillRect(int(x2), 0, w - int(x2), h, QColor(0, 0, 0, 120))
            painter.drawLine(int(x1), 0, int(x1), h)
            painter.drawLine(int(x2), 0, int(x2), h)
            painter.end()

    def grab_high_res_frame(self, width, height, audio_level=0.0):
        self.makeCurrent()
        fbo = QOpenGLFramebufferObject(width, height)
        fbo.bind()
        
        self.gl().glViewport(0, 0, width, height)
        self.shader_program.bind()
        
        aspect = width / height
        self.shader_program.setUniformValue("time", self.time)
        self.shader_program.setUniformValue("aspect", aspect)
        self.shader_program.setUniformValue("u_scene", self.scene_id)
        self.shader_program.setUniformValue("u_scale", self.scale)
        self.shader_program.setUniformValue("u_intensity", self.intensity)
        self.shader_program.setUniformValue("u_audio", audio_level)
        self.shader_program.setUniformValue("u_glow", self.glow)
        self.shader_program.setUniformValue("u_zoom", self.zoom)
        self.shader_program.setUniformValue("u_orbit_speed", self.orbit_speed)
        self.shader_program.setUniformValue("u_palette", self.palette)
        self.shader_program.setUniformValue("u_vignette", self.vignette)
        self.shader_program.setUniformValue("u_bg_col", *self.bg_color)

        vertices = np.array([-1.0, -1.0, 0.0,  1.0, -1.0, 0.0, -1.0, 1.0, 0.0,  1.0, 1.0, 0.0], dtype=np.float32)
        tex_coords = np.array([0.0, 0.0,  1.0, 0.0,  0.0, 1.0,  1.0, 1.0], dtype=np.float32)

        self.shader_program.setAttributeArray(self.pos_attr, vertices)
        self.shader_program.setAttributeArray(self.tex_attr, tex_coords)
        self.shader_program.enableAttributeArray(self.pos_attr)
        self.shader_program.enableAttributeArray(self.tex_attr)
        self.gl().glDrawArrays(self.gl().GL_TRIANGLE_STRIP, 0, 4)

        self.shader_program.disableAttributeArray(self.pos_attr)
        self.shader_program.disableAttributeArray(self.tex_attr)
        self.shader_program.release()
        
        image = fbo.toImage()
        fbo.release()
        self.doneCurrent()
        
        image = image.convertToFormat(image.Format_RGB888)
        return image.constBits().asarray(image.sizeInBytes())


class VideoExporterThread(QThread):
    progress_updated = Signal(int, str)
    export_finished = Signal(bool, str)

    def __init__(self, renderer, width, height, fps, duration, output_file, preset, audio_data=None, sample_rate=44100, seamless=False, speed=1.0):
        super().__init__()
        self.renderer = renderer
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.output_file = output_file
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.seamless = seamless
        self.speed = speed
        
        presets = {
            "Fast": ["-preset", "fast", "-crf", "23"],
            "Standard": ["-preset", "medium", "-crf", "20"],
            "High Quality": ["-preset", "slow", "-crf", "18"]
        }
        self.ffmpeg_args = presets.get(preset, presets["Standard"])

    def get_audio_level(self, current_time):
        if self.audio_data is None:
            return 0.0
        sample_idx = int(current_time * self.sample_rate)
        window_size = int(self.sample_rate / self.fps)
        if sample_idx + window_size > len(self.audio_data):
            return 0.0
        chunk = self.audio_data[sample_idx:sample_idx + window_size]
        return float(np.sqrt(np.mean(chunk**2)))

    def run(self):
        total_frames = int(self.duration * self.fps)
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}', '-pix_fmt', 'rgb24',
            '-r', str(self.fps), '-i', '-'
        ]
        
        if self.audio_data is not None:
            import tempfile
            import soundfile as sf
            self.temp_audio_path = os.path.join(tempfile.gettempdir(), "sdf_studio_audio.wav")
            sf.write(self.temp_audio_path, self.audio_data, self.sample_rate)
            cmd.extend(['-i', self.temp_audio_path])

        cmd.extend([
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            *self.ffmpeg_args, '-movflags', '+faststart'
        ])
        
        if self.audio_data is not None:
            cmd.extend(['-c:a', 'aac', '-b:a', '192k', '-shortest'])

        cmd.append(self.output_file)

        try:
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            for i in range(total_frames):
                if self.seamless:
                    cycle_time = (i / self.fps) % 10.0
                    self.renderer.time = cycle_time if cycle_time < 5.0 else 10.0 - cycle_time
                else:
                    self.renderer.time = (i / self.fps) * self.speed
                
                current_time = i / self.fps
                audio_lvl = self.get_audio_level(current_time)
                
                frame_bytes = self.renderer.grab_high_res_frame(self.width, self.height, audio_level=audio_lvl)
                process.stdin.write(frame_bytes)
                
                progress = int((i / total_frames) * 100)
                self.progress_updated.emit(progress, f"Rendering frame {i+1}/{total_frames}...")

            process.stdin.close()
            process.wait()
            
            if process.returncode == 0:
                self.export_finished.emit(True, self.output_file)
            else:
                self.export_finished.emit(False, "FFmpeg encoding failed.")
            
            if self.audio_data is not None and os.path.exists(self.temp_audio_path):
                os.remove(self.temp_audio_path)

        except Exception as e:
            self.export_finished.emit(False, str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultimate SDF Studio - YouTube & Shorts Maker")
        self.resize(1200, 750)

        if not shutil.which("ffmpeg"):
            QMessageBox.critical(self, "FFmpeg Missing", 
                                 "FFmpeg was not found in your system PATH.\nPlease install it.")

        self.audio_data = None
        self.sample_rate = 44100
        self.smoothed_audio = 0.0
        self.audio_sensitivity = 1.0

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Top layout (Preview + Controls)
        top_layout = QHBoxLayout()
        layout.addLayout(top_layout, stretch=1)

        # Left: OpenGL Preview
        self.renderer = SDFRenderer()
        top_layout.addWidget(self.renderer, stretch=1)

        # Right: Controls
        controls = QWidget()
        controls.setFixedWidth(360)
        ctr_layout = QVBoxLayout(controls)
        top_layout.addWidget(controls)

        # --- Scene Controls ---
        scene_group = QGroupBox("Scene & Parameters")
        scene_layout = QFormLayout(scene_group)
        
        self.scene_combo = QComboBox()
        self.scene_combo.addItems(["Morphing Spheres", "Infinite Box Grid", "Twisted Torus", "Pulsing Fractal", "Audio Terrain"])
        self.scene_combo.currentIndexChanged.connect(lambda i: setattr(self.renderer, 'scene_id', i))
        scene_layout.addRow("Scene:", self.scene_combo)

        self.palette_combo = QComboBox()
        self.palette_combo.addItems(["Rainbow", "Cyberpunk", "Sunset", "Monochrome"])
        self.palette_combo.currentIndexChanged.connect(lambda i: setattr(self.renderer, 'palette', i))
        scene_layout.addRow("Color Palette:", self.palette_combo)

        self.play_btn = QPushButton("⏸ Pause Preview")
        self.play_btn.clicked.connect(self.toggle_play)
        scene_layout.addRow(self.play_btn)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 300)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(lambda v: self.update_param('speed', v / 100.0))
        scene_layout.addRow("Anim Speed:", self.speed_slider)

        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(10, 200)
        self.scale_slider.setValue(100)
        self.scale_slider.valueChanged.connect(lambda v: self.update_param('scale', v / 100.0))
        scene_layout.addRow("Shape Scale:", self.scale_slider)

        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(0, 200)
        self.intensity_slider.setValue(100)
        self.intensity_slider.valueChanged.connect(lambda v: self.update_param('intensity', v / 100.0))
        scene_layout.addRow("Color Intensity:", self.intensity_slider)
        
        ctr_layout.addWidget(scene_group)

        # --- Camera & Post Controls ---
        cam_group = QGroupBox("Camera & Post-Processing")
        cam_layout = QFormLayout(cam_group)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(15, 80) # 1.5 to 8.0
        self.zoom_slider.setValue(35)
        self.zoom_slider.valueChanged.connect(lambda v: self.update_param('zoom', v / 10.0))
        cam_layout.addRow("Zoom (Dist):", self.zoom_slider)

        self.orbit_slider = QSlider(Qt.Horizontal)
        self.orbit_slider.setRange(-100, 100)
        self.orbit_slider.setValue(0)
        self.orbit_slider.valueChanged.connect(lambda v: self.update_param('orbit', v / 100.0))
        cam_layout.addRow("Orbit Speed:", self.orbit_slider)

        self.glow_slider = QSlider(Qt.Horizontal)
        self.glow_slider.setRange(0, 300)
        self.glow_slider.setValue(100)
        self.glow_slider.valueChanged.connect(lambda v: self.update_param('glow', v / 100.0))
        cam_layout.addRow("Glow Strength:", self.glow_slider)

        self.vig_check = QCheckBox("Vignette Effect")
        self.vig_check.setChecked(True)
        self.vig_check.toggled.connect(lambda v: self.update_param('vignette', v))
        cam_layout.addRow(self.vig_check)

        self.bg_btn = QPushButton("Choose Background Color")
        self.bg_btn.clicked.connect(self.choose_bg_color)
        cam_layout.addRow(self.bg_btn)
        
        ctr_layout.addWidget(cam_group)

        # --- Audio Controls ---
        audio_group = QGroupBox("Audio (For Shorts Visualizers)")
        audio_layout = QVBoxLayout(audio_group)
        
        self.load_audio_btn = QPushButton("🎵 Load Audio File (MP3/WAV)")
        self.load_audio_btn.clicked.connect(self.load_audio)
        audio_layout.addWidget(self.load_audio_btn)
        
        self.audio_label = QLabel("No audio loaded")
        self.audio_label.setWordWrap(True)
        audio_layout.addWidget(self.audio_label)

        self.sens_slider = QSlider(Qt.Horizontal)
        self.sens_slider.setRange(0, 300)
        self.sens_slider.setValue(100)
        self.sens_slider.valueChanged.connect(lambda v: setattr(self, 'audio_sensitivity', v / 100.0))
        audio_layout.addWidget(QLabel("Audio Sensitivity:"))
        audio_layout.addWidget(self.sens_slider)

        self.audio_meter = QProgressBar()
        self.audio_meter.setRange(0, 100)
        self.audio_meter.setFormat("Audio Level: %p%")
        audio_layout.addWidget(self.audio_meter)
        
        ctr_layout.addWidget(audio_group)

        # --- Export Controls ---
        export_group = QGroupBox("Export Settings")
        export_layout = QFormLayout(export_group)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["YouTube (1920x1080)", "Shorts (1080x1920)"])
        self.format_combo.currentIndexChanged.connect(self.update_safe_area)
        export_layout.addRow("Format:", self.format_combo)

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Fast", "Standard", "High Quality"])
        export_layout.addRow("Quality:", self.quality_combo)

        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 600)
        self.duration_spin.setValue(15)
        export_layout.addRow("Duration (s):", self.duration_spin)

        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["24", "30", "60"])
        self.fps_combo.setCurrentText("30")
        export_layout.addRow("FPS:", self.fps_combo)
        
        self.loop_check = QCheckBox("Seamless Loop (10s cycle)")
        export_layout.addRow(self.loop_check)

        export_btns = QHBoxLayout()
        self.export_btn = QPushButton("🎬 Export to MP4")
        self.export_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        self.export_btn.clicked.connect(self.start_export)
        export_btns.addWidget(self.export_btn)

        self.thumb_btn = QPushButton("🖼 Thumbnail")
        self.thumb_btn.clicked.connect(self.save_thumbnail)
        export_btns.addWidget(self.thumb_btn)
        export_layout.addRow(export_btns)

        ctr_layout.addWidget(export_group)
        ctr_layout.addStretch()

        # --- Bottom Timeline ---
        timeline_layout = QHBoxLayout()
        timeline_layout.addWidget(QLabel("Timeline:"))
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setRange(0, 1000)
        self.timeline_slider.sliderPressed.connect(self.on_timeline_press)
        self.timeline_slider.sliderReleased.connect(self.on_timeline_release)
        self.timeline_slider.valueChanged.connect(self.on_timeline_change)
        timeline_layout.addWidget(self.timeline_slider)
        layout.addLayout(timeline_layout)

        # --- Progress & Status ---
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        
        self.status_label = QLabel("Ready. Press Space to Play/Pause.")
        layout.addWidget(self.status_label)

        # Timers
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_preview_time)
        self.timer.start(16)
        
        self.audio_timer = QTimer(self)
        self.audio_timer.timeout.connect(self.update_audio_level)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space:
            self.toggle_play()
        elif event.key() == Qt.Key_E and (event.modifiers() & Qt.ControlModifier):
            self.start_export()
        else:
            super().keyPressEvent(event)

    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Audio File", "", "Audio Files (*.mp3 *.wav *.ogg *.flac *.m4a)"
        )
        if file_path:
            self.audio_label.setText("Decoding audio...")
            self.decoder_thread = AudioDecoderThread(file_path)
            self.decoder_thread.finished.connect(self.on_audio_loaded)
            self.decoder_thread.error.connect(lambda e: QMessageBox.critical(self, "Audio Error", e))
            self.decoder_thread.start()

    def on_audio_loaded(self, audio_data, sample_rate):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.audio_label.setText("Audio loaded successfully")
        self.status_label.setText("Audio ready for export and preview.")
        self.audio_timer.start(33)

    def update_audio_level(self):
        if self.audio_data is None:
            return
            
        current_sample = int(self.renderer.time * self.sample_rate)
        window_size = 1024
        if current_sample + window_size < len(self.audio_data):
            chunk = self.audio_data[current_sample:current_sample + window_size]
            raw_lvl = float(np.sqrt(np.mean(chunk**2)))
            
            # Exponential Moving Average for smoothing
            self.smoothed_audio = 0.8 * self.smoothed_audio + 0.2 * raw_lvl
            final_lvl = min(1.0, self.smoothed_audio * self.audio_sensitivity)
            
            self.renderer.audio_level = final_lvl
            self.audio_meter.setValue(int(final_lvl * 100))
        else:
            self.renderer.audio_level = 0.0
            self.audio_meter.setValue(0)

    def toggle_play(self):
        self.renderer.is_playing = not self.renderer.is_playing
        self.play_btn.setText("▶ Play Preview" if not self.renderer.is_playing else "⏸ Pause Preview")

    def on_timeline_press(self):
        self.was_playing = self.renderer.is_playing
        self.renderer.is_playing = False

    def on_timeline_release(self):
        self.renderer.is_playing = self.was_playing

    def on_timeline_change(self, val):
        # Map 0-1000 to 0-60 seconds for scrubbing
        self.renderer.time = (val / 1000.0) * 60.0
        self.renderer.update()

    def update_preview_time(self):
        if self.renderer.is_playing:
            speed = self.speed_slider.value() / 100.0
            self.renderer.time += 0.016 * speed
            # Update timeline slider without triggering signal loop
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(int((self.renderer.time % 60.0) / 60.0 * 1000))
            self.timeline_slider.blockSignals(False)
            self.renderer.update()

    def update_param(self, param, value):
        if param == 'scale':
            self.renderer.scale = value
        elif param == 'intensity':
            self.renderer.intensity = value
        elif param == 'glow':
            self.renderer.glow = value
        elif param == 'zoom':
            self.renderer.zoom = value
        elif param == 'orbit':
            self.renderer.orbit_speed = value
        elif param == 'vignette':
            self.renderer.vignette = value
        self.renderer.update()

    def choose_bg_color(self):
        color = QColorDialog.getColor(Qt.black, self, "Choose Background Color")
        if color.isValid():
            self.renderer.bg_color = (color.redF(), color.greenF(), color.blueF())
            self.renderer.update()

    def update_safe_area(self):
        is_shorts = "Shorts" in self.format_combo.currentText()
        self.renderer.show_safe_area = is_shorts
        self.renderer.update()

    def save_thumbnail(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Thumbnail", "", "PNG Files (*.png)"
        )
        if file_path:
            self.status_label.setText("Generating thumbnail...")
            QApplication.processEvents()
            
            fmt_text = self.format_combo.currentText()
            width, height = 1920, 1080 if "1920x1080" in fmt_text else (1080, 1920)
            
            frame_bytes = self.renderer.grab_high_res_frame(width, height, self.renderer.audio_level)
            
            from PySide6.QtGui import QImage
            img = QImage(frame_bytes, width, height, width * 3, QImage.Format_RGB888)
            img.save(file_path, "PNG")
            self.status_label.setText(f"Thumbnail saved to {file_path}")

    def start_export(self):
        self.export_btn.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText("Starting export...")

        fmt_text = self.format_combo.currentText()
        if "1920x1080" in fmt_text:
            width, height = 1920, 1080
        else:
            width, height = 1080, 1920

        fps = int(self.fps_combo.currentText())
        duration = self.duration_spin.value()
        preset = self.quality_combo.currentText()
        seamless = self.loop_check.isChecked()
        speed = self.speed_slider.value() / 100.0

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save MP4 Video", "", "MP4 Files (*.mp4)"
        )

        if not file_path:
            self.export_btn.setEnabled(True)
            self.status_label.setText("Export canceled.")
            return

        self.renderer.is_playing = False
        self.play_btn.setText("▶ Play Preview")

        self.export_thread = VideoExporterThread(
            self.renderer, width, height, fps, duration, file_path, preset,
            self.audio_data, self.sample_rate, seamless, speed
        )
        self.export_thread.progress_updated.connect(self.on_export_progress)
        self.export_thread.export_finished.connect(self.on_export_done)
        self.export_thread.start()

    def on_export_progress(self, val, text):
        self.progress.setValue(val)
        self.status_label.setText(text)

    def on_export_done(self, success, message):
        self.export_btn.setEnabled(True)
        if success:
            self.progress.setValue(100)
            self.status_label.setText(f"Success! Saved to {message}")
            QMessageBox.information(self, "Export Complete", f"Video successfully exported:\n{message}")
        else:
            self.status_label.setText(f"Error: {message}")
            QMessageBox.critical(self, "Export Failed", f"Failed to export video:\n{message}")


if __name__ == "__main__":
    try:
        import soundfile
    except ImportError:
        print("Installing required 'soundfile' package for audio export...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "soundfile"])
        import soundfile

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 2)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
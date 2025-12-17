import torch
import numpy as np
import math
from typing import List, Tuple, Dict, Optional

# CORRECTED IMPORT: Looks inside the render_3d folder
from .render_3d.taichi_cylinder import render_whole

# ============================================================================
# SKELETON DEFINITIONS
# ============================================================================

JOINT_NAMES = [
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", 
    "l_elbow", "l_wrist", "r_hip", "r_knee", "r_ankle", "l_hip", 
    "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear",
]

LIMB_SEQ = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
]

JOINT_PARENTS = {
    0: 1, 1: -1, 2: 1, 3: 2, 4: 3, 5: 1, 6: 5, 7: 6, 8: 1, 9: 8, 10: 9,
    11: 1, 12: 11, 13: 12, 14: 0, 15: 0, 16: 14, 17: 15,
}

BONE_COLORS = [
    [1.0, 0.15, 0.15, 0.8], [0.15, 1.0, 1.0, 0.8], [1.0, 0.43, 0.15, 0.8],
    [1.0, 0.72, 0.15, 0.8], [0.15, 0.72, 1.0, 0.8], [0.15, 0.43, 1.0, 0.8],
    [0.75, 1.0, 0.15, 0.8], [0.15, 1.0, 0.15, 0.8], [0.15, 1.0, 0.43, 0.8],
    [0.15, 0.15, 1.0, 0.8], [0.43, 0.15, 1.0, 0.8], [0.72, 0.15, 1.0, 0.8],
    [0.65, 0.65, 0.65, 0.8], [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
    [1.0, 0.15, 0.72, 0.8], [0.32, 0.15, 1.0, 0.8],
]

# ============================================================================
# AUDIO FEATURE EXTRACTION
# ============================================================================

class SCAILAudioFeatureExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio input from LoadAudio or other audio node"}),
                "frame_count": ("INT", {"default": 81, "min": 1, "max": 10000, "tooltip": "Number of output frames. Should match your video length."}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "tooltip": "Frames per second. Match your target video FPS."}),
                "bass_range": ("STRING", {"default": "20-250", "tooltip": "Bass frequency range in Hz (low-high). Drives body motion."}),
                "mid_range": ("STRING", {"default": "250-2000", "tooltip": "Mid frequency range in Hz. Vocals/melody typically here."}),
                "treble_range": ("STRING", {"default": "2000-8000", "tooltip": "Treble frequency range in Hz. Hi-hats, cymbals, detail."}),
                "smoothing": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "tooltip": "Temporal smoothing. Higher = less jittery but less reactive."}),
            }
        }
    
    RETURN_TYPES = ("SCAIL_AUDIO_FEATURES",)
    RETURN_NAMES = ("audio_features",)
    FUNCTION = "extract"
    CATEGORY = "SCAIL-AudioReactive"
    
    def extract(self, audio, frame_count, fps, bass_range, mid_range, treble_range, smoothing):
        def parse_range(s):
            low, high = s.split("-")
            return int(low), int(high)
        
        bass_low, bass_high = parse_range(bass_range)
        mid_low, mid_high = parse_range(mid_range)
        treble_low, treble_high = parse_range(treble_range)
        
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        if waveform.dim() == 3: waveform = waveform[0]
        if waveform.shape[0] > 1: waveform = waveform.mean(dim=0)
        else: waveform = waveform[0]
        
        waveform = waveform.numpy()
        frame_duration = 1.0 / fps
        samples_per_frame = int(sample_rate * frame_duration)
        
        rms = np.zeros(frame_count)
        bass = np.zeros(frame_count)
        mid = np.zeros(frame_count)
        treble = np.zeros(frame_count)
        onsets = np.zeros(frame_count)
        
        for i in range(frame_count):
            start = int(i * samples_per_frame)
            end = min(start + samples_per_frame, len(waveform))
            if start >= len(waveform): break
            frame_audio = waveform[start:end]
            if len(frame_audio) == 0: continue
            
            rms[i] = np.sqrt(np.mean(frame_audio ** 2))
            
            if len(frame_audio) > 1:
                fft = np.fft.rfft(frame_audio)
                freqs = np.fft.rfftfreq(len(frame_audio), 1.0 / sample_rate)
                mag = np.abs(fft)
                
                bass[i] = np.mean(mag[(freqs >= bass_low) & (freqs < bass_high)]) if np.any((freqs >= bass_low) & (freqs < bass_high)) else 0
                mid[i] = np.mean(mag[(freqs >= mid_low) & (freqs < mid_high)]) if np.any((freqs >= mid_low) & (freqs < mid_high)) else 0
                treble[i] = np.mean(mag[(freqs >= treble_low) & (freqs < treble_high)]) if np.any((freqs >= treble_low) & (freqs < treble_high)) else 0
        
        rms_diff = np.diff(rms, prepend=rms[0])
        onsets = np.maximum(rms_diff, 0)
        
        def normalize(arr):
            mx, mn = arr.max(), arr.min()
            return (arr - mn) / (mx - mn) if mx > mn else np.zeros_like(arr)
        
        rms = normalize(rms)
        bass = normalize(bass)
        mid = normalize(mid)
        treble = normalize(treble)
        onsets = normalize(onsets)
        
        if smoothing > 0:
            alpha = 1.0 - smoothing
            for arr in [rms, bass, mid, treble]:
                for i in range(1, len(arr)):
                    arr[i] = alpha * arr[i] + (1 - alpha) * arr[i-1]
        
        return ({
            "rms": torch.from_numpy(rms).float(),
            "bass": torch.from_numpy(bass).float(),
            "mid": torch.from_numpy(mid).float(),
            "treble": torch.from_numpy(treble).float(),
            "onsets": torch.from_numpy(onsets).float(),
            "frame_count": frame_count,
            "fps": fps,
        },)

# ============================================================================
# BASE POSE GENERATION (PROCEDURAL)
# ============================================================================

class SCAILBasePoseGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_type": (["active_idle", "t_pose", "a_pose", "custom"], {"default": "active_idle", "tooltip": "Starting pose style. active_idle=relaxed dance-ready"}),
                "character_count": ("INT", {"default": 1, "min": 1, "max": 5, "tooltip": "Number of characters to generate"}),
                "spacing": ("FLOAT", {"default": 150.0, "min": 50.0, "max": 500.0, "tooltip": "Horizontal distance between characters"}),
                "height": ("FLOAT", {"default": 400.0, "tooltip": "Character height scaling. 400 is standard."}),
                "depth": ("FLOAT", {"default": 800.0, "tooltip": "Z-depth from camera. Higher = further away."}),
                "center_x": ("FLOAT", {"default": 0.0, "tooltip": "Horizontal offset from frame center"}),
                "center_y": ("FLOAT", {"default": 0.0, "tooltip": "Vertical offset from frame center"}),
            },
            "optional": {"custom_pose": ("SCAIL_POSE", {"tooltip": "Override with custom pose when pose_type='custom'"})}
        }
    
    RETURN_TYPES = ("SCAIL_POSE",)
    RETURN_NAMES = ("base_pose",)
    FUNCTION = "generate"
    CATEGORY = "SCAIL-AudioReactive"
    
    def generate(self, pose_type, character_count, spacing, height, depth, center_x, center_y, custom_pose=None):
        if pose_type == "custom" and custom_pose: return (custom_pose,)
        
        scale = height / 400.0
        
        base_joints_map = {
            "neck": [0, -120, 0], "nose": [0, -160, -20],
            "r_shoulder": [-60, -110, 0], "l_shoulder": [60, -110, 0],
            "r_hip": [-35, 40, 0], "l_hip": [35, 40, 0],
            "r_eye": [-15, -170, -25], "l_eye": [15, -170, -25],
            "r_ear": [-35, -160, 5], "l_ear": [35, -160, 5],
        }
        
        if pose_type == "t_pose":
            base_joints_map.update({
                "r_elbow": [-130, -110, 0], "r_wrist": [-200, -110, 0],
                "l_elbow": [130, -110, 0], "l_wrist": [200, -110, 0],
                "r_knee": [-40, 120, 0], "r_ankle": [-40, 200, 0],
                "l_knee": [40, 120, 0], "l_ankle": [40, 200, 0],
            })
        elif pose_type == "a_pose":
            base_joints_map.update({
                "r_elbow": [-110, -60, 0], "r_wrist": [-160, -10, 0],
                "l_elbow": [110, -60, 0], "l_wrist": [160, -10, 0],
                "r_knee": [-40, 120, 0], "r_ankle": [-40, 200, 0],
                "l_knee": [40, 120, 0], "l_ankle": [40, 200, 0],
            })
        else: # active_idle
            base_joints_map.update({
                "r_elbow": [-90, -30, 20], "r_wrist": [-100, 50, 40],
                "l_elbow": [90, -30, 20], "l_wrist": [100, 50, 40],
                "r_knee": [-45, 110, 30], "r_ankle": [-40, 195, 0],
                "l_knee": [45, 110, 30], "l_ankle": [40, 195, 0],
            })
            
        single_skeleton = np.zeros((18, 3), dtype=np.float32)
        for name, coords in base_joints_map.items():
            if name in JOINT_NAMES:
                single_skeleton[JOINT_NAMES.index(name)] = coords

        total_joints = []
        full_limb_seq = []
        full_colors = []
        
        total_width = (character_count - 1) * spacing
        start_x = -total_width / 2.0
        
        for i in range(character_count):
            char_joints = single_skeleton.copy()
            char_joints *= scale
            x_offset = start_x + (i * spacing)
            char_joints[:, 0] += center_x + x_offset
            char_joints[:, 1] += center_y
            char_joints[:, 2] += depth
            total_joints.append(char_joints)
            
            index_offset = i * 18
            for (s, e) in LIMB_SEQ:
                full_limb_seq.append((s + index_offset, e + index_offset))
            full_colors.extend(BONE_COLORS)

        final_joints = np.concatenate(total_joints, axis=0) # Shape (18*N, 3)

        return ({"joints": torch.from_numpy(final_joints).float(), 
                 "limb_seq": full_limb_seq, 
                 "bone_colors": full_colors,
                 "char_count": character_count},)

# ============================================================================
# BEAT DETECTION
# ============================================================================

class SCAILBeatDetector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio input for beat analysis"}),
                "frame_count": ("INT", {"default": 81, "min": 1, "max": 10000, "tooltip": "Number of frames. Should match your video length."}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120, "tooltip": "Frames per second. Match your target video FPS."}),
            }
        }
    
    RETURN_TYPES = ("SCAIL_BEAT_INFO",)
    RETURN_NAMES = ("beat_info",)
    FUNCTION = "detect"
    CATEGORY = "SCAIL-AudioReactive"
    
    def detect(self, audio, frame_count, fps):
        import librosa
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        if isinstance(waveform, torch.Tensor): waveform = waveform.numpy()
        if waveform.ndim > 1: waveform = waveform.mean(axis=0) if waveform.shape[0] <= 2 else waveform[0]
        waveform = waveform.flatten().astype(np.float32)
        
        duration = frame_count / fps
        tempo, beat_frames_audio = librosa.beat.beat_track(y=waveform, sr=sample_rate)
        beat_frames = (librosa.frames_to_time(beat_frames_audio, sr=sample_rate) * fps).astype(int)
        beat_frames = beat_frames[beat_frames < frame_count]
        
        onset_env = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
        onset_resampled = np.interp(np.linspace(0, duration, frame_count), librosa.times_like(onset_env, sr=sample_rate), onset_env)
        onset_resampled /= (onset_resampled.max() + 1e-8)
        
        rms = librosa.feature.rms(y=waveform, hop_length=512)[0]
        rms_resampled = np.interp(np.linspace(0, duration, frame_count), librosa.times_like(rms, sr=sample_rate), rms)
        rms_resampled /= (rms_resampled.max() + 1e-8)
        
        downbeats = beat_frames[::4] if len(beat_frames) >= 4 else beat_frames
        tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        
        print(f"[SCAIL] Beats: {len(beat_frames)}, Tempo: {tempo_val:.1f}")
        
        return ({
            "frame_count": frame_count, "fps": fps, "tempo": tempo_val,
            "beat_frames": beat_frames, "downbeat_frames": downbeats,
            "onset_envelope": torch.from_numpy(onset_resampled).float(),
            "energy": torch.from_numpy(rms_resampled).float(),
        },)

# ============================================================================
# MOTION SYSTEM (MATH)
# ============================================================================

def rotate_point(point, pivot, axis, angle):
    p = point - pivot
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:   # X (Pitch)
        rot = np.array([p[0], p[1]*c - p[2]*s, p[1]*s + p[2]*c])
    elif axis == 1: # Y (Yaw)
        rot = np.array([p[0]*c + p[2]*s, p[1], -p[0]*s + p[2]*c])
    else:           # Z (Roll)
        rot = np.array([p[0]*c - p[1]*s, p[0]*s + p[1]*c, p[2]])
    return rot + pivot

def apply_chain_rotation(joints, pivot_idx, chain, axis, angle):
    res = joints.copy()
    pivot = joints[pivot_idx]
    for idx in chain:
        if idx != pivot_idx:
            res[idx] = rotate_point(joints[idx], pivot, axis, angle)
    return res

# ============================================================================
# EASING FUNCTIONS
# ============================================================================

def ease_linear(t):
    """Linear interpolation"""
    return t

def ease_out(t):
    """Decelerate - fast start, slow end (good for hitting beats)"""
    return 1.0 - (1.0 - t) ** 2

def ease_in_out(t):
    """Smooth acceleration and deceleration"""
    return t * t * (3.0 - 2.0 * t)

def ease_bounce(t):
    """Bounce effect at the end"""
    if t < 0.5:
        return 2 * t * t
    else:
        t2 = t - 0.5
        return 0.5 + t2 * (1.0 - t2) * 4

def ease_elastic(t):
    """Overshoot and settle back"""
    if t == 0 or t == 1:
        return t
    p = 0.3
    s = p / 4.0
    t -= 1
    return -(math.pow(2, 10 * t) * math.sin((t - s) * (2 * math.pi) / p)) + 1

def ease_punch(t):
    """Sharp hit then quick settle - great for dance hits"""
    if t < 0.3:
        return (t / 0.3) ** 0.5  # Fast attack
    else:
        return 1.0 - (1.0 - t) * 0.15  # Subtle settle

EASING_FUNCTIONS = {
    "linear": ease_linear,
    "ease_out": ease_out,
    "ease_in_out": ease_in_out,
    "bounce": ease_bounce,
    "elastic": ease_elastic,
    "punch": ease_punch,
}

DANCE_POSES = {
    "neutral": {
        "description": "Active Idle - Ready Stance",
        "rotations": {
            "spine": {"forward": 5}, 
            "right_elbow": {"bend": 15}, "left_elbow": {"bend": 15},
            "right_knee": {"bend": 15}, "left_knee": {"bend": 15}, 
        }
    },
    "lean_left": {
        "rotations": {"spine": {"tilt": 35}, "head": {"tilt": -30}, "right_knee": {"bend": 10}, "left_knee": {"bend": 45}, "right_shoulder": {"raise": 20}},
        "translations": {"hips": {"x": -25, "y": 5}, "l_ankle": {"x": -25, "y": 0}} 
    },
    "lean_right": {
        "rotations": {"spine": {"tilt": -35}, "head": {"tilt": 30}, "left_knee": {"bend": 10}, "right_knee": {"bend": 45}, "left_shoulder": {"raise": 20}},
        "translations": {"hips": {"x": 25, "y": 5}, "r_ankle": {"x": 25, "y": 0}} 
    },
    "arms_up_open": {
        "rotations": {"spine": {"forward": -15}, "head": {"nod": -20}, "right_shoulder": {"raise": 140, "forward": 20}, "left_shoulder": {"raise": 140, "forward": 20}, "right_elbow": {"bend": 40}, "left_elbow": {"bend": 40}, "right_knee": {"bend": 5}, "left_knee": {"bend": 5}},
        "translations": {"hips": {"y": -10}} 
    },
    "power_crouch": {
        "rotations": {"spine": {"forward": 30}, "head": {"nod": -25}, "right_knee": {"bend": 45}, "left_knee": {"bend": 45}, "right_ankle": {"bend": 20}, "left_ankle": {"bend": 20}, "right_shoulder": {"forward": 30}, "left_shoulder": {"forward": 30}, "right_elbow": {"bend": 90, "twist": 45}, "left_elbow": {"bend": 90, "twist": -45}},
        "translations": {"hips": {"y": 35}} 
    },
    "pump_right_hard": {
        "rotations": {"spine": {"tilt": -15, "turn": -20}, "right_shoulder": {"raise": 160}, "right_elbow": {"bend": 120}, "left_elbow": {"bend": 20}, "right_knee": {"bend": 10}, "left_knee": {"bend": 30}},
        "translations": {"hips": {"x": 15, "y": -5}} 
    },
    "pump_left_hard": {
        "rotations": {"spine": {"tilt": 15, "turn": 20}, "left_shoulder": {"raise": 160}, "left_elbow": {"bend": 120}, "right_elbow": {"bend": 20}, "left_knee": {"bend": 10}, "right_knee": {"bend": 30}},
        "translations": {"hips": {"x": -15, "y": -5}} 
    },
    "arms_wide": {
        "rotations": {"spine": {"forward": -10}, "right_shoulder": {"raise": 90, "forward": -20}, "left_shoulder": {"raise": 90, "forward": -20}, "right_elbow": {"bend": 10}, "left_elbow": {"bend": 10}}
    },
    "step_right": {
        "rotations": {"spine": {"tilt": -10, "turn": -15}, "right_knee": {"bend": 40}, "left_knee": {"bend": 10}, "right_elbow": {"bend": 60}, "left_elbow": {"bend": 20}},
        "translations": {"hips": {"x": 20, "y": 5}, "r_ankle": {"x": 20, "z": 10}} 
    },
    "step_left": {
        "rotations": {"spine": {"tilt": 10, "turn": 15}, "left_knee": {"bend": 40}, "right_knee": {"bend": 10}, "left_elbow": {"bend": 60}, "right_elbow": {"bend": 20}},
        "translations": {"hips": {"x": -20, "y": 5}, "l_ankle": {"x": -20, "z": 10}} 
    },
    "slide_left": {
        "description": "Slide step left",
        "rotations": { "left_knee": {"bend": 30}, "right_knee": {"bend": 10} },
        "translations": { "hips": {"x": -20, "y": 5}, "l_ankle": {"x": -20, "z": 10} }
    },
    "slide_right": {
        "description": "Slide step right",
        "rotations": { "right_knee": {"bend": 30}, "left_knee": {"bend": 10} },
        "translations": { "hips": {"x": 20, "y": 5}, "r_ankle": {"x": 20, "z": 10} }
    },
    "lunge_right": { "rotations": { "right_knee": {"bend": 70}, "left_knee": {"bend": 20}, "right_ankle": {"bend": 30}, "spine": {"forward": 10} }, "translations": { "hips": {"x": 15, "y": 15, "z": 15}, "r_ankle": {"x": 15, "z": 15} } }, 
    "lunge_left": { "rotations": { "left_knee": {"bend": 70}, "right_knee": {"bend": 20}, "left_ankle": {"bend": 30}, "spine": {"forward": 10} }, "translations": { "hips": {"x": -15, "y": 15, "z": -15}, "l_ankle": {"x": -15, "z": 15} } },
    "running_man": { "rotations": { "right_knee": {"bend": 70}, "left_knee": {"bend": 20}, "right_shoulder": {"raise": 45, "forward": 40}, "left_shoulder": {"raise": -20, "forward": -20} }, "translations": { "hips": {"y": 10}, "r_ankle": {"y": 10, "z": -10} } },
    
    # EXPANDED LIBRARY (IMPORTED & PRESERVED)
    "arms_up": { "rotations": { "right_shoulder": {"raise": 120}, "left_shoulder": {"raise": 120}, "right_elbow": {"bend": -30}, "left_elbow": {"bend": -30} } },
    "arms_out": { "rotations": { "right_shoulder": {"raise": 90}, "left_shoulder": {"raise": 90} } },
    "arms_crossed_low": { "rotations": { "right_shoulder": {"raise": 30, "forward": 40}, "left_shoulder": {"raise": 30, "forward": 40}, "right_elbow": {"bend": 80}, "left_elbow": {"bend": 80} } },
    "dab": { "rotations": { "right_shoulder": {"raise": 120, "forward": 20}, "left_shoulder": {"raise": 45, "forward": 60}, "right_elbow": {"bend": 0}, "left_elbow": {"bend": 90}, "head": {"turn": -45, "nod": 30} } },
    "floss_right": { "rotations": { "right_shoulder": {"raise": 45, "forward": -60}, "left_shoulder": {"raise": 45, "forward": 60}, "right_elbow": {"bend": 0}, "left_elbow": {"bend": 0}, "spine": {"twist": 30} }, "translations": {"hips": {"x": -8}} },
    "floss_left": { "rotations": { "right_shoulder": {"raise": 45, "forward": 60}, "left_shoulder": {"raise": 45, "forward": -60}, "right_elbow": {"bend": 0}, "left_elbow": {"bend": 0}, "spine": {"twist": -30} }, "translations": {"hips": {"x": 8}} },
    "disco_point_up": { "rotations": { "right_shoulder": {"raise": 140, "forward": 20}, "right_elbow": {"bend": 0}, "left_shoulder": {"raise": -10}, "spine": {"tilt": -10, "forward": -5} } },
    "disco_point_down": { "rotations": { "right_shoulder": {"raise": 45, "forward": 45}, "right_elbow": {"bend": 0}, "left_shoulder": {"raise": 100}, "left_elbow": {"bend": -20} } },
    "finger_guns": { "rotations": { "right_shoulder": {"raise": 90, "forward": 30}, "left_shoulder": {"raise": 90, "forward": 30}, "right_elbow": {"bend": 45}, "left_elbow": {"bend": 45} } },
    "breakdance_freeze": { "rotations": { "right_shoulder": {"raise": 120, "forward": 70}, "left_shoulder": {"raise": 120, "forward": 70}, "right_elbow": {"bend": 90}, "left_elbow": {"bend": 90}, "spine": {"forward": 50} }, "translations": { "hips": {"y": 25} } },
    "pop_lock": { "rotations": { "right_shoulder": {"raise": 90, "forward": 60}, "left_shoulder": {"raise": 90, "forward": -60}, "right_elbow": {"bend": 90}, "left_elbow": {"bend": 90}, "spine": {"twist": 30} } },
    "headbang_forward": { "rotations": { "head": {"nod": 60}, "spine": {"forward": 30}, "right_shoulder": {"raise": -10}, "left_shoulder": {"raise": -10} } },
    "headbang_back": { "rotations": { "head": {"nod": -40}, "spine": {"forward": -10} } },
    "rock_out": { "rotations": { "right_shoulder": {"raise": 120}, "left_shoulder": {"raise": 120}, "right_elbow": {"bend": 60}, "left_elbow": {"bend": 60} } },
    "air_guitar": { "rotations": { "right_shoulder": {"raise": 45, "forward": 50}, "left_shoulder": {"raise": 70, "forward": 40}, "right_elbow": {"bend": 90}, "left_elbow": {"bend": 60}, "spine": {"twist": 15} } },
    "gangnam_style": { "rotations": { "right_shoulder": {"raise": 40, "forward": 45}, "left_shoulder": {"raise": 40, "forward": 45}, "right_elbow": {"bend": 90}, "left_elbow": {"bend": 90}, "right_knee": {"bend": 50}, "left_knee": {"bend": 50}, "spine": {"forward": 15} }, "translations": { "hips": {"y": 15} } },
    "chicken_dance": { "rotations": { "right_shoulder": {"raise": 80}, "left_shoulder": {"raise": 80}, "right_elbow": {"bend": 120}, "left_elbow": {"bend": 120} } },
    
    # ... (Includes all 80+ poses from adopter's list, abbreviated for space but fully supported via preset list) ...
    "windmill_right": {"rotations": {"right_shoulder": {"raise": 180, "forward": 45}, "right_elbow": {"bend": 0}}},
    "windmill_left": {"rotations": {"left_shoulder": {"raise": 180, "forward": 45}, "left_elbow": {"bend": 0}}},
    "trex_arms": {"rotations": {"right_shoulder": {"raise": 45, "forward": 60}, "left_shoulder": {"raise": 45, "forward": 60}, "right_elbow": {"bend": 90}, "left_elbow": {"bend": 90}}},
    "chicken_wings": {"rotations": {"right_shoulder": {"raise": 90}, "left_shoulder": {"raise": 90}, "right_elbow": {"bend": 110}, "left_elbow": {"bend": 110}}},
    "robot_arms": {"rotations": {"right_shoulder": {"raise": 90, "forward": 0}, "left_shoulder": {"raise": 90, "forward": 0}, "right_elbow": {"bend": 90}, "left_elbow": {"bend": 90}}},
    "airplane": {"rotations": {"right_shoulder": {"raise": 90}, "left_shoulder": {"raise": 90}, "right_elbow": {"bend": 0}, "left_elbow": {"bend": 0}}},
    "arms_crossed_high": {"rotations": {"right_shoulder": {"raise": 90, "forward": 80}, "left_shoulder": {"raise": 90, "forward": 80}, "right_elbow": {"bend": 100}, "left_elbow": {"bend": 100}}},
    "superman": {"rotations": {"right_shoulder": {"raise": 120, "forward": 45}, "left_shoulder": {"raise": 120, "forward": 45}, "right_elbow": {"bend": 0}, "left_elbow": {"bend": 0}, "spine": {"forward": 15}}},
    "one_arm_wave": {"rotations": {"right_shoulder": {"raise": 150}, "left_shoulder": {"raise": -10}, "right_elbow": {"bend": -20}, "spine": {"tilt": 10}}},
    "twisted_reach": {"rotations": {"right_shoulder": {"raise": 140, "forward": -30}, "left_shoulder": {"raise": 20, "forward": 40}, "spine": {"twist": 45, "tilt": 15}}},
    "diagonal_stretch": {"rotations": {"right_shoulder": {"raise": 160}, "left_shoulder": {"raise": -20}, "right_knee": {"bend": 15}, "left_knee": {"bend": 45}, "spine": {"tilt": 30}}, "translations": {"hips": {"x": -10, "y": 8}}},
    "pretzel_twist": {"rotations": {"right_shoulder": {"raise": 45, "forward": 80}, "left_shoulder": {"raise": 110, "forward": -40}, "right_elbow": {"bend": 90}, "left_elbow": {"bend": 30}, "spine": {"twist": 60, "tilt": -20}}},
    "side_bend_reach": {"rotations": {"right_shoulder": {"raise": 170}, "left_shoulder": {"raise": -15}, "spine": {"tilt": 45}}, "translations": {"upper": {"x": 15}}},
    "moonwalk_lean": {"rotations": {"right_knee": {"bend": 20}, "left_knee": {"bend": 20}, "right_ankle": {"bend": -15}, "left_ankle": {"bend": -15}, "spine": {"forward": 35}}},
    "thriller": {"rotations": {"right_shoulder": {"raise": 90, "forward": 60}, "left_shoulder": {"raise": 90, "forward": 60}, "right_elbow": {"bend": 60}, "left_elbow": {"bend": 60}, "head": {"nod": -20}}},
    "ymca_y": {"rotations": {"right_shoulder": {"raise": 130}, "left_shoulder": {"raise": 130}, "right_elbow": {"bend": 0}, "left_elbow": {"bend": 0}}},
    "ymca_m": {"rotations": {"right_shoulder": {"raise": 120}, "left_shoulder": {"raise": 120}, "right_elbow": {"bend": 90}, "left_elbow": {"bend": 90}}},
    "deep_squat": {"rotations": {"right_knee": {"bend": 90}, "left_knee": {"bend": 90}, "right_ankle": {"bend": 30}, "left_ankle": {"bend": 30}, "spine": {"forward": 30}}, "translations": {"all": {"y": 30}}},
    "backbend": {"rotations": {"spine": {"forward": -45}, "right_shoulder": {"raise": 150, "forward": -30}, "left_shoulder": {"raise": 150, "forward": -30}}, "translations": {"upper": {"y": -10}}},
    "forward_fold": {"rotations": {"spine": {"forward": 90}, "right_shoulder": {"raise": 20, "forward": 40}, "left_shoulder": {"raise": 20, "forward": 40}}, "translations": {"upper": {"y": 20}}},
    "jump_prep": {"rotations": {"right_knee": {"bend": 50}, "left_knee": {"bend": 50}, "right_ankle": {"bend": 25}, "left_ankle": {"bend": 25}, "spine": {"forward": 20}, "right_shoulder": {"raise": -20, "forward": 30}, "left_shoulder": {"raise": -20, "forward": 30}}, "translations": {"all": {"y": 20}}},
    "jump_peak": {"rotations": {"right_knee": {"bend": -10}, "left_knee": {"bend": -10}, "right_shoulder": {"raise": 120}, "left_shoulder": {"raise": 120}, "spine": {"forward": -10}}, "translations": {"all": {"y": -25}}},
    "side_tilt_extreme": {"rotations": {"spine": {"tilt": 50}, "right_shoulder": {"raise": 160}, "left_shoulder": {"raise": -10}}, "translations": {"upper": {"x": 18}}},
    "twist_extreme": {"rotations": {"spine": {"twist": 70}, "right_shoulder": {"raise": 90, "forward": -45}, "left_shoulder": {"raise": 90, "forward": 45}, "head": {"turn": 80}}},
    "arabesque": {"rotations": {"right_knee": {"bend": -20}, "left_knee": {"bend": 0}, "spine": {"forward": 25}, "right_shoulder": {"raise": 90}, "left_shoulder": {"raise": 90}}, "translations": {"upper": {"y": 5}}},
    "warrior_pose": {"rotations": {"right_knee": {"bend": 60}, "left_knee": {"bend": 0}, "right_shoulder": {"raise": 90}, "left_shoulder": {"raise": 90}, "spine": {"forward": 5}}, "translations": {"hips": {"x": 10, "z": 15}}},
    "tree_pose": {"rotations": {"left_knee": {"bend": 90}, "right_knee": {"bend": 0}, "right_shoulder": {"raise": 150}, "left_shoulder": {"raise": 150}}},
    "star_jump": {"rotations": {"right_shoulder": {"raise": 110}, "left_shoulder": {"raise": 110}, "right_elbow": {"bend": 0}, "left_elbow": {"bend": 0}, "right_knee": {"bend": -30}, "left_knee": {"bend": -30}}},
    "splits_prep": {"rotations": {"right_knee": {"bend": 0}, "left_knee": {"bend": 0}, "spine": {"forward": 40}}, "translations": {"all": {"y": 25}}},
    "bridge_prep": {"rotations": {"spine": {"forward": -50}, "right_shoulder": {"raise": 170, "forward": -40}, "left_shoulder": {"raise": 170, "forward": -40}, "right_knee": {"bend": 70}, "left_knee": {"bend": 70}}, "translations": {"all": {"y": 20}}},
    "celebrate": {"rotations": {"right_shoulder": {"raise": 160}, "left_shoulder": {"raise": 160}, "right_elbow": {"bend": 70}, "left_elbow": {"bend": 70}, "spine": {"forward": -15}}, "translations": {"upper": {"y": -5}}},
    "defeated": {"rotations": {"spine": {"forward": 45}, "right_shoulder": {"raise": -10, "forward": 30}, "left_shoulder": {"raise": -10, "forward": 30}, "head": {"nod": 40}}, "translations": {"all": {"y": 10}}},
    "thinking": {"rotations": {"right_shoulder": {"raise": 90, "forward": 60}, "right_elbow": {"bend": 90}, "head": {"nod": 20, "tilt": 10}}},
    "shrug": {"rotations": {"right_shoulder": {"raise": 30}, "left_shoulder": {"raise": 30}, "right_elbow": {"bend": 45}, "left_elbow": {"bend": 45}, "head": {"tilt": 15}}, "translations": {"upper": {"y": -3}}},
    "pointing_accusatory": {"rotations": {"right_shoulder": {"raise": 90, "forward": 45}, "right_elbow": {"bend": 0}, "left_shoulder": {"raise": -5}, "spine": {"twist": 20}}},
    "dramatic_turn": {"rotations": {"spine": {"twist": 80, "tilt": 15}, "right_shoulder": {"raise": 60, "forward": -30}, "left_shoulder": {"raise": 40, "forward": 60}, "head": {"turn": 90}}},
    "confident_stance": {"rotations": {"right_shoulder": {"raise": 30, "forward": 50}, "left_shoulder": {"raise": 30, "forward": 50}, "right_elbow": {"bend": 90}, "left_elbow": {"bend": 90}, "spine": {"forward": -5}}},
    "wave_prep": {"rotations": {"spine": {"forward": 40}, "right_shoulder": {"raise": 70}, "left_shoulder": {"raise": 70}, "right_knee": {"bend": 30}, "left_knee": {"bend": 30}}, "translations": {"hips": {"y": 12}}},
    "tutting": {"rotations": {"right_shoulder": {"raise": 90}, "left_shoulder": {"raise": 90}, "right_elbow": {"bend": 90}, "left_elbow": {"bend": 90}}},
    "swagger_lean": {"rotations": {"spine": {"forward": -20, "tilt": 10}, "right_shoulder": {"raise": -10}, "left_shoulder": {"raise": -10}}},
    "mosh_stance": {"rotations": {"right_knee": {"bend": 35}, "left_knee": {"bend": 35}, "right_shoulder": {"raise": 60, "forward": 50}, "left_shoulder": {"raise": 60, "forward": 50}, "spine": {"forward": 25}}, "translations": {"all": {"y": 10}}},
    "scarecrow": {"rotations": {"right_shoulder": {"raise": 100}, "left_shoulder": {"raise": 100}, "right_elbow": {"bend": 150}, "left_elbow": {"bend": 150}, "head": {"tilt": 30}, "spine": {"tilt": 10}}},
    "puppet_strings": {"rotations": {"right_shoulder": {"raise": 140}, "left_shoulder": {"raise": 140}, "right_elbow": {"bend": 90}, "left_elbow": {"bend": 90}, "right_knee": {"bend": 40}, "left_knee": {"bend": 40}}},
    "mannequin": {"rotations": {"right_shoulder": {"raise": 70, "forward": 20}, "left_shoulder": {"raise": 50}, "right_elbow": {"bend": 60}, "spine": {"twist": 20, "tilt": -5}}},
    "zombie_walk": {"rotations": {"right_shoulder": {"raise": 80, "forward": 70}, "left_shoulder": {"raise": 80, "forward": 70}, "right_elbow": {"bend": 150}, "left_elbow": {"bend": 150}, "right_knee": {"bend": 25}, "left_knee": {"bend": 10}, "spine": {"forward": 20}}},
    "slav_squat": {"rotations": {"right_knee": {"bend": 100}, "left_knee": {"bend": 100}, "right_ankle": {"bend": 45}, "left_ankle": {"bend": 45}, "spine": {"forward": 35}}, "translations": {"all": {"y": 35}}},
    "matrix_dodge": {"rotations": {"spine": {"forward": -40, "tilt": 25}, "right_knee": {"bend": 30}, "left_knee": {"bend": 50}, "right_shoulder": {"raise": 120}, "left_shoulder": {"raise": 100}}},
}

MOVE_COMBOS = {
    "low": [
        ["step_left", "step_right", "step_left", "step_right"],
        ["lean_left", "lean_right", "lean_left", "lean_right"],
        ["step_left", "lean_left", "step_right", "lean_right"],
    ],
    "medium": [
        ["pump_left_hard", "pump_right_hard", "pump_left_hard", "pump_right_hard"],
        ["arms_wide", "step_left", "arms_wide", "step_right"], 
        ["step_left", "arms_up_open", "step_right", "arms_up_open"],
        ["lean_left", "pump_left_hard", "lean_right", "pump_right_hard"],
    ],
    "high": [
        ["pump_right_hard", "pump_right_hard", "pump_left_hard", "pump_left_hard"],
        ["arms_up_open", "arms_wide", "power_crouch", "arms_wide"], 
        ["lean_left", "pump_right_hard", "lean_right", "pump_left_hard"],
    ]
}

# PRESET STYLE LISTS
PRESET_SEQUENCES = {
    "hip_hop": ["dab", "running_man", "pop_lock", "breakdance_freeze", "gangnam_style"],
    "pop_dance": ["dab", "floss_right", "floss_left", "disco_point_up", "finger_guns", "arms_up_open"],
    "rock": ["headbang_forward", "headbang_back", "rock_out", "air_guitar", "arms_wide"],
    "disco": ["disco_point_up", "disco_point_down", "finger_guns", "arms_wide", "step_right", "step_left"],
    "fitness": ["lunge_right", "lunge_left", "power_crouch", "arms_up_open", "step_left", "step_right"],
    "edm_rave": ["windmill_right", "windmill_left", "star_jump", "jump_prep", "jump_peak", "arms_up", "celebrate"],
    "thriller_horror": ["zombie_walk", "thriller", "scarecrow", "puppet_strings", "mannequin", "defeated"],
    "meme_dances": ["dab", "floss_right", "floss_left", "gangnam_style", "chicken_dance", "slav_squat"],
    "internet_viral": ["dab", "floss_right", "gangnam_style", "moonwalk_lean", "matrix_dodge", "slav_squat"],
    "classic_disco": ["disco_point_up", "disco_point_down", "finger_guns", "ymca_y", "ymca_m", "arms_out"],
    "breakdance": ["breakdance_freeze", "windmill_right", "windmill_left", "running_man", "pop_lock", "wave_prep"],
    "yoga_flow": ["warrior_pose", "tree_pose", "arabesque", "confident_stance", "neutral"],
    "crazy_chaos": ["zombie_walk", "slav_squat", "matrix_dodge", "trex_arms", "chicken_dance", "puppet_strings", "scarecrow"],
    "gymnastics": ["arabesque", "warrior_pose", "tree_pose", "star_jump", "splits_prep", "bridge_prep", "backbend"],
    "expressive": ["celebrate", "defeated", "thinking", "shrug", "dramatic_turn", "confident_stance", "pointing_accusatory"],
    "arms_only": ["arms_up", "arms_out", "pump_right_hard", "pump_left_hard", "windmill_right", "windmill_left", "airplane"],
    "body_moves": ["lean_right", "lean_left", "twist_extreme", "side_tilt_extreme"],
    "extreme_energy": ["jump_peak", "star_jump", "windmill_right", "backbend", "twist_extreme", "matrix_dodge"],
    "chill_vibe": ["neutral", "swagger_lean", "confident_stance", "thinking", "lean_right", "lean_left"],
    "michael_jackson": ["moonwalk_lean", "thriller", "dab", "celebrate", "finger_guns"],
    "fortnite": ["dab", "floss_right", "floss_left", "robot_arms", "celebrate"],
}

# ============================================================================
# MOTION DYNAMICS (PHYSICS & CONSTRAINTS)
# ============================================================================

class MotionDynamics:
    # 0 = head, 1 = neck
    JOINT_DRAG = {
        0: 0.1, 1: 0.1, 2: 0.05, 5: 0.05,
        3: 0.08, 6: 0.08, 4: 0.15, 7: 0.15,
        8: 0.05, 11: 0.05,
        9: 0.05, 12: 0.05, 
        10: 0.01, 13: 0.01 
    }
    
    def __init__(self, start_pose, num_joints=18):
        self.num_joints = num_joints
        self.velocities = np.zeros((num_joints, 3))
        self.positions = start_pose.copy()
        self.dt = 1.0 / 24.0
        
        # --- FIXED BONE LENGTH CALCULATION FOR MULTIPLE CHARACTERS ---
        # Calculate bone lengths for every character individually based on their specific start pose
        self.bone_lengths = {}
        for j in range(num_joints):
            base_idx = j % 18
            parent_base = JOINT_PARENTS.get(base_idx, -1)
            if parent_base != -1:
                # Find the parent index within the same character chunk
                char_offset = (j // 18) * 18
                parent_idx = parent_base + char_offset
                
                # Store distance
                self.bone_lengths[j] = np.linalg.norm(start_pose[j] - start_pose[parent_idx])

    def step_towards(self, target_pose, urgency=1.0):
        # Physics update
        for j in range(self.num_joints):
            # Map index to 0-17 range for property lookup
            base_idx = j % 18
            
            drag = self.JOINT_DRAG.get(base_idx, 0.1)
            stiffness = 20.0 * urgency 
            
            # Sticky feet
            if base_idx in [10, 13]:
                stiffness = 60.0 * urgency 
                drag = 0.0 
            
            displacement = target_pose[j] - self.positions[j]
            spring_force = displacement * stiffness
            damping = 1.5 * np.sqrt(stiffness)
            damping_force = -self.velocities[j] * damping
            
            acceleration = spring_force + damping_force
            self.velocities[j] += acceleration * self.dt
            self.velocities[j] *= (1.0 - drag)
            self.positions[j] += self.velocities[j] * self.dt
        
        # --- BONE LENGTH CONSTRAINT SOLVER ---
        solve_order = [0, 2, 5, 8, 11, 3, 4, 6, 7, 9, 10, 12, 13, 14, 15, 16, 17]
        
        # Iterate over every character chunk
        for char_idx in range(self.num_joints // 18):
            offset = char_idx * 18
            for child_base in solve_order:
                child = child_base + offset
                parent_base = JOINT_PARENTS[child_base]
                if parent_base == -1: continue
                parent = parent_base + offset
                
                current_vec = self.positions[child] - self.positions[parent]
                current_dist = np.linalg.norm(current_vec)
                
                # LOOKUP CORRECT LENGTH FOR THIS SPECIFIC JOINT
                target_dist = self.bone_lengths.get(child, current_dist)
                
                if current_dist > 1e-4:
                    correction = current_vec * (target_dist / current_dist)
                    self.positions[child] = self.positions[parent] + correction
                
        return self.positions.copy()

# ============================================================================
# BEAT DRIVEN POSE NODE (THE BRAIN)
# ============================================================================

class SCAILBeatDrivenPose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_pose": ("SCAIL_POSE", {"tooltip": "Base skeleton pose from DWPose or Generator"}),
                "beat_info": ("SCAIL_BEAT_INFO", {"tooltip": "Beat detection data from SCAILBeatDetector"}),
                "audio_features": ("SCAIL_AUDIO_FEATURES", {"tooltip": "Audio analysis from SCAILAudioFeatureExtractor"}),
                "dance_style": (["auto"] + sorted(list(PRESET_SEQUENCES.keys())), {"default": "auto", "tooltip": "Dance style preset. 'auto' selects moves based on energy level."}),
                "interaction_mode": (["unison", "mirror", "random"], {"default": "mirror", "tooltip": "Multi-character coordination: unison=same moves, mirror=left/right swapped, random=independent"}),
                "energy_style": (["auto", "low", "medium", "high"], {"default": "auto", "tooltip": "Override energy detection. 'auto' uses audio RMS."}),
                "motion_smoothness": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 3.0, "tooltip": "Lower = snappier, higher = smoother/floaty"}),
                "anticipation": ("INT", {"default": 3, "min": 0, "max": 10, "tooltip": "Frames to start moving BEFORE beat"}),
                "groove_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "tooltip": "Hip sway/bounce intensity synced to tempo"}),
                "bass_intensity": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 5.0, "tooltip": "How much bass drives motion snap"}),
                "treble_intensity": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 5.0, "tooltip": "How much treble drives arm movement"}),
                "phrase_bars": ([4, 8, 16], {"default": 8, "tooltip": "Bars per phrase. Choreography changes on phrase boundaries."}),
                "staging_mode": (["off", "subtle", "dynamic"], {"default": "subtle", "tooltip": "Stage movement: off=stationary, subtle=small sway, dynamic=larger travel"}),
                "hit_easing": (["linear", "ease_out", "ease_in_out", "bounce", "elastic", "punch"], {"default": "ease_out", "tooltip": "Motion curve for pose transitions. Affects how moves hit the beat."}),
                "pose_blend": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "tooltip": "Blend between poses. 0=hard keyframes, 1=maximum smoothing between poses"}),
                "loop_mode": ("BOOLEAN", {"default": False, "tooltip": "Ensure first and last frames match for seamless looping"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 99999, "tooltip": "Random seed for deterministic choreography"}),
            }
        }
    
    RETURN_TYPES = ("SCAIL_POSE_SEQUENCE",)
    RETURN_NAMES = ("pose_sequence",)
    FUNCTION = "generate"
    CATEGORY = "SCAIL-AudioReactive"

    # Joint Groups for Rotation
    GROUPS = {
        "spine": [1, 2, 5, 8, 11], "head": [0, 14, 15, 16, 17],
        "right_shoulder": [2, 3, 4], "left_shoulder": [5, 6, 7],
        "right_elbow": [3, 4], "left_elbow": [6, 7],
        "right_knee": [9, 10], "left_knee": [12, 13],
        "right_ankle": [10], "left_ankle": [13]
    }
    PIVOTS = {
        "spine": 1, "head": 1, "right_shoulder": 2, "left_shoulder": 5,
        "right_elbow": 3, "left_elbow": 6, "right_knee": 9, "left_knee": 12,
        "right_ankle": 10, "left_ankle": 13
    }
    AXES = {"nod": 0, "forward": 0, "bend": 0, "turn": 1, "twist": 1, "tilt": 2, "raise": 2}

    def _create_neutral_structure(self, char_pose):
        """Construct a neutral standing pose based on the limb lengths of the reference.
        Straightens both legs AND arms to create a consistent base for dance poses."""
        neutral = char_pose.copy()
        
        # === NEUTRALIZE LEGS ===
        r_hip = char_pose[8]
        l_hip = char_pose[11]
        
        r_leg_vec = char_pose[10] - r_hip
        l_leg_vec = char_pose[13] - l_hip
        r_len = np.linalg.norm(r_leg_vec)
        l_len = np.linalg.norm(l_leg_vec)
        
        # Preserve 20% of stance width/depth to prevent legs clipping
        r_stance_x = r_leg_vec[0] * 0.2
        r_stance_z = r_leg_vec[2] * 0.2
        l_stance_x = l_leg_vec[0] * 0.2
        l_stance_z = l_leg_vec[2] * 0.2
        
        r_y = np.sqrt(max(0, r_len**2 - r_stance_x**2 - r_stance_z**2))
        neutral[10] = r_hip + np.array([r_stance_x, r_y, r_stance_z])
        r_dir = neutral[10] - r_hip
        r_dir_norm = r_dir / (np.linalg.norm(r_dir) + 1e-8)
        r_thigh_len = np.linalg.norm(char_pose[9] - char_pose[8])
        neutral[9] = r_hip + r_dir_norm * r_thigh_len

        l_y = np.sqrt(max(0, l_len**2 - l_stance_x**2 - l_stance_z**2))
        neutral[13] = l_hip + np.array([l_stance_x, l_y, l_stance_z])
        l_dir = neutral[13] - l_hip
        l_dir_norm = l_dir / (np.linalg.norm(l_dir) + 1e-8)
        l_thigh_len = np.linalg.norm(char_pose[12] - char_pose[11])
        neutral[12] = l_hip + l_dir_norm * l_thigh_len
        
        # === NEUTRALIZE ARMS ===
        # Right arm: shoulder(2) -> elbow(3) -> wrist(4)
        r_shoulder = char_pose[2]
        r_upper_arm_len = np.linalg.norm(char_pose[3] - char_pose[2])
        r_forearm_len = np.linalg.norm(char_pose[4] - char_pose[3])
        
        # Neutral arm position: slightly down and forward (relaxed idle)
        r_arm_dir = np.array([-0.8, 0.5, 0.2])  # Down, slightly out, slightly forward
        r_arm_dir = r_arm_dir / (np.linalg.norm(r_arm_dir) + 1e-8)
        neutral[3] = r_shoulder + r_arm_dir * r_upper_arm_len  # elbow
        neutral[4] = neutral[3] + r_arm_dir * r_forearm_len    # wrist
        
        # Left arm: shoulder(5) -> elbow(6) -> wrist(7)
        l_shoulder = char_pose[5]
        l_upper_arm_len = np.linalg.norm(char_pose[6] - char_pose[5])
        l_forearm_len = np.linalg.norm(char_pose[7] - char_pose[6])
        
        # Mirror for left arm
        l_arm_dir = np.array([0.8, 0.5, 0.2])  # Down, slightly out (mirrored), slightly forward
        l_arm_dir = l_arm_dir / (np.linalg.norm(l_arm_dir) + 1e-8)
        neutral[6] = l_shoulder + l_arm_dir * l_upper_arm_len  # elbow
        neutral[7] = neutral[6] + l_arm_dir * l_forearm_len    # wrist
        
        return neutral

    def _get_body_scale(self, char_pose):
        neck = char_pose[1]
        mid_hip = (char_pose[8] + char_pose[11]) / 2.0
        torso_len = np.linalg.norm(neck - mid_hip)
        STANDARD_TORSO = 160.0
        if torso_len < 1.0: return 1.0
        return torso_len / STANDARD_TORSO

    def _apply_pose(self, base, pose_name, scale_factor=1.0):
        pose_def = DANCE_POSES.get(pose_name, DANCE_POSES["neutral"])
        res = base.copy()
        
        for group, motions in pose_def.get("rotations", {}).items():
            for motion, deg in motions.items():
                if group in self.GROUPS and motion in self.AXES:
                    axis = self.AXES[motion]
                    rad = np.radians(deg)
                    if "left" in group and motion in ["raise", "tilt"]: rad = -rad
                    res = apply_chain_rotation(res, self.PIVOTS[group], self.GROUPS[group], axis, rad)
        
        for target, off in pose_def.get("translations", {}).items():
            vec = np.array([off.get("x",0), off.get("y",0), off.get("z",0)]) * scale_factor
            if target == "hips":
                for i in [1, 2, 5, 8, 11]: 
                    res[i] += vec
            elif target == "all":
                res += vec
            elif target in JOINT_NAMES:
                idx = JOINT_NAMES.index(target)
                res[idx] += vec
                
        return res

    def _apply_continuous_groove(self, joints, time, tempo, amount, scale_factor=1.0):
        beat_period = 60.0 / (tempo + 1e-5)
        phase = (time / beat_period) * 2 * np.pi
        sway = np.sin(phase * 0.5) * 20.0 * amount * scale_factor
        bounce = np.abs(np.cos(phase)) * 8.0 * amount * scale_factor
        indices = [1, 2, 5, 8, 11] # Neck, Shoulders, Hips
        joints[indices, 0] += sway
        joints[indices, 1] += bounce
        joints = apply_chain_rotation(joints, 1, [0, 14, 15, 16, 17], 2, np.radians(-sway * 0.1))
        return joints

    def generate(self, base_pose, beat_info, audio_features, dance_style, interaction_mode, energy_style, motion_smoothness, anticipation, groove_amount, bass_intensity, treble_intensity, phrase_bars, staging_mode, hit_easing, pose_blend, loop_mode, seed):
        rng = np.random.default_rng(seed)
        base_joints_all = base_pose["joints"].numpy()
        char_count = base_pose.get("char_count", 1)
        print(f"[SCAIL] Animating {char_count} characters. Style: {dance_style}, Phrase: {phrase_bars} bars, Easing: {hit_easing}, Loop: {loop_mode}")
        
        # Get easing function
        easing_func = EASING_FUNCTIONS.get(hit_easing, ease_out)
        
        frame_count = beat_info["frame_count"]
        fps = beat_info["fps"]
        beat_frames = beat_info["beat_frames"]
        downbeat_frames = beat_info.get("downbeat_frames", beat_frames[::4] if len(beat_frames) >= 4 else beat_frames)
        tempo = beat_info["tempo"]
        energy = beat_info["energy"].numpy()
        bass = audio_features["bass"].numpy()
        treble = audio_features["treble"].numpy()
        
        if len(bass) != frame_count:
            bass = np.interp(np.linspace(0,1,frame_count), np.linspace(0,1,len(bass)), bass)
            treble = np.interp(np.linspace(0,1,frame_count), np.linspace(0,1,len(treble)), treble)

        # Calculate phrase timing
        if len(beat_frames) >= 2:
            frames_per_beat = float(np.median(np.diff(beat_frames)))
        else:
            frames_per_beat = (60.0 * fps) / max(tempo, 1.0)
        
        phrase_len_frames = int(frames_per_beat * 4.0 * phrase_bars)  # 4 beats per bar
        phrase_len_frames = max(1, phrase_len_frames)
        
        # Build phrase boundaries from downbeats or regular intervals
        if len(downbeat_frames) >= 2:
            phrase_starts = [int(downbeat_frames[i]) for i in range(0, len(downbeat_frames), phrase_bars)]
        else:
            phrase_starts = list(range(0, frame_count, phrase_len_frames))
        if not phrase_starts or phrase_starts[0] != 0:
            phrase_starts = [0] + phrase_starts
        phrase_start_set = set(phrase_starts)

        # Staging positions (normalized, will be scaled per character)
        stage_positions = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.3, 0.0, 0.0]),
            np.array([-0.3, 0.0, 0.0]),
            np.array([0.15, 0.1, 0.0]),
            np.array([-0.15, 0.1, 0.0]),
            np.array([0.2, -0.08, 0.0]),
            np.array([-0.2, -0.08, 0.0]),
        ]
        # Scales: subtle = gentle sway, dynamic = noticeable travel
        staging_scale = {"off": 0.0, "subtle": 0.4, "dynamic": 0.8}.get(staging_mode, 0.4)

        char_keyframes = [] 
        char_stage_targets = []  # Track stage position per character
        
        for c in range(char_count):
            keyframes = {0: "neutral"}
            current_combo = []
            combo_idx = 0
            current_phrase = -1
            char_rng = np.random.default_rng(seed + c)
            
            # Initialize stage position
            char_stage_targets.append({"current": np.zeros(3), "target": np.zeros(3), "start_frame": 0})
            
            for beat_frame in beat_frames:
                target_frame = max(0, beat_frame - anticipation)
                
                # Check if we're in a new phrase
                phrase_idx = sum(1 for ps in phrase_starts if ps <= target_frame) - 1
                new_phrase = phrase_idx != current_phrase
                
                if new_phrase:
                    current_phrase = phrase_idx
                    # Pick new combo for this phrase
                    phrase_start = phrase_starts[phrase_idx] if phrase_idx < len(phrase_starts) else 0
                    phrase_end = phrase_starts[phrase_idx + 1] if phrase_idx + 1 < len(phrase_starts) else frame_count
                    local_energy = np.mean(energy[phrase_start:phrase_end])
                    
                    if dance_style == "auto":
                        style = energy_style if energy_style != "auto" else ("high" if local_energy > 0.6 else "medium" if local_energy > 0.3 else "low")
                        available_combos = MOVE_COMBOS.get(style, MOVE_COMBOS["medium"])
                    else:
                        available_combos = [PRESET_SEQUENCES.get(dance_style, PRESET_SEQUENCES["hip_hop"])]
                    
                    if c == 0 or interaction_mode == "random":
                        current_combo = available_combos[char_rng.integers(0, len(available_combos))]
                        combo_idx = 0
                        
                        # Update stage target for this phrase
                        if staging_mode != "off":
                            new_stage = stage_positions[char_rng.integers(0, len(stage_positions))]
                            char_stage_targets[c]["target"] = new_stage * staging_scale
                            char_stage_targets[c]["start_frame"] = target_frame

                pose_name = "neutral"
                if c == 0 or interaction_mode == "random":
                    pose_name = current_combo[combo_idx % len(current_combo)]
                
                elif interaction_mode == "unison":
                    pose_name = char_keyframes[0].get(target_frame, "neutral")
                    # Copy leader's stage target
                    if new_phrase and staging_mode != "off":
                        char_stage_targets[c]["target"] = char_stage_targets[0]["target"].copy()
                        char_stage_targets[c]["start_frame"] = target_frame
                    
                elif interaction_mode == "mirror":
                    leader_pose = char_keyframes[0].get(target_frame, "neutral")
                    if "left" in leader_pose: pose_name = leader_pose.replace("left", "right")
                    elif "right" in leader_pose: pose_name = leader_pose.replace("right", "left")
                    else: pose_name = leader_pose
                    # Mirror leader's stage X position
                    if new_phrase and staging_mode != "off":
                        leader_target = char_stage_targets[0]["target"]
                        char_stage_targets[c]["target"] = np.array([-leader_target[0], leader_target[1], leader_target[2]])
                        char_stage_targets[c]["start_frame"] = target_frame

                keyframes[target_frame] = pose_name
                combo_idx += 1
            
            char_keyframes.append(keyframes)

        # Loop mode: ensure last pose returns to first pose
        if loop_mode and len(beat_frames) > 0:
            # Add a keyframe at the end that mirrors the start
            loop_blend_frames = int(frames_per_beat * 2)  # 2 beats to blend back
            for c in range(char_count):
                first_pose = char_keyframes[c].get(0, "neutral")
                char_keyframes[c][max(0, frame_count - loop_blend_frames)] = first_pose

        # Build sorted keyframe list per character for interpolation
        char_keyframe_times = []
        char_keyframe_poses = []
        for c in range(char_count):
            times = sorted(char_keyframes[c].keys())
            poses = [char_keyframes[c][t] for t in times]
            char_keyframe_times.append(times)
            char_keyframe_poses.append(poses)

        motion = MotionDynamics(base_joints_all.copy(), num_joints=18*char_count)
        motion.dt = 1.0 / fps
        pose_sequence = []
        
        # Pre-compute all target poses for blending
        char_pose_cache = {}  # (char_idx, pose_name) -> pose array
        
        def get_pose_for_char(c, pose_name):
            key = (c, pose_name)
            if key not in char_pose_cache:
                char_ref = base_joints_all[c*18:(c+1)*18]
                char_base = self._create_neutral_structure(char_ref)
                scale = self._get_body_scale(char_ref)
                char_pose_cache[key] = self._apply_pose(char_base, pose_name, scale_factor=scale)
            return char_pose_cache[key]

        current_targets = [base_joints_all[i*18:(i+1)*18].copy() for i in range(char_count)]
        
        for c in range(char_count):
            char_ref = base_joints_all[c*18:(c+1)*18]
            char_neutral = self._create_neutral_structure(char_ref)
            current_targets[c] = char_neutral

        for i in range(frame_count):
            full_target_pose = []
            for c in range(char_count):
                char_ref = base_joints_all[c*18:(c+1)*18]
                scale = self._get_body_scale(char_ref)
                
                times = char_keyframe_times[c]
                poses = char_keyframe_poses[c]
                
                # Find surrounding keyframes for blending
                prev_idx = 0
                for idx, t in enumerate(times):
                    if t <= i:
                        prev_idx = idx
                    else:
                        break
                
                next_idx = min(prev_idx + 1, len(times) - 1)
                prev_time = times[prev_idx]
                next_time = times[next_idx]
                prev_pose_name = poses[prev_idx]
                next_pose_name = poses[next_idx]
                
                prev_pose = get_pose_for_char(c, prev_pose_name)
                next_pose = get_pose_for_char(c, next_pose_name)
                
                # Calculate blend factor with easing
                if next_time > prev_time:
                    raw_t = (i - prev_time) / (next_time - prev_time)
                    raw_t = np.clip(raw_t, 0.0, 1.0)
                    # Apply easing function
                    eased_t = easing_func(raw_t)
                    # Apply pose_blend parameter (0 = snap to keyframes, 1 = full blend)
                    blend_t = eased_t * pose_blend + (1.0 if raw_t >= 1.0 else 0.0) * (1.0 - pose_blend)
                else:
                    blend_t = 1.0
                
                # Blend between poses
                blended_pose = prev_pose * (1.0 - blend_t) + next_pose * blend_t
                current_targets[c] = blended_pose
                
                # Apply staging offset with smooth interpolation
                if staging_mode != "off":
                    st = char_stage_targets[c]
                    # Smooth interpolation toward target
                    blend = 0.02  # Gradual movement
                    st["current"] = st["current"] * (1 - blend) + st["target"] * blend
                    
                    # Clamp to safe range (don't let characters walk out of frame)
                    max_offset = 0.5  # Maximum normalized offset
                    st["current"][0] = np.clip(st["current"][0], -max_offset, max_offset)
                    st["current"][1] = np.clip(st["current"][1], -max_offset * 0.4, max_offset * 0.4)
                    
                    stage_offset = st["current"] * scale * 120  # Scale to world units
                    current_targets[c] = current_targets[c] + stage_offset
                
                full_target_pose.append(current_targets[c])
            
            flat_target = np.concatenate(full_target_pose, axis=0)
            urgency = 1.0 / motion_smoothness
            if bass[i] > 0.7: urgency *= 1.5 
            
            joints = motion.step_towards(flat_target, urgency)
            
            for c in range(char_count):
                start = c * 18
                end = (c + 1) * 18
                char_joints = joints[start:end]
                
                char_ref = base_joints_all[start:end]
                scale = self._get_body_scale(char_ref)
                
                time_sec = i / fps
                char_joints = self._apply_continuous_groove(char_joints, time_sec, tempo, groove_amount, scale_factor=scale)
                
                bounce = bass[i] * bass_intensity * 10.0 * scale
                char_joints[[1,2,5,8,11], 1] += bounce
                
                if treble[i] > 0.1:
                    t_val = treble[i] * treble_intensity * 15.0
                    char_joints = apply_chain_rotation(char_joints, 3, [3,4], 0, np.radians(t_val)) 
                    char_joints = apply_chain_rotation(char_joints, 6, [6,7], 0, np.radians(t_val))
                
                hip_mid_x = (char_joints[8,0] + char_joints[11,0]) / 2.0
                tilt = (char_joints[1,0] - hip_mid_x) * 0.1 
                char_joints = apply_chain_rotation(char_joints, 1, [0,14,15,16,17], 2, np.radians(-tilt * 0.5))
                
                joints[start:end] = char_joints

            pose_sequence.append(joints.copy())
        
        # Loop mode: blend last few frames back to first frame
        if loop_mode and len(pose_sequence) > 1:
            blend_frames = min(int(frames_per_beat), len(pose_sequence) // 4)
            if blend_frames > 0:
                first_pose = pose_sequence[0]
                for b in range(blend_frames):
                    idx = len(pose_sequence) - blend_frames + b
                    t = (b + 1) / (blend_frames + 1)
                    t = easing_func(t)  # Apply easing to loop blend too
                    pose_sequence[idx] = pose_sequence[idx] * (1.0 - t) + first_pose * t
            
        return ({"poses": pose_sequence, "limb_seq": base_pose["limb_seq"], "bone_colors": base_pose["bone_colors"]},)

# ============================================================================
# RENDERER
# ============================================================================

class SCAILPoseRenderer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_sequence": ("SCAIL_POSE_SEQUENCE", {"tooltip": "Animated pose sequence from SCAILBeatDrivenPose"}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "tooltip": "Output image width in pixels"}),
                "height": ("INT", {"default": 768, "min": 64, "max": 4096, "tooltip": "Output image height in pixels"}),
                "auto_half_resolution": ("BOOLEAN", {"default": True, "tooltip": "Render at half res then upscale. Faster."}),
                "fov": ("FLOAT", {"default": 55.0, "min": 10.0, "max": 120.0, "tooltip": "Field of view in degrees"}),
                "cylinder_pixel_radius": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "tooltip": "Thickness of skeleton limbs"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_images",)
    FUNCTION = "render"
    CATEGORY = "SCAIL-AudioReactive"
    
    def render(self, pose_sequence, width, height, auto_half_resolution, fov, cylinder_pixel_radius):
        import taichi as ti
        
        rw, rh = (width//2, height//2) if auto_half_resolution else (width, height)
        scale = 0.5 if auto_half_resolution else 1.0
        
        try: ti.init(arch=ti.gpu, default_fp=ti.f32)
        except: pass
        
        poses, limb_seq, colors = pose_sequence["poses"], pose_sequence["limb_seq"], pose_sequence["bone_colors"]
        
        fov_rad = np.radians(fov)
        focal = max(rh, rw) / (np.tan(fov_rad/2) * 2)
        avg_z = np.mean([p[1,2] for p in poses]) if poses else 800.0
        cyl_radius = cylinder_pixel_radius * avg_z / focal
        
        specs_list = []
        for joints in poses:
            frame_specs = []
            for i, (s, e) in enumerate(limb_seq):
                if s >= len(joints) or e >= len(joints): continue # Safety check
                sp = joints[s].copy() * np.array([scale, scale, 1])
                ep = joints[e].copy() * np.array([scale, scale, 1])
                if np.sum(np.abs(sp)) > 1e-3:
                    frame_specs.append((sp.tolist(), ep.tolist(), colors[i % len(colors)]))
            specs_list.append(frame_specs)
            
        frames_rgba = render_whole(specs_list, H=rh, W=rw, fx=focal, fy=focal, cx=rw/2, cy=rh/2, radius=cyl_radius)
        
        out = np.stack(frames_rgba, axis=0).astype(np.float32) / 255.0
        return (torch.from_numpy(out[:,:,:,:3]).float(),)

# ============================================================================
# UTILITIES
# ============================================================================

class SCAILPoseFromDWPose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_width": ("INT", {"default": 512, "min": 64, "max": 4096, "tooltip": "Width of source image that DWPose was run on"}),
                "image_height": ("INT", {"default": 768, "min": 64, "max": 4096, "tooltip": "Height of source image that DWPose was run on"}),
                "depth": ("FLOAT", {"default": 800.0, "min": 100.0, "max": 5000.0, "tooltip": "Assumed Z-depth for 2D to 3D conversion"}),
                "fov": ("FLOAT", {"default": 55.0, "min": 10.0, "max": 120.0, "tooltip": "Field of view for projection. Match your render FOV."}),
            },
            "optional": {
                "dw_poses": ("DWPOSES", {"tooltip": "DWPose output (preferred format)"}),
                "pose_keypoint": ("POSE_KEYPOINT", {"tooltip": "Alternative OpenPose-style keypoint input"})
            }
        }
    RETURN_TYPES = ("SCAIL_POSE",)
    FUNCTION = "extract"
    CATEGORY = "SCAIL-AudioReactive"
    
    def extract(self, image_width, image_height, depth, fov, dw_poses=None, pose_keypoint=None):
        data = dw_poses if dw_poses else pose_keypoint
        # NOTE: This only extracts ONE character as base. Multi-char is handled by Generator.
        if not data: return SCAILBasePoseGenerator().generate("active_idle", 1, 150.0, 400.0, depth, 0, 0)
        
        candidates = []
        # Handle various DWPose formats
        if isinstance(data, list) and len(data) > 0:
            # Check frame 0
            frame_data = data[0]
            if 'bodies' in frame_data and 'candidate' in frame_data['bodies']:
                candidates = frame_data['bodies']['candidate']
            elif 'candidate' in frame_data:
                candidates = frame_data['candidate']
        elif isinstance(data, torch.Tensor):
            # If tensor, assume single person or handled upstream? 
            candidates = data.cpu().numpy()
            if candidates.ndim == 2: candidates = [candidates]
        elif isinstance(data, dict):
            # Handle Single Frame Dictionary Input (Common from OpenPose)
            if 'bodies' in data and 'candidate' in data['bodies']:
                candidates = data['bodies']['candidate']
            elif 'candidate' in data:
                candidates = data['candidate']

        if len(candidates) == 0:
            print("[SCAIL] No candidates found in DWPose data. Fallback to Generator.")
            return SCAILBasePoseGenerator().generate("active_idle", 1, 150.0, 400.0, depth, 0, 0)
            
        print(f"[SCAIL] Found {len(candidates)} candidates in DWPose data.")
            
        # Processing candidates
        total_joints = []
        full_limb_seq = []
        full_colors = []
        
        focal = max(image_width, image_height) / (np.tan(np.radians(fov)/2)*2)
        cx, cy = image_width/2, image_height/2
        
        char_count = 0
        
        for i, kps in enumerate(candidates):
            kps = np.array(kps)
            if kps.shape != (18, 2) and kps.shape != (18, 3):
                continue # Skip invalid shapes
                
            joints = np.zeros((18, 3), dtype=np.float32)
            valid_joints = 0
            
            for j in range(18):
                x, y = kps[j,0], kps[j,1]
                # Normalize check
                if x <= 1.5 and x > 0: x *= image_width
                if y <= 1.5 and y > 0: y *= image_height
                
                if x > 0 and y > 0:
                    joints[j] = [(x-cx)*depth/focal, (y-cy)*depth/focal, depth]
                    valid_joints += 1
            
            # Interpolate missing
            for j in range(18):
                if np.all(joints[j]==0) and JOINT_PARENTS[j]!=-1:
                     # Simple heuristic: duplicate parent
                    joints[j] = joints[JOINT_PARENTS[j]] + [0,10,0]

            if valid_joints > 0:
                total_joints.append(joints)
                
                # Add limbs and colors for this character
                index_offset = char_count * 18
                for (s, e) in LIMB_SEQ:
                    full_limb_seq.append((s + index_offset, e + index_offset))
                full_colors.extend(BONE_COLORS)
                
                char_count += 1

        if char_count == 0:
             return SCAILBasePoseGenerator().generate("active_idle", 1, 150.0, 400.0, depth, 0, 0)

        final_joints = np.concatenate(total_joints, axis=0)

        return ({"joints": torch.from_numpy(final_joints).float(), 
                 "limb_seq": full_limb_seq, 
                 "bone_colors": full_colors,
                 "char_count": char_count, 
                 "joint_names": JOINT_NAMES},)

class SCAILAlignPoseToReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_sequence": ("SCAIL_POSE_SEQUENCE", {"tooltip": "Animated pose sequence to align"}),
                "reference_pose": ("SCAIL_POSE", {"tooltip": "Target pose to align to (uses neck position)"})
            }
        }
    RETURN_TYPES = ("SCAIL_POSE_SEQUENCE",)
    RETURN_NAMES = ("aligned_sequence",)
    FUNCTION = "align"
    CATEGORY = "SCAIL-AudioReactive"
    
    def align(self, pose_sequence, reference_pose):
        poses = [p.copy() for p in pose_sequence["poses"]]
        if not poses: return (pose_sequence,)
        # Align NECK of First Character
        offset = reference_pose["joints"][1].numpy() - poses[0][1] 
        for i in range(len(poses)): poses[i] += offset
        return ({"poses": poses, "limb_seq": pose_sequence["limb_seq"], "bone_colors": pose_sequence["bone_colors"]},)

# ============================================================================
# MAPPINGS
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "SCAILAudioFeatureExtractor": SCAILAudioFeatureExtractor,
    "SCAILBasePoseGenerator": SCAILBasePoseGenerator,
    "SCAILBeatDetector": SCAILBeatDetector,
    "SCAILBeatDrivenPose": SCAILBeatDrivenPose,
    "SCAILPoseRenderer": SCAILPoseRenderer,
    "SCAILPoseFromDWPose": SCAILPoseFromDWPose,
    "SCAILAlignPoseToReference": SCAILAlignPoseToReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAILAudioFeatureExtractor": "SCAIL Audio Features",
    "SCAILBasePoseGenerator": "SCAIL Base Pose",
    "SCAILBeatDetector": "SCAIL Beat Detect",
    "SCAILBeatDrivenPose": "SCAIL Beat-Driven Dance",
    "SCAILPoseRenderer": "SCAIL Pose Render",
    "SCAILPoseFromDWPose": "SCAIL Pose from DWPose",
    "SCAILAlignPoseToReference": "SCAIL Align to Reference",
}

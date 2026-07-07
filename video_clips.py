"""Video Clips drop-in effect (formerly Video Showcase).

An audio-*reactive* clip montage: sequences short video clips with a crossfade
and ten rotating presentation styles, cover-fit with Ken-Burns drift, reacting to
the *system* audio (bass/mid/treble → zoom/palette/vignette).  It decodes video
frames only — the clip's own audio is not played (that is the Video Player
drop-in's job).

Directory selection (per activation)
------------------------------------
The videos directory (default ``drop-ins/video-clips-01/videos/`` or
``[effects.VideoClips].video_dir``) is scanned into **groups**:

- each immediate **subdirectory** containing videos is one group;
- **all loose videos** directly in the videos dir form **one shared group**.

On every activation the effect randomly picks **one group** and plays only that
group's clips for the run, shuffled (no repeats until the group is exhausted,
then reshuffled).  A single clip loops only when it is the only clip in the
chosen group.  The transition styles are freshly shuffled per activation too.
All randomisation uses ``self.rng`` so runs are independent.

Preload / memory
----------------
Clip metadata (fps/size/frame count) is probed once and cached per file; each
effect instance keeps its own capture + a single GPU texture per active clip, so
memory stays bounded.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import moderngl
import numpy as np

try:
    import cv2  # type: ignore
    _CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2 = False

from unicornviz.effects.base import BaseEffect, AudioData

log = logging.getLogger(__name__)

_VIDEO_EXTS = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'}

# Cache of probed, immutable clip metadata keyed by resolved file path.
_META_CACHE: dict[str, '_ClipMeta'] = {}


class _ClipMeta(NamedTuple):
    """Immutable, shareable metadata for a single clip file."""

    path: Path
    name: str
    fps: float
    frame_count: int
    size: tuple[int, int]


def scan_groups(video_dir: Path) -> list[list[Path]]:
    """Return playback groups for ``video_dir``.

    Each immediate subdirectory that contains ≥1 video becomes one group; all
    loose videos directly in ``video_dir`` form one additional shared group.
    Returns a list of non-empty groups (each a sorted list of file paths); the
    list is empty when no videos are found.  Pure/testable — no GL, no cv2.
    """
    if not video_dir.exists() or not video_dir.is_dir():
        return []

    groups: list[list[Path]] = []
    loose: list[Path] = []
    for entry in sorted(video_dir.iterdir(), key=lambda p: p.name.lower()):
        if entry.is_dir():
            vids = sorted(
                (p for p in entry.rglob('*')
                 if p.is_file() and p.suffix.lower() in _VIDEO_EXTS),
                key=lambda p: str(p).lower(),
            )
            if vids:
                groups.append(vids)
        elif entry.is_file() and entry.suffix.lower() in _VIDEO_EXTS:
            loose.append(entry)

    if loose:
        groups.append(loose)
    return groups


class _VideoClip:
    """Per-instance streaming state wrapping shared :class:`_ClipMeta`."""

    __slots__ = ('meta', 'texture', 'cap', 'last_frame_index', 'loaded')

    def __init__(self, meta: _ClipMeta) -> None:
        self.meta = meta
        self.texture: moderngl.Texture | None = None
        self.cap = None  # cv2.VideoCapture or None
        self.last_frame_index: int = -1
        self.loaded: bool = False


class _StylePreset(NamedTuple):
    name: str
    mix_shape: str


_STYLE_NAMES = [
    'soft', 'pulse', 'drift', 'flash', 'cinema',
    'halo', 'split', 'glow', 'dream', 'nervous',
]


_VERT = """
#version 330
in vec2 in_vert;
out vec2 v_uv;
void main() {
    v_uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

_FRAG = """
#version 330
uniform sampler2D texA;
uniform sampler2D texB;
uniform vec2 texASize;
uniform vec2 texBSize;
uniform vec2 iResolution;
uniform float iTime;
uniform float iBass;
uniform float iMid;
uniform float iTreble;
uniform float iMix;
uniform int iStyle;
uniform float iZoom;
in vec2 v_uv;
out vec4 fragColor;

vec3 palette(float t) {
    vec3 a = vec3(0.5);
    vec3 b = vec3(0.5);
    vec3 c = vec3(1.0);
    vec3 d = vec3(0.18, 0.10, 0.24);
    return a + b * cos(6.28318 * (c * t + d));
}

vec2 coverUV(vec2 uv, vec2 texSize, vec2 outSize) {
    float inAspect  = texSize.x / max(texSize.y, 1.0);
    float outAspect = outSize.x / max(outSize.y, 1.0);
    vec2 st = uv;
    if (inAspect > outAspect) {
        float s = outAspect / inAspect;
        st.x = (uv.x - 0.5) * s + 0.5;
    } else {
        float s = inAspect / outAspect;
        st.y = (uv.y - 0.5) * s + 0.5;
    }
    return st;
}

vec3 sampleSlide(sampler2D tex, vec2 uv, vec2 texSize, vec2 outSize,
                 float t, float drift, float zoom) {
    vec2 cuv = coverUV(uv, texSize, outSize);
    cuv = (cuv - 0.5) / zoom + 0.5;
    cuv += vec2(sin(t * 0.23 + drift), cos(t * 0.19 + drift)) * 0.03;
    return texture(tex, clamp(cuv, 0.0, 1.0)).rgb;
}

float styleBlend(float mixValue, vec2 uv, float noiseSeed) {
    if (iStyle == 0) {
        return smoothstep(0.0, 1.0, mixValue);
    } else if (iStyle == 1) {
        return smoothstep(0.08, 0.92, mixValue + sin(uv.y * 10.0) * 0.04);
    } else if (iStyle == 2) {
        return clamp(mixValue, 0.0, 1.0);
    } else if (iStyle == 3) {
        return step(0.5, mixValue);
    } else if (iStyle == 4) {
        return smoothstep(0.0, 1.0, mixValue * mixValue);
    } else if (iStyle == 5) {
        return smoothstep(0.12, 0.88, mixValue + 0.06 * sin(uv.x * 18.0));
    } else if (iStyle == 6) {
        return smoothstep(0.0, 1.0, mixValue + (noiseSeed - 0.5) * 0.06);
    } else if (iStyle == 7) {
        return smoothstep(0.0, 1.0, mixValue + 0.08 * sin(uv.x * 6.0 + uv.y * 4.0));
    } else if (iStyle == 8) {
        return smoothstep(0.0, 1.0, mixValue * 0.85 + 0.08 * sin(uv.y * 8.0));
    }
    return smoothstep(0.0, 1.0, mixValue + (noiseSeed - 0.5) * 0.12);
}

void main() {
    vec2 uv = v_uv;
    uv = (uv - 0.5) / max(iZoom, 0.01) + 0.5;
    float t = iTime;
    float stylePhase = float(iStyle) / 10.0;

    float zoomA = 1.0 + 0.04 * sin(t * (0.21 + stylePhase * 0.03)) + iBass * 0.04;
    float zoomB = 1.0 + 0.04 * cos(t * (0.17 + stylePhase * 0.02)) + iTreble * 0.04;

    vec3 a = sampleSlide(texA, uv, texASize, iResolution, t, 0.1, zoomA);
    vec3 b = sampleSlide(texB, uv, texBSize, iResolution, t, 1.4, zoomB);

    float wipe  = smoothstep(0.0, 1.0, iMix);
    float noise = fract(sin(dot(uv * 200.0, vec2(12.9898, 78.233))) * 43758.5453);
    float blend = styleBlend(wipe, uv, noise);

    vec3 col = mix(a, b, blend);
    col = mix(col, col * palette(t * (0.05 + stylePhase * 0.03) + uv.y * 0.25),
              0.06 + iMid * 0.07);

    vec2 p   = uv * 2.0 - 1.0;
    float vig = 1.0 - dot(p, p) * (0.16 + stylePhase * 0.08);
    col *= vig;

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


class VideoClips(BaseEffect):
    """Audio-reactive video-clip montage with per-activation group selection."""

    NAME = 'Video Clips'
    AUTHOR = 'unicorn-viz drop-in'
    TAGS = ['media', 'videos', 'slideshow']

    def _init(self) -> None:
        self.parameters = {
            'speed': float(self.config.get('speed', 1.0)),
            'mix_time': float(self.config.get('mix_time', 10.0)),
            'zoom': float(self.config.get('zoom', 1.0)),
        }
        self.scale_when_framed = bool(self.config.get('scale_when_framed', True))

        self._prog = self._make_program(_VERT, _FRAG)
        self._vao, self._vbo = self._fullscreen_quad(self._prog)

        self._timer = 0.0
        self._mix = 0.0
        self._bass = 0.0
        self._mid = 0.0
        self._treble = 0.0

        # Per-activation clip selection + shuffle (independent per instance).
        self._clips: list[_VideoClip] = self._select_group_clips()
        n = len(self._clips)
        self._clip_order: list[int] = list(self.rng.permutation(n)) if n else []
        self._order_pos = 0

        # Per-activation transition-style shuffle.
        self._style_order: list[int] = list(self.rng.permutation(len(_STYLE_NAMES)))
        self._style_index = 0

    def _select_group_clips(self) -> list[_VideoClip]:
        """Pick one directory group at random and build its clip list."""
        video_dir = self._video_dir(self.config)
        groups = scan_groups(video_dir)
        if not groups:
            log.warning('VideoClips: no videos found in %s', video_dir)
            return []
        gi = int(self.rng.integers(len(groups)))
        chosen = groups[gi]
        clips: list[_VideoClip] = []
        for path in chosen:
            try:
                clips.append(_VideoClip(self._probe_meta(path)))
            except Exception as exc:
                log.warning('VideoClips: failed to probe %s: %s', path, exc)
        log.info(
            'VideoClips: group %d/%d selected (%d clips) from %s',
            gi + 1, len(groups), len(clips), video_dir,
        )
        return clips

    @classmethod
    def _video_dir(cls, cfg: dict) -> Path:
        override = str(cfg.get('video_dir', '')).strip()
        if override:
            return Path(override).expanduser()
        return Path(__file__).resolve().parent / 'videos'

    @classmethod
    def _probe_meta(cls, path: Path) -> _ClipMeta:
        """Probe (and cache) immutable clip metadata."""
        key = str(path.resolve() if path.exists() else path)
        cached = _META_CACHE.get(key)
        if cached is not None:
            return cached
        if not _CV2:
            raise RuntimeError('opencv-python-headless is required for VideoClips')
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f'failed to open {path}')
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if fps <= 0.0:
            fps = 30.0
        if frame_count <= 0:
            frame_count = max(1, int(fps * 10.0))
        if width <= 0 or height <= 0:
            width, height = 1, 1
        meta = _ClipMeta(path=path, name=path.stem, fps=fps,
                         frame_count=frame_count, size=(width, height))
        _META_CACHE[key] = meta
        return meta

    def _order_index(self, offset: int) -> int:
        """Return the clip index at ``offset`` from the current order position."""
        if not self._clip_order:
            return 0
        return self._clip_order[(self._order_pos + offset) % len(self._clip_order)]

    def _ensure_loaded(self, clip: _VideoClip) -> None:
        if clip.loaded or not _CV2:
            return
        cap = cv2.VideoCapture(str(clip.meta.path))
        if not cap.isOpened():
            cap.release()
            log.warning('VideoClips: failed to open %s', clip.meta.path)
            return
        ok, frame = cap.read()
        if ok and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 0)
            tex = self.ctx.texture((frame.shape[1], frame.shape[0]), 3, frame.tobytes())
        else:
            w, h = clip.meta.size
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            tex = self.ctx.texture((w, h), 3, blank.tobytes())
        tex.filter = moderngl.LINEAR, moderngl.LINEAR
        clip.texture = tex
        clip.cap = cap
        clip.last_frame_index = 0
        clip.loaded = True

    def update(self, dt: float, audio: AudioData) -> None:
        super().update(dt, audio)
        self._bass = audio.bass
        self._mid = audio.mid
        self._treble = audio.treble

        if not self._clips:
            return

        self._timer += dt * max(0.1, self.parameters['speed'])
        mix_time = max(1.0, self.parameters['mix_time'])
        phase = self._timer / mix_time
        if audio.beat:
            phase = min(1.0, phase + 0.12)

        if phase >= 1.0:
            self._timer = 0.0
            self._mix = 0.0
            # Advance to the next clip in the shuffled order; reshuffle on wrap.
            self._order_pos += 1
            if self._order_pos >= len(self._clip_order):
                self._order_pos = 0
                if len(self._clip_order) > 1:
                    last = self._clip_order[-1]
                    self._clip_order = list(self.rng.permutation(len(self._clip_order)))
                    if self._clip_order[0] == last:
                        self._clip_order[0], self._clip_order[-1] = (
                            self._clip_order[-1], self._clip_order[0])
            # Advance the transition style; reshuffle on wrap.
            self._style_index += 1
            if self._style_index >= len(self._style_order):
                self._style_index = 0
                self._style_order = list(self.rng.permutation(len(_STYLE_NAMES)))
        else:
            self._mix = phase

        self._ensure_loaded(self._clips[self._order_index(0)])
        nxt = self._order_index(1)
        if nxt != self._order_index(0):
            self._ensure_loaded(self._clips[nxt])

    def _clip_frame_texture(self, clip: _VideoClip) -> moderngl.Texture | None:
        self._ensure_loaded(clip)
        if clip.texture is None:
            return None
        frame_count = max(1, clip.meta.frame_count)
        frame_index = int(
            self.time * clip.meta.fps * max(0.1, self.parameters['speed'])
        ) % frame_count
        cap = clip.cap
        if cap is not None and frame_index != clip.last_frame_index:
            if clip.last_frame_index < 0 or abs(frame_index - clip.last_frame_index) > 3:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                frame_index = 0
            if ok and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 0)
                clip.texture.write(frame.tobytes())
                clip.last_frame_index = frame_index
        return clip.texture

    def render(self) -> None:
        if not self._clips:
            return
        clip_a = self._clips[self._order_index(0)]
        clip_b = self._clips[self._order_index(1)]
        tex_a = self._clip_frame_texture(clip_a)
        tex_b = self._clip_frame_texture(clip_b)
        if tex_a is None or tex_b is None:
            return
        tex_a.use(location=0)
        tex_b.use(location=1)

        def set_uniform(name: str, value) -> None:
            try:
                self._prog[name].value = value
            except KeyError:
                pass

        set_uniform('texA', 0)
        set_uniform('texB', 1)
        set_uniform('texASize', tuple(float(v) for v in clip_a.meta.size))
        set_uniform('texBSize', tuple(float(v) for v in clip_b.meta.size))
        set_uniform('iResolution', (float(self.width), float(self.height)))
        set_uniform('iTime', self.time)
        set_uniform('iBass', self._bass)
        set_uniform('iMid', self._mid)
        set_uniform('iTreble', self._treble)
        set_uniform('iMix', self._mix)
        set_uniform('iStyle', int(self._style_order[self._style_index]))
        set_uniform('iZoom', float(self.parameters['zoom']))
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def destroy(self) -> None:
        for clip in self._clips:
            if clip.cap is not None:
                clip.cap.release()
                clip.cap = None
            if clip.texture is not None:
                clip.texture.release()
                clip.texture = None
            clip.loaded = False
            clip.last_frame_index = -1
        self._vao.release()
        self._vbo.release()
        self._prog.release()

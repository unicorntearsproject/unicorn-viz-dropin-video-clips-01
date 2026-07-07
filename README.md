# video-clips-01 — Video Clips

An audio-**reactive** video-clip montage drop-in for
[unicorn-viz](https://github.com/iDoMeteor/unicorn-viz). Formerly "Video
Showcase".

## Description

Sequences short video clips with a crossfade and ten rotating presentation
styles, cover-fit with a slow Ken-Burns drift. It reacts to the **system** audio
(bass / mid / treble drive zoom, palette shift and vignette; beats nudge the
crossfade). It decodes **video frames only** — a clip's own soundtrack is not
played. For whole videos with their own audio, use the **Video Player** drop-in
(`videos-01`).

## Directory selection (per activation)

Point it at a folder of videos (default `videos/`, or set
`[effects.VideoClips].video_dir`). The folder is scanned into **groups**:

- each **subdirectory** containing videos is one group;
- **all loose videos** directly in the folder form **one shared group**.

Every time the effect activates it randomly picks **one group** and plays only
that group's clips for the run — shuffled, with no repeats until the group is
exhausted, then reshuffled. A single clip loops only when it is the only clip in
the chosen group. The transition styles are freshly shuffled each activation too.

## Adding clips

Drop `.mp4` / `.mov` / `.mkv` / `.webm` / `.avi` / `.m4v` files into `videos/`
(optionally in themed subfolders). They are picked up on the next run.

## Config

```toml
[effects.VideoClips]
speed     = 1.0     # playback + drift rate
mix_time  = 10.0    # seconds per clip before the crossfade
zoom      = 1.0     # base zoom
# video_dir = "/path/to/clips"   # override the bundled videos/ dir
```

## Dependencies

- `opencv-python-headless` (video frame decoding; shared with the core install).

## Hotkeys

None dedicated. Standard `+`/`-` adjust `speed`.

## Troubleshooting

- **No video shows** — confirm the folder has recognized files and that
  `opencv-python-headless` is installed. Check the log for `VideoClips: no
  videos found`.
- **Always the same set** — that's by design: one group is chosen per
  activation. Switch away and back (or let the playlist cycle) to re-roll.

## Tags

`media` · `videos` · `slideshow`

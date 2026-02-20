

python voidstar_anim_mp4.py input.png --preset v5 --width 1920 \
  --twinkle 0 \
  --glow-scale 0.82 --pulse-center-mix 0 \
  --arcs 34 --arc-strength 0.70 --micro-arc-strength 0.22 --halo-blur 1.6 --halo-intensity 0.90 --alpha-halo 125 \
  --events 900 --event-sigma 0.055 --trail-frames 9 --trail-decay 0.84 --alpha-fire 120 \
  --pulse-sigma 0.022 --pulse-blur 1.0 --pulse-range 0.90 --pulse-level 10 --alpha-pulse 14 --pulse-intensity 0.70



python voidstar_anim_mp4.py input.png --preset v5 --width 1920 \
  --glow-scale 0.78 --alpha-halo 105 --halo-intensity 0.80 \
  --alpha-fire 95 --trail-decay 0.70 \
  --pulse-level 22 --alpha-pulse 26


python voidstar_anim_mp4.py input.png --out-dir ~/WinVideos/neurons/ \
  --pulse-range 0.45 \
  --frames 200


python voidstar_anim_gif.py input.png --pulse-range .3 \
 --pulse-sigma 0.05 --pulse-intensity 1.0 --pulse-level 32 --pulse-blur 2.0 \
 --out-dir ~/WinVideos/neurons/ --format mp4 --auto-name 

 python voidstar_anim_gif.py input.png --preset v7 --frame-ms 70 \
  --out-dir ~/WinVideos/neurons/ --format mp4 --auto-name 


python voidstar_anim_gif.py input.png --out out.gif --glow-scale 0.75
python voidstar_anim_gif.py input.png --pulse-level 200 --pulse-sigma 0.045 --alpha-pulse 30
python voidstar_anim_gif.py input.png --pulse-range 1.2
python voidstar_anim_gif.py input.png --trail-frames 7 --trail-decay 0.80
##python voidstar_anim_gif.py input.png --crawl-x-freq 4


python voidstar_morph_loop.py left.png right.png --seconds 6 --fps 30 #--output voidstar_neurons_loop.mp4






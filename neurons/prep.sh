# input: side_by_side.png  (left|right)
# ffmpeg -y -i voidstar_neurons_side_by_side.png -vf "crop=iw/2:ih:0:0" left.png
# ffmpeg -y -i voidstar_neurons_side_by_side.png -vf "crop=iw/2:ih:iw/2:0" right.png

PAD=10
ffmpeg -y -i voidstar_neurons_side_by_side.png -vf "crop=iw/2-${PAD}:ih:${PAD}:0" left.png
ffmpeg -y -i voidstar_neurons_side_by_side.png -vf "crop=iw/2-${PAD}:ih:iw/2+${PAD}:0" right.png



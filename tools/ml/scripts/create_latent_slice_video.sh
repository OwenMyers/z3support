ffmpeg -framerate 50 -i slice_%00d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4

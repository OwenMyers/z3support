#!/bin/bash


cd figures/lattices;

ffmpeg -start_number 0\
    -r 1\
    -i lattice_%d.png\
    -s:v 1280x720\
    -pix_fmt yuv420p\
    -r 10\
    -vcodec mpeg4 test.mp4;

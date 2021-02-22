#!/usr/bin/env bash

# convert video with mask frames to images
ffmpeg -i mobile-duo.mp4 -vf fps=8,crop=1080:1080:0:480,scale=540:540 mask_on/mask-duo-mobile-%d.png
ffmpeg -i mobile-camo.mp4 -vf fps=8,crop=1080:1080:0:480,scale=540:540 mask_on/mask-camo-mobile-%d.png
ffmpeg -i mobile-black.mp4 -vf fps=8,crop=1080:1080:0:480,scale=540:540 mask_on/mask-black-mobile-%d.png

# convert video with no mask frames to images
ffmpeg -i mobile-none.mp4 -vf fps=16,crop=1080:1080:0:480,scale=540:540 mask_off/mask-none-mobile-%d.png

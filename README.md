# About
This is a simple tool which processes long video footage taken and extracts only the section where there is detected motion. This tool assumes the footage is taken on a tripod and majority of the pixels do not change from frame to frame. 

# Install steps
 1) `python -m venv venv`
 1) `source venv/bin/activate`
 1) `pip install --upgrade pip`
 1) - `pip install -r requirements.txt`
    <br>or
    - `pip install plotly numpy opencv-python tqdm`
 1) `python boris1.py video.mp4`

# Usage
1) Find the correct motion threshold. This may be different from video to video based on wind, bird sizes, etc. 
    1)Test on a small portion of the video with `python bird_cutter.py test4k.mp4 --start_time 120 --end_time 300`. This portion should contain a motion less time frame so that we can determine the noise floor. 
    1)Check the generated plot `difference_sum_plot_interactive.html` of motion and compare it with your threshold. The threashold line should be just above detected motion when there is no bird activity. 
    1)Increase or lower the threshold using the `--sensitivity` option. Example: `python bird_cutter.py test4k.mp4 --start_time 120 --end_time 300 --sensitivity 0.5`. The sensitivity specify the percentage of pixel value change between frames. You can start with sensitivity of `0.2` for `1080p` and `0.5` for `4k` source video. 
2) When satisfied with the selected sensitivity, you can process the entire source video with `python bird_cutter.py test4k.mp4`.

# Issues
- This is just a proof of concept
- Low performance
- Many of the options don't work as intended
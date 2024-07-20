import cv2
import subprocess
import os
import argparse
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
from collections import deque
from datetime import timedelta

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process video for bird activity.')
    parser.add_argument('--video_path', type=str, help='Path to the input video file (used for both scanning and cutting if --input_hires is not provided)')
    parser.add_argument('--input_lowres', type=str, help='Path to the input low resolution video file')
    parser.add_argument('--input_hires', type=str, help='Path to the input high resolution video file')
    parser.add_argument('--output_dir', type=str, help='Directory to save output files')
    parser.add_argument('--sensitivity', type=float, default=0.5, help='Sensitivity for motion detection in percentage (0-100) to calculate motion threshold. Examples: 1080p = 0.2, 4k = 0.5)')
    parser.add_argument('--buffer_before', type=int, default=3, help='Seconds before motion event')
    parser.add_argument('--buffer_after', type=int, default=3, help='Seconds after motion event')
    parser.add_argument('--min_interval', type=int, default=5, help='Minimum seconds between segments to keep them separate')
    parser.add_argument('--min_avg_diff_sum', type=int, default=500, help='Minimum average difference sum to keep a segment')
    parser.add_argument('--example', action='store_true', help='Show an example of the CLI syntax and exit')
    parser.add_argument('--check_frame', type=int, help='Specify the nth frame to check for differences')
    parser.add_argument('--start_time', type=float, default=0.0, help='Specify the starting time in seconds for processing')
    parser.add_argument('--end_time', type=float, help='Specify the ending time in seconds for processing')
    parser.add_argument('--smoothing', type=int, default=0, help='Enable smoothing with a window size in frames (0 to disable, 30 to average over 30 frames, etc.)')
    parser.add_argument('--resize_factor', type=float, default=0.1, help='Resize factor for processing. Example: 0.5 for 50% of original resolution')
    parser.add_argument('--motion_threshold', type=int, help='Threshold for motion detection in pixel value sum. Examples: 1080p = 50000, 4k = 500000')

    args = parser.parse_args()

    if args.example:
        print("Example CLI syntax:")
        print("Single video input:")
        print("python process_video.py --video_path 'path/to/video.mp4' --output_dir 'path/to/output dir' --sensitivity 5.0 --buffer_before 3 --buffer_after 3 --min_interval 5 --min_avg_diff_sum 50000 --check_frame 30 --start_time 60.0 --end_time 180.0 --smoothing 30")
        print("\nSeparate low-res and high-res input:")
        print("python process_video.py --input_lowres 'path/to/lowres_video.mp4' --input_hires 'path/to/hires_video.mp4' --output_dir 'path/to/output dir' --sensitivity 5.0 --buffer_before 3 --buffer_after 3 --min_interval 5 --min_avg_diff_sum 50000 --check_frame 30 --start_time 60.0 --end_time 180.0 --smoothing 30")
        exit(0)

    if not args.video_path and not (args.input_lowres and args.input_hires):
        parser.error("Either --video_path or both --input_lowres and --input_hires must be provided")

    return args

def detect_motion(frame1, frame2, motion_threshold, resize_factor):
    if resize_factor != 1:
        frame1 = cv2.resize(frame1, (0, 0), fx=resize_factor, fy=resize_factor)
        frame2 = cv2.resize(frame2, (0, 0), fx=resize_factor, fy=resize_factor)
    
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray1, gray2)
    
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    diff_sum = np.sum(thresh)
    
    return diff_sum, diff_sum > motion_threshold

def smooth_diffs(diff_sums, window_size):
    smoothed_diffs = []
    window = deque(maxlen=window_size)
    for diff in diff_sums:
        window.append(diff)
        smoothed_diffs.append(np.mean(window))
    return smoothed_diffs

def format_time(seconds):
    return str(timedelta(seconds=seconds))

def create_plot(log_file_path, output_dir, motion_threshold, filtered_segments, fps):
    try:
        frame_numbers = []
        diff_sums = []
        
        with open(log_file_path, 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                try:
                    frame_number, diff_sum = map(float, line.strip().split(',')[:2])
                    frame_numbers.append(frame_number)
                    diff_sums.append(diff_sum)
                except ValueError:
                    print(f"Skipping invalid line: {line.strip()}")

        times = [frame_number / fps for frame_number in frame_numbers]

        plt.figure(figsize=(12, 6))
        plt.plot(frame_numbers, diff_sums, label='Difference Sum', color='blue')

        for (start_time, end_time) in filtered_segments:
            plt.axvline(x=start_time * fps, color='green', linestyle='--', label='Segment Start' if plt.gca().get_legend_handles_labels()[1].count('Segment Start') == 0 else "")
            plt.axvline(x=end_time * fps, color='red', linestyle='--', label='Segment End' if plt.gca().get_legend_handles_labels()[1].count('Segment End') == 0 else "")

        plt.axhline(y=motion_threshold, color='gray', linestyle='solid', label='Motion Threshold')
        
        xticks = plt.xticks()[0]
        xlabels = [f"{int(tick)}\n({format_time(tick / fps)})" for tick in xticks]
        plt.xticks(ticks=xticks, labels=xlabels)
        
        plt.xlabel('Frame Number (Time in hh:mm:ss)')
        plt.ylabel('Difference Sum')
        plt.title('Motion Detection')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_file_path = os.path.join(output_dir, 'motion_detection_plot.png')
        plt.savefig(plot_file_path)
        plt.close()

        print(f"Plot saved as {plot_file_path}")

        times_formatted = [format_time(time) for time in times]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frame_numbers, y=diff_sums, mode='lines', name='Difference Sum',
                                 hovertemplate='<br><b>Frame:</b> %{x}<br><b>Time:</b> %{text}<br><b>Diff Sum:</b> %{y}',
                                 text=times_formatted))
        fig.add_trace(go.Scatter(x=[0, max(frame_numbers)], y=[motion_threshold, motion_threshold], mode='lines', name='Motion Threshold', line=dict(color='gray', dash='solid')))

        segment_number = 0
        for (start_time, end_time) in filtered_segments:
            fig.add_vline(x=start_time * fps, line=dict(color='green', dash='dash'), annotation_text=f'Start {segment_number}', annotation_position='top left')
            fig.add_vline(x=end_time * fps, line=dict(color='red', dash='dash'), annotation_text=f'End {segment_number}', annotation_position='top right')

            segment_number += 1

        fig.update_layout(
            title='Frame Difference Sum Over Time',
            xaxis_title='Frame Number (Time in hh:mm:ss)',
            yaxis_title='Difference Sum',
            legend_title='Legend',
            autosize=True,
            hovermode='x unified',
            xaxis=dict(
                tickmode='array',
                tickvals=frame_numbers[::len(frame_numbers)//10],
                ticktext=[f"{int(f)} ({format_time(f / fps)})" for f in frame_numbers[::len(frame_numbers)//10]]
            )
        )

        interactive_plot_file_path = os.path.join(output_dir, 'motion_detection_plot_interactive.html')
        fig.write_html(interactive_plot_file_path)
        print(f"Interactive plot saved as {interactive_plot_file_path}")

    except Exception as e:
        print(f"Error creating plot: {e}")

def create_unique_output_dir(base_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = f"{base_dir}_{timestamp}"
    os.makedirs(new_dir, exist_ok=True)
    print(f"Output directory created: {new_dir}")  
    return new_dir

def main():
    args = parse_arguments()

    global motion_threshold
    if args.video_path:
        lowres_video_path = args.video_path
        hires_video_path = args.video_path
    else:
        lowres_video_path = args.input_lowres
        hires_video_path = args.input_hires

    output_dir = args.output_dir or os.path.splitext(os.path.basename(lowres_video_path))[0]
    sensitivity = args.sensitivity
    buffer_before_seconds = args.buffer_before
    buffer_after_seconds = args.buffer_after
    min_interval_seconds = args.min_interval
    min_avg_diff_sum = args.min_avg_diff_sum
    check_frame_interval = args.check_frame
    start_time = args.start_time
    end_time = args.end_time
    smoothing_window = args.smoothing
    resize_factor = args.resize_factor
    motion_threshold = args.motion_threshold

    output_dir = create_unique_output_dir(output_dir)

    log_file_path = os.path.join(output_dir, 'motion_log.csv')
    print(f"Log file path: {log_file_path}")  

    cap = cv2.VideoCapture(lowres_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {lowres_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_pixels = width * height
    motion_threshold = motion_threshold or int(sensitivity * total_pixels)
    print(f"Total frames: {total_frames}, Frame size: {width}x{height}, FPS: {fps}, Motion threshold: {motion_threshold}")  

    if end_time is not None and end_time * fps > total_frames:
        print(f"Error: --end_time must be less than or equal to {total_frames / fps:.2f} seconds.")
        cap.release()
        return

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time is not None else total_frames

    print(f"Processing frames from {start_frame} to {end_frame}")

    start_processing_time = time.time()

    try:
        detected_segments = []
        prev_frame = None
        diff_buffer = []
        buffer_size = int(fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        log_file = open(log_file_path, 'w')
        log_file.write('frame_number,difference_sum,motion_detected\n')

        with tqdm(total=end_frame - start_frame, desc="Processing Frames") as pbar:
            for frame_number in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    print(f"Error reading frame {frame_number}.")
                    break

                if prev_frame is not None:
                    diff_sum, motion_detected = detect_motion(prev_frame, frame, motion_threshold, resize_factor)

                    if smoothing_window > 0:
                        diff_buffer.append(diff_sum)
                        if len(diff_buffer) > smoothing_window:
                            diff_buffer.pop(0)
                        smoothed_diff_sum = np.mean(diff_buffer)
                    else:
                        smoothed_diff_sum = diff_sum

                    log_file.write(f"{frame_number},{smoothed_diff_sum},{motion_detected}\n")

                    if motion_detected:
                        start_time = max(0, (frame_number - buffer_before_seconds * fps) / fps)
                        end_time = min(total_frames / fps, (frame_number + buffer_after_seconds * fps) / fps)
                        detected_segments.append((start_time, end_time))

                prev_frame = frame
                pbar.update(1)
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        log_file.close()
        cap.release()

    print(f"Detected {len(detected_segments)} motion segments.")

    merged_segments = []
    last_end_time = None
    for start_time, end_time in detected_segments:
        if last_end_time is None or start_time > last_end_time + min_interval_seconds:
            merged_segments.append((start_time, end_time))
        else:
            merged_segments[-1] = (merged_segments[-1][0], max(last_end_time, end_time))
        last_end_time = merged_segments[-1][1]

    print(f"Merged {len(merged_segments)} segments: {merged_segments}")

    filtered_segments = []
    try:
        with open(log_file_path, 'r') as log_file:
            lines = log_file.readlines()

        for start_time, end_time in merged_segments:
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            diff_sums_segment = []
            for line in lines:
                if line.strip() == 'frame_number,difference_sum,motion_detected':
                    continue
                frame_num, diff_sum = map(int, line.strip().split(',')[:2])
                if start_frame <= frame_num <= end_frame:
                    diff_sums_segment.append(diff_sum)

            if diff_sums_segment:
                avg_diff_sum = np.mean(diff_sums_segment)
                if avg_diff_sum >= min_avg_diff_sum:
                    filtered_segments.append((start_time, end_time))

        print(f"Filtered {len(filtered_segments)} segments: {filtered_segments}")
    except Exception as e:
        print(f"Error reading log file: {e}")

    print("\nSegment Details:")
    segment_details = []
    header_format = "{:<20} {:<20} {:<20} {:<20} {:<20} {:<20}"
    row_format = "{:<20.2f} {:<20.2f} {:<20.2f} {:<20} {:<20} {:<20}"
    print(header_format.format('Start Time (s)', 'End Time (s)', 'Duration (s)', 'Avg Diff Sum', 'Min Diff Sum', 'Max Diff Sum'))

    try:
        with open(log_file_path, 'r') as log_file:
            lines = log_file.readlines()

        for start_time, end_time in filtered_segments:
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            duration = end_time - start_time

            diff_sums_segment = []
            for line in lines:
                if line.strip() == 'frame_number,difference_sum,motion_detected':
                    continue
                frame_num, diff_sum = map(int, line.strip().split(',')[:2])
                if start_frame <= frame_num <= end_frame:
                    diff_sums_segment.append(diff_sum)

            if diff_sums_segment:
                avg_diff_sum = np.mean(diff_sums_segment)
                min_diff_sum = np.min(diff_sums_segment)
                max_diff_sum = np.max(diff_sums_segment)
            else:
                avg_diff_sum = min_diff_sum = max_diff_sum = 0

            segment_info = row_format.format(start_time, end_time, duration, avg_diff_sum, min_diff_sum, max_diff_sum)
            segment_details.append(segment_info)
            print(segment_info)
    except Exception as e:
        print(f"Error processing log file: {e}")

    details_file_path = os.path.join(output_dir, 'segment_details.txt')
    print(f"Details file path: {details_file_path}")

    try:
        with open(details_file_path, 'w') as details_file:
            details_file.write(header_format.format('Start Time (s)', 'End Time (s)', 'Duration (s)', 'Avg Diff Sum', 'Min Diff Sum', 'Max Diff Sum') + '\n')
            for detail in segment_details:
                details_file.write(detail + '\n')
    except Exception as e:
        print(f"Error writing details file: {e}")

    print("Extracting video segments...")
    segment_files = []
    for i, (start_time, end_time) in enumerate(filtered_segments):
        segment_file = os.path.join(output_dir, f'segment_{i:03d}.mp4')
        segment_files.append(segment_file)
        command = [
            'ffmpeg',
            '-ss', str(start_time),
            '-to', str(end_time),
            '-i', hires_video_path,
            '-c', 'copy',
            '-fflags', '+genpts',
            segment_file
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"Error extracting segment {i+1}: {result.stderr.decode()}")
        else:
            print(f"Extracted segment {i+1}/{len(filtered_segments)}")

    print("Creating file list for concatenation...")
    file_list_path = os.path.join(output_dir, 'file_list.txt')
    try:
        with open(file_list_path, 'w') as file_list:
            for segment_file in segment_files:
                file_list.write(f"file '{segment_file}'\n")
    except Exception as e:
        print(f"Error writing file list for concatenation: {e}")

    print("Combining video segments...")
    final_output = os.path.join(output_dir, 'final_output.mp4')
    result = subprocess.run([
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', file_list_path,
        '-c', 'copy',
        '-fflags', '+genpts',
        final_output
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Print debug information
    print(result.stderr.decode())

    create_plot(log_file_path, output_dir, motion_threshold, filtered_segments, fps)

    end_processing_time = time.time()
    print(f'Video processing complete. Check {final_output} and the details in {details_file_path}.')
    print(f"Total Processing Time: {end_processing_time - start_processing_time:.2f} seconds")

if __name__ == "__main__":
    main()

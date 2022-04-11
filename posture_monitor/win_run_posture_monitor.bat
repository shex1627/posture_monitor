cd C:\Users\alistar\Desktop\ds\posture_gcp\posture_monitor
:: call conda activate posture_monitor
:: start python posture_monitor_1.py --window_stay_top --track_basic_landmark --window_size large
:: start python keyboard_monitor.py --window_stay_top --camera_index 1 
call conda activate posture_monitor
call python posture_monitor_1.py --window_stay_top --track_basic_landmark --window_size small
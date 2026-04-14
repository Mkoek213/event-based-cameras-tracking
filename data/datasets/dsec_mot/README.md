DSEC-MOT local storage layout

This directory stores the final dataset used in this project:

- DSEC raw files for the 12 sequences used by DSEC-MOT
- DSEC-MOT annotation files

Per sequence, place the following files directly in the sequence directory:

- events_left.zip
- images_rectified_left.zip
- calibration.zip
- image_timestamps.txt
- image_exposure_timestamps_left.txt

Place the DSEC-MOT annotation .txt files in:

- annotations/train/
- annotations/test/

Expected structure:

- train/
- test/
- annotations/train/
- annotations/test/

Train sequences:

- interlaken_00_a
- interlaken_00_b
- interlaken_00_c
- interlaken_00_e
- zurich_city_01_d
- zurich_city_01_e
- zurich_city_04_b
- zurich_city_09_c
- zurich_city_09_d
- zurich_city_14_b

Test sequences:

- interlaken_00_d
- zurich_city_00_b

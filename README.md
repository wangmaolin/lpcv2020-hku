# LPCV2020 Contest

## Board Setup

1. flash PYNQ V2.5 to the SD card
2. upgrade the board using <https://github.com/Xilinx/DPU-PYNQ>

## Test Different Models

1. Rename the util-modelXX.py to utils.py and put it on the board together with model file: modelXX.elf.
2. Run test_accuracy_latency.ipynb to get the latency and accuracy of the tested model.
3. change the threadnum in utils.py to use different number of threads.

## Image files

The ImageNet validation set should be put under ./val which contains ILSVRC2012_val_00000001.JPEG, ILSVRC2012_val_00000002.JPEG, ..., ILSVRC2012_val_00050000.JPEG.
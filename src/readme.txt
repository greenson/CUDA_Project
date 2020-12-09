BMH running instruction:
compilation: nvcc -o bmh bmh.cu
usage: ./bmh target who pattern_mode pattern_path
target = path to the target text file
who = 0: sequential code on CPU, 1: Single-GPU execution, 2: Multi-GPU execution
pattern input mode = 0: read-in by command line, 1: read-in by file
pattern_path = (if choose file input) pattern file path

